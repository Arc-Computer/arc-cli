"""Arc analyze command - analyze failures from a run."""

import asyncio
from typing import Optional, List, Dict
from datetime import datetime, timedelta

import click
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from sqlalchemy import text

from arc.cli.utils import ArcConsole, CLIState, format_error, format_success
from arc.cli.utils import db_manager, HybridState

console = ArcConsole()
# Will be initialized based on database availability
state = None


async def _initialize_state():
    """Initialize state with database connection if available."""
    global state
    
    # Check if database is available
    db_connected = db_manager.is_available
    if not db_connected:
        db_connected = await db_manager.initialize()
    
    if db_connected:
        state = HybridState(db_connected=True)
    else:
        state = CLIState()
    
    return db_connected


@click.command()
@click.option('--run', 'run_id', help='Specific run ID to analyze (default: last run)')
@click.option('--historical', is_flag=True, help='Include historical failure patterns from database')
@click.option('--days', default=30, help='Days of history to include (default: 30)')
@click.option('--json', 'json_output', is_flag=True, help='Output JSON instead of rich text')
def analyze(run_id: Optional[str], historical: bool, days: int, json_output: bool):
    """Analyze failures from the last run.
    
    This command:
    1. Loads results from the specified or last run
    2. Clusters failures by pattern
    3. Identifies root causes
    4. Shows actionable insights
    5. Optionally includes historical failure patterns from database
    
    Example:
        arc analyze
        arc analyze --run run_20240110_143022_abc123
        arc analyze --historical --days 7
    """
    try:
        # Run async analysis
        asyncio.run(_analyze_async(run_id, historical, days, json_output))
    except Exception as e:
        if json_output:
            import json
            print(json.dumps({"error": str(e)}, indent=2))
        else:
            console.print(format_error(f"Analysis failed: {str(e)}"))
        raise click.exceptions.Exit(1)


async def _analyze_async(run_id: Optional[str], historical: bool, days: int, json_output: bool):
    """Perform async analysis with database queries."""
    # Initialize state
    db_connected = await _initialize_state()
    
    # Load run results
    run_result = state.get_run(run_id)
    if not run_result:
        if json_output:
            import json
            print(json.dumps({"error": "No run found"}, indent=2))
        else:
            console.print(format_error("No run found. Run 'arc run' first."))
        raise click.exceptions.Exit(1)
    
    # Perform analysis
    failures = run_result.failures
    
    # Initialize analysis data
    analysis = {
        "run_id": run_result.run_id,
        "total_failures": len(failures),
        "failure_rate": 1 - run_result.reliability_score if failures else 0,
        "database_connected": db_connected
    }
    
    # Get historical data if requested and database is available
    historical_patterns = None
    cross_run_insights = None
    
    if historical and db_connected:
        db_client = db_manager.get_client()
        if db_client:
            if not json_output:
                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    console=console
                ) as progress:
                    progress.add_task("Analyzing historical patterns...", total=None)
                    
                    historical_patterns = await _get_historical_patterns(db_client, days)
                    cross_run_insights = await _get_cross_run_insights(db_client, run_result.config_path, days)
            else:
                historical_patterns = await _get_historical_patterns(db_client, days)
                cross_run_insights = await _get_cross_run_insights(db_client, run_result.config_path, days)
    
    if not failures:
        analysis["message"] = "No failures to analyze"
        analysis["clusters"] = []
        analysis["top_issues"] = []
        
        if json_output:
            import json
            print(json.dumps(analysis, indent=2))
        else:
            console.print(format_success("✓ No failures found in this run!"))
            console.print(f"Reliability: {run_result.reliability_percentage:.1f}%")
            
            # Show historical context if available
            if historical_patterns:
                console.print()
                console.print("[primary]Historical Context[/primary]")
                console.print(f"This is better than {historical_patterns.get('percentile', 0):.0f}% of runs in the last {days} days")
        return
    
    # Cluster failures
    failure_clusters = _cluster_failures(failures)
    
    # Enhance with historical data if available
    if historical_patterns:
        failure_clusters = _enhance_with_historical_data(failure_clusters, historical_patterns)
    
    # Update analysis
    analysis["clusters"] = failure_clusters
    analysis["top_issues"] = _identify_top_issues(failure_clusters, historical_patterns)
    
    if historical_patterns:
        analysis["historical_patterns"] = historical_patterns
    if cross_run_insights:
        analysis["cross_run_insights"] = cross_run_insights
    
    # Save analysis
    state.save_analysis(run_result.run_id, analysis)
    
    # Save to database if available
    if db_connected and isinstance(state, HybridState):
        db_client = db_manager.get_client()
        if db_client:
            try:
                await _save_analysis_to_database(db_client, run_result, analysis)
            except Exception as e:
                console.print(f"Warning: Failed to save analysis to database: {str(e)}", style="warning")
    
    # Display results
    if json_output:
        import json
        print(json.dumps(analysis, indent=2))
    else:
        _display_analysis_ui(run_result, analysis, historical_patterns, cross_run_insights)


async def _get_historical_patterns(db_client, days: int) -> dict:
    """Get historical failure patterns from database."""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    # Get overall failure statistics
    stats_query = text("""
    SELECT 
        COUNT(DISTINCT s.simulation_id) as total_runs,
        COUNT(CASE WHEN o.status = 'error' THEN 1 END) as total_failures,
        AVG(s.overall_score) as avg_reliability,
        PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY s.overall_score) as median_reliability
    FROM simulations s
    JOIN outcomes o ON s.simulation_id = o.simulation_id
    WHERE s.created_at >= :start_date AND s.created_at <= :end_date
    """)
    
    # Get failure patterns
    patterns_query = text("""
    SELECT 
        o.error_category,
        COUNT(*) as occurrence_count,
        COUNT(DISTINCT s.simulation_id) as affected_runs,
        AVG(o.cost_usd) as avg_cost,
        STRING_AGG(DISTINCT c.name, ', ' ORDER BY c.name) as affected_configs
    FROM outcomes o
    JOIN simulations s ON o.simulation_id = s.simulation_id
    JOIN config_versions cv ON s.config_version_id = cv.version_id
    JOIN configurations c ON cv.config_id = c.config_id
    WHERE o.status = 'error' 
        AND s.created_at >= :start_date 
        AND s.created_at <= :end_date
        AND o.error_category IS NOT NULL
    GROUP BY o.error_category
    ORDER BY occurrence_count DESC
    """)
    
    async with db_client.engine.begin() as conn:
        stats_result = await conn.execute(stats_query, {"start_date": start_date, "end_date": end_date})
        stats = stats_result.fetchone()
        
        patterns_result = await conn.execute(patterns_query, {"start_date": start_date, "end_date": end_date})
        patterns = []
        for row in patterns_result:
            patterns.append({
                "category": row[0],
                "count": row[1],
                "affected_runs": row[2],
                "avg_cost": float(row[3] or 0),
                "configs": row[4].split(", ") if row[4] else []
            })
    
    return {
        "period_days": days,
        "total_runs": stats[0] or 0,
        "total_failures": stats[1] or 0,
        "avg_reliability": float(stats[2] or 0),
        "median_reliability": float(stats[3] or 0),
        "patterns": patterns
    }


async def _get_cross_run_insights(db_client, config_path: str, days: int) -> dict:
    """Get insights across multiple runs for the same config."""
    config_name = config_path.split("/")[-1]
    
    # Use string formatting for interval since PostgreSQL doesn't allow parameters in INTERVAL
    query = text(f"""
    SELECT
        s.simulation_name,
        s.created_at,
        s.overall_score,
        COUNT(CASE WHEN o.status = 'error' THEN 1 END) as failure_count,
        STRING_AGG(DISTINCT o.error_category, ', ' ORDER BY o.error_category) as error_categories
    FROM simulations s
    JOIN config_versions cv ON s.config_version_id = cv.version_id
    JOIN configurations c ON cv.config_id = c.config_id
    JOIN outcomes o ON s.simulation_id = o.simulation_id
    WHERE c.name = :config_name
        AND s.created_at >= NOW() - INTERVAL '{days} days'
    GROUP BY s.simulation_id, s.simulation_name, s.created_at, s.overall_score
    ORDER BY s.created_at DESC
    LIMIT 10
    """)

    async with db_client.engine.begin() as conn:
        result = await conn.execute(query, {"config_name": config_name})
        
    runs = []
    for row in result:
        runs.append({
            "run_id": row[0],
            "timestamp": row[1].isoformat(),
            "reliability": float(row[2] or 0),
            "failures": row[3],
            "error_types": row[4].split(", ") if row[4] else []
        })
    
    # Calculate trends
    if len(runs) >= 2:
        recent_reliability = runs[0]["reliability"]
        previous_reliability = sum(r["reliability"] for r in runs[1:4]) / min(3, len(runs) - 1)
        trend = "improving" if recent_reliability > previous_reliability else "declining" if recent_reliability < previous_reliability else "stable"
    else:
        trend = "insufficient_data"
    
    return {
        "config_name": config_name,
        "recent_runs": runs,
        "trend": trend
    }


async def _save_analysis_to_database(db_client, run_result, analysis):
    """Save analysis results to database."""
    # This would save to a dedicated analysis table if it exists
    # For now, we'll just log that it would be saved
    pass


def _enhance_with_historical_data(clusters: List[Dict], historical_patterns: dict) -> List[Dict]:
    """Enhance failure clusters with historical context."""
    historical_by_category = {p["category"]: p for p in historical_patterns.get("patterns", [])}
    
    for cluster in clusters:
        # Map cluster pattern to error category
        category = _pattern_to_category(cluster["pattern"])
        if category in historical_by_category:
            hist = historical_by_category[category]
            cluster["historical_frequency"] = hist["count"]
            cluster["historical_affected_runs"] = hist["affected_runs"]
            cluster["historical_avg_cost"] = hist["avg_cost"]
            
            # Adjust impact based on historical frequency
            if hist["affected_runs"] > 10:
                cluster["impact"] = "High"
            elif cluster["impact"] == "Low" and hist["affected_runs"] > 5:
                cluster["impact"] = "Medium"
    
    return clusters


def _pattern_to_category(pattern: str) -> str:
    """Map failure pattern to error category."""
    mapping = {
        "Currency Assumption Violation": "currency_assumption",
        "Timeout/Performance Issue": "timeout",
        "Tool Execution Error": "tool_error",
        "External API Failure": "api_error"
    }
    return mapping.get(pattern, "other")


def _display_analysis_ui(run_result, analysis, historical_patterns, cross_run_insights):
    """Display analysis results in rich UI."""
    console.print()
    console.print_header(f"Failure Analysis for {run_result.config_path}")
    
    console.print_metric("Run ID", run_result.run_id, style="muted")
    console.print_metric("Total failures", f"{analysis['total_failures']} / {run_result.scenario_count}")
    console.print_metric("Failure rate", f"{analysis['failure_rate']:.1%}")
    
    # Show historical context if available
    if historical_patterns:
        percentile = (1 - (run_result.reliability_score / historical_patterns["avg_reliability"])) * 100
        console.print_metric("Historical comparison", 
                           f"{'Better' if percentile > 50 else 'Worse'} than {abs(percentile - 50):.0f}% of runs")
    console.print()
    
    # Failure clusters table
    if analysis["clusters"]:
        console.print("[primary]Failure Clusters[/primary]")
        console.print()
        
        table = Table(show_header=True, header_style="bold")
        table.add_column("Pattern", style="error")
        table.add_column("Count", justify="right")
        table.add_column("Impact", style="warning")
        
        if historical_patterns:
            table.add_column("Historical", justify="right", style="muted")
        
        table.add_column("Example", style="muted", max_width=40)
        
        for cluster in analysis["clusters"]:
            row = [
                cluster["pattern"],
                str(cluster["count"]),
                cluster["impact"]
            ]
            
            if historical_patterns and "historical_frequency" in cluster:
                row.append(f"{cluster['historical_frequency']} total")
            
            row.append(cluster["example"][:40] + "..." if len(cluster["example"]) > 40 else cluster["example"])
            
            table.add_row(*row)
        
        console.print(table)
        console.print()
    
    # Cross-run insights
    if cross_run_insights and cross_run_insights.get("recent_runs"):
        console.print("[primary]Recent Performance Trend[/primary]")
        trend = cross_run_insights["trend"]
        trend_icon = "↑" if trend == "improving" else "↓" if trend == "declining" else "→"
        trend_style = "success" if trend == "improving" else "error" if trend == "declining" else "muted"
        
        console.print(f"[{trend_style}]{trend_icon} Reliability {trend} over last {len(cross_run_insights['recent_runs'])} runs[/{trend_style}]")
        console.print()
    
    # Top issues
    if analysis["top_issues"]:
        console.print("[primary]Top Issues to Address[/primary]")
        console.print()
        for i, issue in enumerate(analysis["top_issues"], 1):
            console.print(f"{i}. [error]{issue['title']}[/error]")
            console.print(f"   {issue['description']}", style="muted")
            if "historical_context" in issue:
                console.print(f"   {issue['historical_context']}", style="info")
            console.print()
    
    console.print("Run [info]arc recommend[/info] for specific configuration improvements", style="muted")
    console.print()


def _cluster_failures(failures: List[Dict]) -> List[Dict]:
    """Cluster failures by pattern (simplified implementation)."""
    clusters = {}
    
    for failure in failures:
        reason = failure.get("failure_reason", "Unknown")
        
        # Simple pattern matching
        if "currency" in reason.lower():
            pattern = "Currency Assumption Violation"
        elif "timeout" in reason.lower():
            pattern = "Timeout/Performance Issue"
        elif "tool" in reason.lower():
            pattern = "Tool Execution Error"
        elif "api" in reason.lower():
            pattern = "External API Failure"
        else:
            pattern = "Other"
        
        if pattern not in clusters:
            clusters[pattern] = {
                "pattern": pattern,
                "count": 0,
                "failures": [],
                "example": reason
            }
        
        clusters[pattern]["count"] += 1
        clusters[pattern]["failures"].append(failure)
    
    # Convert to list and add impact
    cluster_list = []
    for pattern, data in clusters.items():
        data["impact"] = "High" if data["count"] > 5 else "Medium" if data["count"] > 2 else "Low"
        cluster_list.append(data)
    
    # Sort by count
    cluster_list.sort(key=lambda x: x["count"], reverse=True)
    
    return cluster_list


def _identify_top_issues(clusters: List[Dict], historical_patterns: Optional[Dict] = None) -> List[Dict]:
    """Identify top issues from failure clusters."""
    issues = []
    
    for cluster in clusters[:3]:  # Top 3 issues
        issue = {}
        
        if cluster["pattern"] == "Currency Assumption Violation":
            issue = {
                "title": "Hard-coded Currency Assumptions",
                "description": "Agent assumes USD for all transactions. Add multi-currency support.",
                "severity": "high",
                "recommendation": "Implement currency detection and conversion logic"
            }
        elif cluster["pattern"] == "Timeout/Performance Issue":
            issue = {
                "title": "Slow External API Calls",
                "description": "Multiple scenarios timing out on API calls. Consider caching or async calls.",
                "severity": "medium",
                "recommendation": "Add request timeouts and retry logic"
            }
        elif cluster["pattern"] == "Tool Execution Error":
            issue = {
                "title": "Tool Configuration Issues",
                "description": "Tools failing to execute properly. Check tool definitions and parameters.",
                "severity": "medium",
                "recommendation": "Validate tool configurations and add error handling"
            }
        else:
            issue = {
                "title": cluster["pattern"],
                "description": f"{cluster['count']} failures detected in this category",
                "severity": "low",
                "recommendation": "Review individual failures for patterns"
            }
        
        # Add historical context if available
        if historical_patterns and "historical_frequency" in cluster:
            issue["historical_context"] = (
                f"Historically affects {cluster['historical_affected_runs']} runs "
                f"({cluster['historical_frequency']} total occurrences)"
            )
        
        issues.append(issue)
    
    return issues