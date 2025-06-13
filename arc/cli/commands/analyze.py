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
@click.option('--detailed', is_flag=True, help='Show detailed scenario-by-scenario breakdown')
@click.option('--json', 'json_output', is_flag=True, help='Output JSON instead of rich text')
def analyze(run_id: Optional[str], historical: bool, days: int, detailed: bool, json_output: bool):
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
        asyncio.run(_analyze_async(run_id, historical, days, detailed, json_output))
    except Exception as e:
        if json_output:
            import json
            print(json.dumps({"error": str(e)}, indent=2))
        else:
            console.print(format_error(f"Analysis failed: {str(e)}"))
        raise click.exceptions.Exit(1)


async def _analyze_async(run_id: Optional[str], historical: bool, days: int, detailed: bool, json_output: bool):
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
    
    # Extract failures from results (since run_result.failures is not populated)
    failures = _extract_failures_from_results(run_result.results)
    
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
        
        # Add historical data to analysis even for successful runs
        if historical_patterns:
            analysis["historical_patterns"] = historical_patterns
        if cross_run_insights:
            analysis["cross_run_insights"] = cross_run_insights
        
        # Save analysis even for successful runs (needed for recommend command)
        state.save_analysis(run_result.run_id, analysis)
        
        # Save to database if available
        if db_connected and isinstance(state, HybridState):
            db_client = db_manager.get_client()
            if db_client:
                try:
                    await _save_analysis_to_database(db_client, run_result, analysis)
                except Exception as e:
                    console.print(f"Warning: Failed to save analysis to database: {str(e)}", style="warning")
        
        if json_output:
            import json
            print(json.dumps(analysis, indent=2))
        else:
            # Show comprehensive analysis even for successful runs
            _display_analysis_ui(run_result, analysis, historical_patterns, cross_run_insights, detailed)
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
        _display_analysis_ui(run_result, analysis, historical_patterns, cross_run_insights, detailed)


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


def _display_analysis_ui(run_result, analysis, historical_patterns, cross_run_insights, detailed=False):
    """Display comprehensive analysis results with detailed trajectory insights."""
    console.print()
    console.print_header(f"Simulation Analysis for {run_result.config_path}")
    
    # Run Overview
    console.print_metric("Run ID", run_result.run_id, style="muted")
    console.print_metric("Scenarios", f"{run_result.scenario_count} total")
    console.print_metric("Success Rate", f"{run_result.success_count}/{run_result.scenario_count} ({(run_result.success_count/run_result.scenario_count)*100:.1f}%)")
    console.print_metric("Reliability Score", f"{run_result.reliability_percentage:.1f}%")
    console.print_metric("Total Cost", f"${run_result.total_cost:.4f}")
    console.print_metric("Execution Time", f"{run_result.execution_time:.1f}s")
    
    # Show historical context if available
    if historical_patterns:
        percentile = (1 - (run_result.reliability_score / historical_patterns["avg_reliability"])) * 100
        console.print_metric("Historical comparison", 
                           f"{'Better' if percentile > 50 else 'Worse'} than {abs(percentile - 50):.0f}% of runs")
    console.print()
    
    # Detailed Execution Analysis
    _display_execution_analysis(run_result, console)
    
    # Performance Insights
    _display_performance_insights(run_result, console)
    
    # Failure Analysis (if any failures)
    if analysis["total_failures"] > 0:
        _display_failure_analysis(analysis, historical_patterns, console)
    
    # Success Pattern Analysis
    _display_success_patterns(run_result, console)
    
    # Cross-run insights
    if cross_run_insights and cross_run_insights.get("recent_runs"):
        console.print("[primary]Performance Trend[/primary]")
        trend = cross_run_insights["trend"]
        trend_icon = "↑" if trend == "improving" else "↓" if trend == "declining" else "→"
        trend_style = "success" if trend == "improving" else "error" if trend == "declining" else "muted"
        
        console.print(f"[{trend_style}]{trend_icon} Reliability {trend} over last {len(cross_run_insights['recent_runs'])} runs[/{trend_style}]")
        console.print()
    
    # Detailed scenario breakdown if requested
    if detailed:
        _display_detailed_scenarios(run_result, console)
    
    # Actionable Insights
    _display_actionable_insights(analysis, run_result, console)
    
    console.print("Run [info]arc recommend[/info] for specific configuration improvements", style="muted")
    if not detailed:
        console.print("Run [info]arc analyze --detailed[/info] for scenario-by-scenario breakdown", style="muted")
    console.print()


def _extract_failures_from_results(results: List[Dict]) -> List[Dict]:
    """Extract and structure failure data from run results."""
    failures = []
    
    for result in results:
        trajectory = result.get("trajectory", {})
        scenario = result.get("scenario", {})
        
        if trajectory.get("status") == "error":
            # Extract meaningful failure information
            error_msg = trajectory.get("error", "Unknown error")
            scenario_name = scenario.get("name", "Unknown scenario")
            task_prompt = scenario.get("task_prompt", trajectory.get("task_prompt", ""))
            expected_failure = scenario.get("potential_failure_mode", "")
            expected_tools = scenario.get("expected_tools", [])
            
            failure = {
                "scenario_name": scenario_name,
                "error_message": error_msg,
                "task_prompt": task_prompt,
                "expected_failure_mode": expected_failure,
                "execution_time": trajectory.get("execution_time_seconds", 0),
                "trajectory_events": trajectory.get("full_trajectory", []),
                "scenario_type": scenario.get("inferred_domain", "unknown"),
                "complexity": scenario.get("complexity_level", "unknown"),
                "expected_tools": expected_tools,
                "failure_reason": _categorize_error_type(error_msg, expected_failure, trajectory, expected_tools),
                "raw_trajectory": trajectory,
                "raw_scenario": scenario
            }
            
            failures.append(failure)
    
    return failures


def _categorize_error_type(error_msg: str, expected_failure: str, trajectory: Dict, expected_tools: List[str] = None) -> str:
    """Categorize the type of error based on error message and context."""
    error_lower = error_msg.lower()
    expected_lower = expected_failure.lower()
    expected_tools = expected_tools or []
    
    # First check for tool-related context - this is important!
    trajectory_events = trajectory.get("full_trajectory", [])
    has_tool_calls = any(event.get("type") == "tool_call" for event in trajectory_events)
    has_expected_tools = len(expected_tools) > 0
    
    # Check for specific error patterns with tool context priority
    if "currency" in expected_lower:
        return "Currency Assumption Violation"
    elif "ambiguous" in expected_lower:
        return "Ambiguous Input Handling"
    elif has_tool_calls or "tool" in error_lower:
        # If there were actual tool calls or tool mentioned in error
        return "Tool Execution Error"
    elif has_expected_tools and ("nonetype" in error_lower or "parsing" in expected_lower or "json" in expected_lower):
        # If tools were expected but agent failed (likely tool-related even if no tool calls recorded)
        return "Tool Execution Error"
    elif len(trajectory_events) == 0 and not has_expected_tools:
        # Only categorize as early execution failure if no tools were expected
        return "Early Execution Failure"
    elif "nonetype" in error_lower and "attribute" in error_lower:
        # NoneType errors without tool context are code execution issues
        return "Code Execution Error"
    elif "timeout" in error_lower or "time" in error_lower:
        return "Timeout/Performance Issue"
    elif "connection" in error_lower or "network" in error_lower:
        return "Network/API Error"
    elif "validation" in error_lower or "invalid" in error_lower:
        return "Validation Error"
    elif "database" in error_lower or "sql" in error_lower:
        return "Database Error"
    else:
        return "Unknown Error"


def _cluster_failures(failures: List[Dict]) -> List[Dict]:
    """Cluster failures by pattern using improved categorization."""
    clusters = {}
    
    for failure in failures:
        pattern = failure.get("failure_reason", "Unknown Error")
        
        if pattern not in clusters:
            clusters[pattern] = {
                "pattern": pattern,
                "count": 0,
                "failures": [],
                "example": failure.get("error_message", "Unknown"),
                "scenarios": [],
                "common_characteristics": {
                    "domains": set(),
                    "complexity_levels": set(),
                    "expected_tools": set()
                }
            }
        
        clusters[pattern]["count"] += 1
        clusters[pattern]["failures"].append(failure)
        clusters[pattern]["scenarios"].append(failure.get("scenario_name", "Unknown"))
        
        # Track common characteristics
        clusters[pattern]["common_characteristics"]["domains"].add(failure.get("scenario_type", "unknown"))
        clusters[pattern]["common_characteristics"]["complexity_levels"].add(failure.get("complexity", "unknown"))
        for tool in failure.get("expected_tools", []):
            clusters[pattern]["common_characteristics"]["expected_tools"].add(tool)
    
    # Convert to list and add impact analysis
    cluster_list = []
    for pattern, data in clusters.items():
        # Convert sets to lists for JSON serialization
        data["common_characteristics"]["domains"] = list(data["common_characteristics"]["domains"])
        data["common_characteristics"]["complexity_levels"] = list(data["common_characteristics"]["complexity_levels"])
        data["common_characteristics"]["expected_tools"] = list(data["common_characteristics"]["expected_tools"])
        
        # Determine impact based on count and characteristics
        count = data["count"]
        if count > 5:
            data["impact"] = "High"
        elif count > 2:
            data["impact"] = "Medium"
        elif pattern in ["Code Execution Error", "Early Execution Failure"]:
            data["impact"] = "High"  # These are always serious
        else:
            data["impact"] = "Low"
        
        cluster_list.append(data)
    
    # Sort by impact and count
    impact_order = {"High": 3, "Medium": 2, "Low": 1}
    cluster_list.sort(key=lambda x: (impact_order.get(x["impact"], 0), x["count"]), reverse=True)
    
    return cluster_list


def _identify_top_issues(clusters: List[Dict], historical_patterns: Optional[Dict] = None) -> List[Dict]:
    """Identify top issues from failure clusters with detailed analysis."""
    issues = []
    
    for cluster in clusters[:3]:  # Top 3 issues
        pattern = cluster["pattern"]
        count = cluster["count"]
        scenarios = cluster.get("scenarios", [])
        characteristics = cluster.get("common_characteristics", {})
        
        # Generate issue based on pattern type
        if pattern == "Code Execution Error":
            issue = {
                "title": "Code Execution Failures",
                "description": f"Agent code is failing with runtime errors. Affects {count} scenario(s): {', '.join(scenarios[:3])}",
                "severity": "high",
                "recommendation": "Review agent code for null pointer exceptions and type errors. Add defensive programming practices.",
                "technical_details": f"Common error: {cluster['example']}"
            }
        elif pattern == "Ambiguous Input Handling":
            issue = {
                "title": "Ambiguous Input Processing",
                "description": f"Agent struggles with ambiguous or unclear inputs. Affects {count} scenario(s) in {', '.join(characteristics.get('domains', []))} domain(s).",
                "severity": "medium",
                "recommendation": "Implement clarification requests and input validation. Add examples of how to handle ambiguous cases.",
                "technical_details": "Agent should ask for clarification rather than making assumptions"
            }
        elif pattern == "Early Execution Failure":
            issue = {
                "title": "Early Execution Termination",
                "description": f"Agent fails before completing any meaningful work. {count} scenario(s) affected.",
                "severity": "high",
                "recommendation": "Check agent initialization, tool availability, and basic error handling.",
                "technical_details": "No trajectory events recorded - failure occurs during setup"
            }
        elif pattern == "Currency Assumption Violation":
            issue = {
                "title": "Currency Handling Issues",
                "description": f"Agent makes incorrect currency assumptions. Expected in {count} scenario(s).",
                "severity": "medium",
                "recommendation": "Implement explicit currency detection and multi-currency support.",
                "technical_details": "Add currency validation and conversion logic"
            }
        elif pattern == "Tool Execution Error":
            expected_tools = characteristics.get("expected_tools", [])
            issue = {
                "title": "Tool Execution Problems",
                "description": f"Tools failing to execute properly. Expected tools: {', '.join(expected_tools[:3])}",
                "severity": "medium",
                "recommendation": "Validate tool configurations, parameters, and error handling.",
                "technical_details": f"Tool-related errors in {count} scenarios"
            }
        else:
            issue = {
                "title": pattern,
                "description": f"{count} failures detected in this category. Scenarios: {', '.join(scenarios[:3])}",
                "severity": "low" if count <= 2 else "medium",
                "recommendation": "Investigate individual failures for common patterns and root causes.",
                "technical_details": f"Example error: {cluster['example']}"
            }
        
        # Add scenario context
        issue["affected_scenarios"] = scenarios
        issue["scenario_characteristics"] = characteristics
        
        # Add historical context if available
        if historical_patterns and "historical_frequency" in cluster:
            issue["historical_context"] = (
                f"Historically affects {cluster['historical_affected_runs']} runs "
                f"({cluster['historical_frequency']} total occurrences)"
            )
        
        issues.append(issue)
    
    return issues


def _display_execution_analysis(run_result, console):
    """Display detailed execution trail analysis."""
    console.print("[primary]Execution Trail Analysis[/primary]")
    
    # Analyze trajectory patterns across all scenarios
    tool_usage = {}
    execution_patterns = {"success": [], "error": []}
    token_usage_by_scenario = []
    
    for result in run_result.results:
        trajectory = result.get("trajectory", {})
        scenario = result.get("scenario", {})
        status = trajectory.get("status", "unknown")
        
        # Track tool usage
        for event in trajectory.get("full_trajectory", []):
            if event.get("type") == "tool_call":
                tool_name = event.get("tool", "unknown")
                if tool_name not in tool_usage:
                    tool_usage[tool_name] = {"success": 0, "error": 0, "total_calls": 0}
                
                tool_usage[tool_name]["total_calls"] += 1
                if "error" in str(event.get("tool_output", "")).lower():
                    tool_usage[tool_name]["error"] += 1
                else:
                    tool_usage[tool_name]["success"] += 1
        
        # Track execution patterns
        pattern = {
            "scenario": scenario.get("name", "unknown"),
            "status": status,
            "execution_time": trajectory.get("execution_time_seconds", 0),
            "steps": len(trajectory.get("full_trajectory", [])),
            "tokens": trajectory.get("token_usage", {}).get("total_tokens", 0),
            "cost": trajectory.get("token_usage", {}).get("total_cost", 0)
        }
        
        execution_patterns[status if status in ["success", "error"] else "error"].append(pattern)
        token_usage_by_scenario.append(pattern["tokens"])
    
    # Display tool usage analysis
    if tool_usage:
        console.print("\n[bold]Tool Execution Analysis[/bold]")
        tool_table = Table(show_header=True, header_style="bold")
        tool_table.add_column("Tool", style="info")
        tool_table.add_column("Total Calls", justify="right")
        tool_table.add_column("Success Rate", justify="right")
        tool_table.add_column("Status", style="success")
        
        for tool, stats in sorted(tool_usage.items(), key=lambda x: x[1]["total_calls"], reverse=True):
            success_rate = (stats["success"] / stats["total_calls"]) * 100 if stats["total_calls"] > 0 else 0
            status_style = "success" if success_rate >= 80 else "warning" if success_rate >= 60 else "error"
            status_text = "Excellent" if success_rate >= 90 else "Good" if success_rate >= 80 else "Poor"
            
            tool_table.add_row(
                tool,
                str(stats["total_calls"]),
                f"{success_rate:.1f}%",
                f"[{status_style}]{status_text}[/{status_style}]"
            )
        
        console.print(tool_table)
    
    # Display execution pattern summary
    console.print("\n[bold]Execution Patterns[/bold]")
    
    if execution_patterns["success"]:
        avg_success_time = sum(p["execution_time"] for p in execution_patterns["success"]) / len(execution_patterns["success"])
        avg_success_steps = sum(p["steps"] for p in execution_patterns["success"]) / len(execution_patterns["success"])
        console.print(f"✓ Successful scenarios: {len(execution_patterns['success'])}")
        console.print(f"  Average execution time: {avg_success_time:.2f}s")
        console.print(f"  Average steps: {avg_success_steps:.1f}")
    
    if execution_patterns["error"]:
        avg_error_time = sum(p["execution_time"] for p in execution_patterns["error"]) / len(execution_patterns["error"])
        console.print(f"✗ Failed scenarios: {len(execution_patterns['error'])}")
        console.print(f"  Average time to failure: {avg_error_time:.2f}s")
        
        # Show common failure points
        error_scenarios = [p["scenario"] for p in execution_patterns["error"]]
        console.print(f"  Failed scenarios: {', '.join(error_scenarios[:3])}")
        if len(error_scenarios) > 3:
            console.print(f"  ... and {len(error_scenarios) - 3} more")
    
    console.print()


def _display_performance_insights(run_result, console):
    """Display performance and cost analysis."""
    console.print("[primary]Performance Insights[/primary]")
    
    # Calculate performance metrics
    total_tokens = 0
    total_cost = 0
    execution_times = []
    cost_per_scenario = []
    
    for result in run_result.results:
        trajectory = result.get("trajectory", {})
        token_usage = trajectory.get("token_usage", {})
        
        tokens = token_usage.get("total_tokens", 0)
        cost = token_usage.get("total_cost", 0)
        exec_time = trajectory.get("execution_time_seconds", 0)
        
        total_tokens += tokens
        total_cost += cost
        execution_times.append(exec_time)
        cost_per_scenario.append(cost)
    
    # Performance metrics table
    perf_table = Table(show_header=True, header_style="bold")
    perf_table.add_column("Metric", style="info")
    perf_table.add_column("Value", justify="right")
    perf_table.add_column("Per Scenario", justify="right", style="muted")
    
    avg_exec_time = sum(execution_times) / len(execution_times) if execution_times else 0
    avg_cost = sum(cost_per_scenario) / len(cost_per_scenario) if cost_per_scenario else 0
    avg_tokens = total_tokens / len(run_result.results) if run_result.results else 0
    
    perf_table.add_row("Total Tokens", f"{total_tokens:,}", f"{avg_tokens:.0f}")
    perf_table.add_row("Total Cost", f"${total_cost:.4f}", f"${avg_cost:.4f}")
    perf_table.add_row("Execution Time", f"{sum(execution_times):.1f}s", f"{avg_exec_time:.1f}s")
    
    # Cost efficiency
    cost_per_token = (total_cost / total_tokens * 1000) if total_tokens > 0 else 0
    perf_table.add_row("Cost per 1K tokens", f"${cost_per_token:.4f}", "")
    
    console.print(perf_table)
    
    # Performance distribution
    if len(execution_times) > 1:
        console.print(f"\n[bold]Performance Distribution[/bold]")
        fastest = min(execution_times)
        slowest = max(execution_times)
        console.print(f"Fastest scenario: {fastest:.2f}s")
        console.print(f"Slowest scenario: {slowest:.2f}s")
        if fastest > 0:
            console.print(f"Performance variance: {slowest/fastest:.1f}x")
        else:
            console.print("Performance variance: N/A (some scenarios had 0s execution time)")
    
    console.print()


def _display_failure_analysis(analysis, historical_patterns, console):
    """Display detailed failure analysis."""
    console.print("[primary]Failure Analysis[/primary]")
    
    # Failure clusters table
    if analysis["clusters"]:
        console.print("\n[bold]Failure Patterns[/bold]")
        
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
    
    # Top issues
    if analysis["top_issues"]:
        console.print("\n[bold]Critical Issues[/bold]")
        for i, issue in enumerate(analysis["top_issues"], 1):
            severity_style = "error" if issue.get("severity") == "high" else "warning" if issue.get("severity") == "medium" else "info"
            console.print(f"{i}. [{severity_style}]{issue['title']}[/{severity_style}]")
            console.print(f"   {issue['description']}", style="muted")
            console.print(f"   [bold]Recommendation:[/bold] {issue['recommendation']}")
            
            if "technical_details" in issue:
                console.print(f"   [bold]Technical Details:[/bold] {issue['technical_details']}", style="muted")
            
            if "affected_scenarios" in issue and len(issue["affected_scenarios"]) > 3:
                console.print(f"   [bold]All Affected Scenarios:[/bold] {', '.join(issue['affected_scenarios'])}", style="muted")
            
            if "historical_context" in issue:
                console.print(f"   [bold]Historical Context:[/bold] {issue['historical_context']}", style="info")
            console.print()
    
    console.print()


def _display_success_patterns(run_result, console):
    """Display analysis of successful execution patterns."""
    console.print("[primary]Success Pattern Analysis[/primary]")
    
    successful_results = [r for r in run_result.results if r.get("trajectory", {}).get("status") == "success"]
    
    if not successful_results:
        console.print("No successful scenarios to analyze.")
        console.print()
        return
    
    # Analyze successful patterns
    tool_sequences = []
    response_qualities = []
    execution_strategies = []
    
    for result in successful_results:
        trajectory = result.get("trajectory", {})
        scenario = result.get("scenario", {})
        
        # Extract tool sequence
        tools_used = []
        for event in trajectory.get("full_trajectory", []):
            if event.get("type") == "tool_call":
                tools_used.append(event.get("tool", "unknown"))
        
        if tools_used:
            tool_sequences.append(" → ".join(tools_used))
        
        # Analyze response quality indicators
        final_response = trajectory.get("final_response", "")
        response_qualities.append({
            "length": len(final_response),
            "has_structure": any(marker in final_response for marker in ["1.", "2.", "•", "-", "\n\n"]),
            "scenario": scenario.get("name", "unknown")
        })
    
    # Display common success patterns
    if tool_sequences:
        from collections import Counter
        common_sequences = Counter(tool_sequences).most_common(3)
        
        console.print("[bold]Common Tool Usage Patterns[/bold]")
        for sequence, count in common_sequences:
            console.print(f"• {sequence} ({count} scenarios)")
    
    # Response quality analysis
    if response_qualities:
        avg_length = sum(r["length"] for r in response_qualities) / len(response_qualities)
        structured_responses = sum(1 for r in response_qualities if r["has_structure"])
        
        console.print(f"\n[bold]Response Quality Indicators[/bold]")
        console.print(f"Average response length: {avg_length:.0f} characters")
        console.print(f"Structured responses: {structured_responses}/{len(response_qualities)} ({structured_responses/len(response_qualities)*100:.1f}%)")
    
    console.print()


def _display_actionable_insights(analysis, run_result, console):
    """Display actionable insights and recommendations."""
    console.print("[primary]Key Insights & Next Steps[/primary]")
    
    insights = []
    
    # Performance insights
    if run_result.total_cost > 0.1:
        insights.append("💰 High cost detected - consider optimizing model usage or prompt efficiency")
    
    if run_result.execution_time > 60:
        insights.append("⏱️ Long execution time - consider parallel processing or timeout optimization")
    
    # Reliability insights
    if run_result.reliability_score < 0.8:
        insights.append("🎯 Reliability below 80% - focus on error handling and validation")
    
    if run_result.failure_count > run_result.success_count:
        insights.append("⚠️ More failures than successes - review agent configuration and tools")
    
    # Success pattern insights
    successful_results = [r for r in run_result.results if r.get("trajectory", {}).get("status") == "success"]
    if successful_results:
        avg_success_time = sum(r.get("trajectory", {}).get("execution_time_seconds", 0) for r in successful_results) / len(successful_results)
        if avg_success_time < 5:
            insights.append("⚡ Fast successful executions - good agent efficiency")
    
    # Display insights
    if insights:
        for insight in insights:
            console.print(f"  {insight}")
    else:
        console.print("  ✅ No major issues detected - agent performing well")
    
    console.print()
    
    # Next steps
    console.print("[bold]Recommended Next Steps[/bold]")
    if analysis["total_failures"] > 0:
        console.print("1. Run [info]arc recommend[/info] for specific configuration improvements")
        console.print("2. Focus on the top failure patterns identified above")
        console.print("3. Test fixes with a smaller scenario set first")
    else:
        console.print("1. Run [info]arc recommend[/info] for cost optimization opportunities")
        console.print("2. Consider expanding scenario coverage")
        console.print("3. Monitor performance trends with [info]arc status[/info]")
    
    console.print()


def _display_detailed_scenarios(run_result, console):
    """Display detailed scenario-by-scenario breakdown."""
    console.print("[primary]Detailed Scenario Analysis[/primary]")
    
    for i, result in enumerate(run_result.results, 1):
        scenario = result.get("scenario", {})
        trajectory = result.get("trajectory", {})
        reliability_score = result.get("reliability_score", 0)
        
        scenario_name = scenario.get("name", f"Scenario {i}")
        status = trajectory.get("status", "unknown")
        
        # Status indicator
        status_icon = "✓" if status == "success" else "✗" if status == "error" else "?"
        status_style = "success" if status == "success" else "error" if status == "error" else "warning"
        
        console.print(f"\n[bold]{i}. [{status_style}]{status_icon} {scenario_name}[/{status_style}][/bold]")
        
        # Basic metrics
        exec_time = trajectory.get("execution_time_seconds", 0)
        token_usage = trajectory.get("token_usage", {})
        tokens = token_usage.get("total_tokens", 0)
        cost = token_usage.get("total_cost", 0)
        
        console.print(f"   Status: [{status_style}]{status.title()}[/{status_style}]")
        console.print(f"   Reliability: {reliability_score:.1%}" if isinstance(reliability_score, (int, float)) else f"   Reliability: {reliability_score}")
        console.print(f"   Execution: {exec_time:.2f}s | Tokens: {tokens:,} | Cost: ${cost:.4f}")
        
        # Task prompt
        task_prompt = scenario.get("task_prompt", trajectory.get("task_prompt", ""))
        if task_prompt:
            console.print(f"   Task: {task_prompt[:100]}{'...' if len(task_prompt) > 100 else ''}", style="muted")
        
        # Execution trail
        full_trajectory = trajectory.get("full_trajectory", [])
        if full_trajectory:
            console.print(f"   [bold]Execution Trail ({len(full_trajectory)} steps):[/bold]")
            for step_idx, event in enumerate(full_trajectory[:5], 1):  # Show first 5 steps
                event_type = event.get("type", "unknown")
                
                if event_type == "tool_call":
                    tool_name = event.get("tool", "unknown")
                    tool_output = str(event.get("tool_output", ""))
                    is_error = "error" in tool_output.lower()
                    
                    console.print(f"     {step_idx}. [info]Tool Call:[/info] {tool_name}")
                    if is_error:
                        console.print(f"        [error]Error: {tool_output[:80]}{'...' if len(tool_output) > 80 else ''}[/error]")
                    else:
                        console.print(f"        Output: {tool_output[:80]}{'...' if len(tool_output) > 80 else ''}", style="muted")
                
                elif event_type == "final_response":
                    response = event.get("content", "")
                    console.print(f"     {step_idx}. [info]Final Response:[/info]")
                    console.print(f"        {response[:100]}{'...' if len(response) > 100 else ''}", style="muted")
                
                else:
                    console.print(f"     {step_idx}. [info]{event_type.title()}[/info]")
            
            if len(full_trajectory) > 5:
                console.print(f"     ... and {len(full_trajectory) - 5} more steps")
        
        # Error details for failed scenarios
        if status == "error":
            error_msg = trajectory.get("error", "Unknown error")
            console.print(f"   [error]Error Details:[/error] {error_msg}")
            
            # Expected failure mode if available
            expected_failure = scenario.get("potential_failure_mode", "")
            if expected_failure:
                console.print(f"   [warning]Expected Failure:[/warning] {expected_failure[:150]}{'...' if len(expected_failure) > 150 else ''}")
        
        # Final response for successful scenarios
        elif status == "success":
            final_response = trajectory.get("final_response", "")
            if final_response:
                console.print(f"   [success]Final Response:[/success] {final_response[:150]}{'...' if len(final_response) > 150 else ''}")
    
    console.print()