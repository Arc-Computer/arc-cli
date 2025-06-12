"""Arc status command - display usage metrics and system health."""

import asyncio
from datetime import datetime, timedelta
import click
from rich.table import Table
from rich.panel import Panel
from rich.columns import Columns
from sqlalchemy import text

from arc.cli.utils import ArcConsole, CLIState, format_error
from arc.cli.utils import db_manager

console = ArcConsole()
state = CLIState()


@click.command()
@click.option('--days', '-d', default=30, help='Number of days of history to show (default: 30)')
@click.option('--limit', '-l', default=10, help='Number of runs to show (default: 10)')
@click.option('--json', 'json_output', is_flag=True, help='Output JSON instead of rich text')
def status(days: int, limit: int, json_output: bool):
    """Display Arc system status, usage metrics, and trends.
    
    This command shows:
    - Database connection status
    - Usage metrics (costs, runs, scenarios)
    - Reliability trends over time
    - Recent run history
    - Resource utilization
    
    Example:
        arc status
        arc status --days 7
        arc status --json
    """
    try:
        # Run async status check
        asyncio.run(_show_status_async(days, limit, json_output))
    except Exception as e:
        if json_output:
            import json
            print(json.dumps({"error": str(e)}, indent=2))
        else:
            console.print(format_error(f"Status check failed: {str(e)}"))
        raise click.exceptions.Exit(1)


async def _show_status_async(days: int, limit: int, json_output: bool):
    """Show system status asynchronously."""
    # Check database connection
    db_connected = db_manager.is_available
    
    if not db_connected:
        # Try to initialize if not already done
        db_connected = await db_manager.initialize()
    
    status_data = {
        "database_connected": db_connected,
        "timestamp": datetime.now().isoformat(),
        "days_requested": days
    }
    
    if db_connected:
        # Get metrics from database
        db_client = db_manager.get_client()
        if db_client:
            try:
                # Get usage metrics
                usage_metrics = await _get_usage_metrics(db_client, days)
                status_data["usage_metrics"] = usage_metrics
                
                # Get recent runs
                recent_runs = await _get_recent_runs(db_client, limit=limit)
                status_data["recent_runs"] = recent_runs
                
                # Get reliability trends
                reliability_trends = await _get_reliability_trends(db_client, days)
                status_data["reliability_trends"] = reliability_trends
                
                # Get failure patterns
                failure_patterns = await _get_top_failure_patterns(db_client, days)
                status_data["failure_patterns"] = failure_patterns
                
            except Exception as e:
                console.print(f"Warning: Database query failed: {str(e)}", style="warning")
                status_data["database_error"] = str(e)
    
    # Get file-based metrics as fallback or addition
    file_metrics = _get_file_metrics(days, limit)
    status_data["file_metrics"] = file_metrics
    
    # Display results
    if json_output:
        import json
        print(json.dumps(status_data, indent=2))
    else:
        _display_status_ui(status_data)


async def _get_usage_metrics(db_client, days: int) -> dict:
    """Get usage metrics from database."""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    query = text("""
    SELECT 
        COUNT(DISTINCT s.simulation_id) as total_runs,
        COUNT(o.outcome_id) as total_scenarios,
        SUM(o.cost_usd) as total_cost,
        AVG(o.reliability_score) as avg_reliability,
        SUM(o.tokens_used) as total_tokens
    FROM simulations s
    JOIN outcomes o ON s.simulation_id = o.simulation_id
    WHERE s.created_at >= :start_date AND s.created_at <= :end_date
    """)
    
    async with db_client.engine.begin() as conn:
        result = await conn.execute(
            query,
            {"start_date": start_date, "end_date": end_date}
        )
        row = result.fetchone()
    
    return {
        "total_runs": row[0] or 0,
        "total_scenarios": row[1] or 0,
        "total_cost": float(row[2] or 0),
        "avg_reliability": float(row[3] or 0),
        "total_tokens": row[4] or 0,
        "period_days": days
    }


async def _get_recent_runs(db_client, limit: int = 10) -> list:
    """Get recent simulation runs."""
    query = text("""
    SELECT 
        s.simulation_id,
        s.simulation_name,
        c.name as config_name,
        s.created_at,
        s.overall_score,
        s.total_cost_usd,
        COUNT(o.outcome_id) as scenario_count,
        SUM(CASE WHEN o.status = 'success' THEN 1 ELSE 0 END) as success_count
    FROM simulations s
    JOIN config_versions cv ON s.config_version_id = cv.version_id
    JOIN configurations c ON cv.config_id = c.config_id
    JOIN outcomes o ON s.simulation_id = o.simulation_id
    GROUP BY s.simulation_id, s.simulation_name, c.name,
             s.created_at, s.overall_score, s.total_cost_usd
    ORDER BY s.created_at DESC
    LIMIT :limit
    """)
    
    async with db_client.engine.begin() as conn:
        result = await conn.execute(query, {"limit": limit})
    
    runs = []
    for row in result:
        # Always coerce to str for JSON-friendly output
        run_id = str(row[0])
        
        runs.append({
            "run_id": run_id,  # simulation_id
            "config": row[2],
            "timestamp": row[3].isoformat(),
            "reliability": float(row[4] or 0),
            "cost": float(row[5] or 0),
            "scenarios": row[6],
            "successes": row[7]
        })
    
    return runs


async def _get_reliability_trends(db_client, days: int) -> dict:
    """Get reliability trends over time."""
    query = text(f"""
    SELECT 
        DATE_TRUNC('day', s.created_at) as day,
        AVG(s.overall_score) as avg_reliability,
        COUNT(DISTINCT s.simulation_id) as run_count
    FROM simulations s
    WHERE s.created_at >= NOW() - INTERVAL '{days} days'
    GROUP BY DATE_TRUNC('day', s.created_at)
    ORDER BY day DESC
    """)
    
    async with db_client.engine.begin() as conn:
        result = await conn.execute(query)
    
    trends = []
    for row in result:
        trends.append({
            "date": row[0].isoformat(),
            "reliability": float(row[1] or 0),
            "runs": row[2]
        })
    
    # Calculate trend direction
    if len(trends) >= 2:
        recent_avg = sum(t["reliability"] for t in trends[:7]) / min(7, len(trends))
        older_avg = sum(t["reliability"] for t in trends[7:14]) / min(7, len(trends[7:14])) if len(trends) > 7 else recent_avg
        trend_direction = "improving" if recent_avg > older_avg else "declining" if recent_avg < older_avg else "stable"
    else:
        trend_direction = "insufficient_data"
    
    return {
        "daily_trends": trends,
        "trend_direction": trend_direction
    }


async def _get_top_failure_patterns(db_client, days: int) -> list:
    """Get top failure patterns."""
    query = text(f"""
    SELECT 
        o.error_category,
        COUNT(*) as occurrence_count,
        AVG(o.cost_usd) as avg_cost_per_failure
    FROM outcomes o
    JOIN simulations s ON o.simulation_id = s.simulation_id
    WHERE o.status = 'error' 
        AND s.created_at >= NOW() - INTERVAL '{days} days'
        AND o.error_category IS NOT NULL
    GROUP BY o.error_category
    ORDER BY occurrence_count DESC
    LIMIT 5
    """)
    
    async with db_client.engine.begin() as conn:
        result = await conn.execute(query)
    
    patterns = []
    for row in result:
        patterns.append({
            "category": row[0],
            "count": row[1],
            "avg_cost": float(row[2] or 0)
        })
    
    return patterns


def _get_file_metrics(days: int, limit: int) -> dict:
    """Get metrics from file-based storage."""
    # Get recent runs using existing state methods
    runs = state.list_runs(limit=limit)
    
    # Calculate costs for different periods
    cost_30d = state.get_total_cost(days=30)
    cost_7d = state.get_total_cost(days=7)
    cost_1d = state.get_total_cost(days=1)
    
    # Filter runs by date if needed
    cutoff_date = datetime.now() - timedelta(days=days)
    recent_runs = []
    for run in runs:
        try:
            run_date = datetime.fromisoformat(run["timestamp"])
            if run_date >= cutoff_date:
                recent_runs.append(run)
        except:
            continue
    
    return {
        "total_runs": len(recent_runs),
        "recent_runs": recent_runs,
        "costs": {
            "last_30_days": cost_30d,
            "last_7_days": cost_7d,
            "last_24_hours": cost_1d
        }
    }


def _display_status_ui(status_data: dict):
    """Display status in rich formatted UI."""
    console.print()
    console.print(Panel.fit(
        "[primary]Arc Status[/primary]",
        border_style="primary"
    ))
    console.print()
    
    # System Status
    console.print("[bold]System Status[/bold]")
    status_table = Table(show_header=False, box=None, padding=(0, 2))
    status_table.add_column("Component")
    status_table.add_column("Status")
    
    db_status = "✓ Connected" if status_data["database_connected"] else "✗ Disconnected"
    db_style = "success" if status_data["database_connected"] else "error"
    status_table.add_row("Database", f"[{db_style}]{db_status}[/{db_style}]")
    status_table.add_row("File Storage", "[success]✓ Available[/success]")
    
    console.print(status_table)
    console.print()
    
    # Usage Summary
    if status_data.get("usage_metrics"):
        metrics = status_data["usage_metrics"]
        console.print("[bold]Usage Summary[/bold]")
        console.print(f"Last {metrics['period_days']} days")
        console.print()
        
        # Create metric cards
        cards = []
        
        # Total runs card
        runs_card = Panel(
            f"[bold]{metrics['total_runs']}[/bold]\nRuns",
            width=20,
            style="blue"
        )
        cards.append(runs_card)
        
        # Total scenarios card
        scenarios_card = Panel(
            f"[bold]{metrics['total_scenarios']}[/bold]\nScenarios",
            width=20,
            style="green"
        )
        cards.append(scenarios_card)
        
        # Total cost card
        cost_card = Panel(
            f"[bold]${metrics['total_cost']:.4f}[/bold]\nTotal Cost",
            width=20,
            style="yellow"
        )
        cards.append(cost_card)
        
        # Average reliability card
        reliability_card = Panel(
            f"[bold]{metrics['avg_reliability']:.1%}[/bold]\nAvg Reliability",
            width=20,
            style="cyan"
        )
        cards.append(reliability_card)
        
        console.print(Columns(cards))
        console.print()
    elif status_data.get("file_metrics", {}).get("costs"):
        # Fallback to file-based cost summary
        costs = status_data["file_metrics"]["costs"]
        console.print("[bold]Usage Summary[/bold]")
        console.print()
        console.print_cost("Last 24 hours", costs["last_24_hours"])
        console.print_cost("Last 7 days", costs["last_7_days"])
        console.print_cost("Last 30 days", costs["last_30_days"])
        console.print()
    
    # Recent Runs
    recent_runs = status_data.get("recent_runs", [])
    if not recent_runs and status_data.get("file_metrics", {}).get("recent_runs"):
        recent_runs = status_data["file_metrics"]["recent_runs"]
    
    if recent_runs:
        console.print("[bold]Recent Runs[/bold]")
        runs_table = Table(show_header=True, header_style="bold")
        runs_table.add_column("Run ID", style="info")
        runs_table.add_column("Config")
        runs_table.add_column("Time", style="muted")
        runs_table.add_column("Reliability", justify="right")
        runs_table.add_column("Scenarios", justify="right")
        runs_table.add_column("Cost", justify="right")
        
        for run in recent_runs[:10]:
            # Format timestamp
            if "timestamp" in run:
                try:
                    dt = datetime.fromisoformat(run["timestamp"])
                    time_str = _format_time_ago(dt)
                except:
                    time_str = "Unknown"
            else:
                time_str = "Unknown"
            
            # Get reliability value
            reliability = run.get("reliability", run.get("reliability_score", 0))
            
            # Format reliability with color
            rel_str = f"{reliability:.1%}"
            if reliability >= 0.9:
                rel_style = "success"
            elif reliability >= 0.7:
                rel_style = "warning"
            else:
                rel_style = "error"
            
            runs_table.add_row(
                run.get("run_id", "Unknown")[:15] + "…",
                run.get("config", run.get("config_path", "Unknown")).split("/")[-1][:15] + "…",
                time_str,
                f"[{rel_style}]{rel_str}[/{rel_style}]",
                str(run.get("scenarios", run.get("scenario_count", 0))),
                f"${run.get('cost', run.get('total_cost', 0)):.4f}"
            )
        
        console.print(runs_table)
        console.print()
    
    # Reliability Trends
    if status_data.get("reliability_trends"):
        trends = status_data["reliability_trends"]
        console.print("[bold]Reliability Trend[/bold]")
        
        trend_icon = "↑" if trends["trend_direction"] == "improving" else "↓" if trends["trend_direction"] == "declining" else "→"
        trend_style = "success" if trends["trend_direction"] == "improving" else "error" if trends["trend_direction"] == "declining" else "muted"
        
        console.print(f"[{trend_style}]{trend_icon} Reliability {trends['trend_direction']}[/{trend_style}]")
        console.print()
    
    # Top Failure Patterns
    if status_data.get("failure_patterns"):
        console.print("[bold]Top Failure Patterns[/bold]")
        patterns_table = Table(show_header=True, header_style="bold")
        patterns_table.add_column("Pattern", style="error")
        patterns_table.add_column("Occurrences", justify="right")
        patterns_table.add_column("Avg Cost", justify="right")
        
        for pattern in status_data["failure_patterns"]:
            patterns_table.add_row(
                pattern["category"],
                str(pattern["count"]),
                f"${pattern['avg_cost']:.4f}"
            )
        
        console.print(patterns_table)
        console.print()
    
    # Footer
    if not recent_runs:
        console.print("No runs found. Get started with [info]arc run your_agent.yaml[/info]")
        console.print()
    
    console.print(f"[muted]Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}[/muted]")
    console.print()


def _format_time_ago(timestamp: datetime) -> str:
    """Format timestamp as time ago."""
    now = datetime.now()
    diff = now - timestamp
    
    if diff < timedelta(minutes=1):
        return "just now"
    elif diff < timedelta(hours=1):
        minutes = int(diff.total_seconds() / 60)
        return f"{minutes}m ago"
    elif diff < timedelta(days=1):
        hours = int(diff.total_seconds() / 3600)
        return f"{hours}h ago"
    elif diff < timedelta(days=7):
        days = diff.days
        return f"{days}d ago"
    else:
        return timestamp.strftime("%Y-%m-%d")