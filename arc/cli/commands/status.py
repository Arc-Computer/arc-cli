"""Arc status command - show recent runs and costs."""

from datetime import datetime, timedelta
from pathlib import Path

import click
from rich.table import Table
from rich.panel import Panel

from arc.cli.utils import ArcConsole, CLIState, format_error

console = ArcConsole()
state = CLIState()


@click.command()
@click.option('--limit', '-l', default=10, help='Number of runs to show (default: 10)')
@click.option('--json', 'json_output', is_flag=True, help='Output JSON instead of rich text')
def status(limit: int, json_output: bool):
    """Show recent runs and usage statistics.
    
    This command displays:
    - Recent run history
    - Total costs over time
    - Current reliability trends
    
    Example:
        arc status
        arc status --limit 20
    """
    try:
        # Get recent runs
        runs = state.list_runs(limit=limit)
        
        # Calculate costs
        cost_30d = state.get_total_cost(days=30)
        cost_7d = state.get_total_cost(days=7)
        cost_1d = state.get_total_cost(days=1)
        
        if json_output:
            import json
            output = {
                "runs": runs,
                "costs": {
                    "last_30_days": cost_30d,
                    "last_7_days": cost_7d,
                    "last_24_hours": cost_1d
                }
            }
            print(json.dumps(output, indent=2))
        else:
            console.print()
            console.print_header("Arc Status")
            
            # Cost summary
            console.print("[primary]Usage Summary[/primary]")
            console.print()
            console.print_cost("Last 24 hours", cost_1d)
            console.print_cost("Last 7 days", cost_7d)
            console.print_cost("Last 30 days", cost_30d)
            console.print()
            
            # Recent runs
            if runs:
                console.print("[primary]Recent Runs[/primary]")
                console.print()
                
                table = Table(show_header=True, header_style="bold")
                table.add_column("Run ID", style="info")
                table.add_column("Config", style="muted")
                table.add_column("Time", style="muted")
                table.add_column("Reliability", justify="right")
                table.add_column("Scenarios", justify="right")
                table.add_column("Cost", justify="right")
                
                for run in runs:
                    # Format timestamp
                    ts = datetime.fromisoformat(run["timestamp"])
                    time_str = _format_time_ago(ts)
                    
                    # Format reliability with color
                    rel = run["reliability_score"]
                    rel_str = f"{rel:.1%}"
                    if rel >= 0.9:
                        rel_style = "success"
                    elif rel >= 0.7:
                        rel_style = "warning"
                    else:
                        rel_style = "error"
                    
                    table.add_row(
                        run["run_id"][:20] + "...",
                        Path(run["config_path"]).name,
                        time_str,
                        f"[{rel_style}]{rel_str}[/{rel_style}]",
                        str(run["scenario_count"]),
                        f"${run['total_cost']:.4f}"
                    )
                
                console.print(table)
                console.print()
                
                # Show trend if enough data
                if len(runs) >= 3:
                    reliabilities = [r["reliability_score"] for r in runs[:5]]
                    avg_recent = sum(reliabilities[:2]) / 2
                    avg_older = sum(reliabilities[2:]) / len(reliabilities[2:])
                    
                    if avg_recent > avg_older + 0.05:
                        console.print("[success]↑ Reliability improving[/success]", style="success")
                    elif avg_recent < avg_older - 0.05:
                        console.print("[error]↓ Reliability declining[/error]", style="error")
                    else:
                        console.print("→ Reliability stable", style="muted")
                    console.print()
            else:
                console.print("No runs found. Get started with [info]arc run your_agent.yaml[/info]")
                console.print()
    
    except Exception as e:
        if json_output:
            import json
            print(json.dumps({"error": str(e)}, indent=2))
        else:
            console.print(format_error(f"Status check failed: {str(e)}"))
        raise click.exceptions.Exit(1)


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