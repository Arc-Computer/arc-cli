"""Arc analyze command - analyze failures from a run."""

from typing import Optional, List, Dict

import click
from rich.table import Table
from rich.panel import Panel

from arc.cli.utils import ArcConsole, CLIState, format_error, format_success

console = ArcConsole()
state = CLIState()


@click.command()
@click.option('--run', 'run_id', help='Specific run ID to analyze (default: last run)')
@click.option('--json', 'json_output', is_flag=True, help='Output JSON instead of rich text')
def analyze(run_id: Optional[str], json_output: bool):
    """Analyze failures from the last run.
    
    This command:
    1. Loads results from the specified or last run
    2. Clusters failures by pattern
    3. Identifies root causes
    4. Shows actionable insights
    
    Example:
        arc analyze
        arc analyze --run run_20240110_143022_abc123
    """
    try:
        # Load run results
        run_result = state.get_run(run_id)
        if not run_result:
            if json_output:
                import json
                print(json.dumps({"error": "No run found"}, indent=2))
            else:
                console.print(format_error("No run found. Run 'arc run' first."))
            raise click.Exit(1)
        
        # Perform analysis
        failures = run_result.failures
        if not failures:
            if json_output:
                import json
                print(json.dumps({"message": "No failures to analyze", "run_id": run_result.run_id}, indent=2))
            else:
                console.print(format_success("No failures found in this run!"))
                console.print(f"Reliability: {run_result.reliability_percentage:.1f}%")
            return
        
        # Cluster failures (simplified for now)
        failure_clusters = _cluster_failures(failures)
        
        # Save analysis
        analysis = {
            "run_id": run_result.run_id,
            "total_failures": len(failures),
            "failure_rate": 1 - run_result.reliability_score,
            "clusters": failure_clusters,
            "top_issues": _identify_top_issues(failure_clusters)
        }
        state.save_analysis(run_result.run_id, analysis)
        
        # Display results
        if json_output:
            import json
            print(json.dumps(analysis, indent=2))
        else:
            console.print()
            console.print_header(f"Failure Analysis for {run_result.config_path}")
            
            console.print_metric("Run ID", run_result.run_id, style="muted")
            console.print_metric("Total failures", f"{len(failures)} / {run_result.scenario_count}")
            console.print_metric("Failure rate", f"{(1 - run_result.reliability_score):.1%}")
            console.print()
            
            # Failure clusters table
            console.print("[primary]Failure Clusters[/primary]")
            console.print()
            
            table = Table(show_header=True, header_style="bold")
            table.add_column("Pattern", style="error")
            table.add_column("Count", justify="right")
            table.add_column("Impact", style="warning")
            table.add_column("Example", style="muted", max_width=50)
            
            for cluster in failure_clusters:
                table.add_row(
                    cluster["pattern"],
                    str(cluster["count"]),
                    cluster["impact"],
                    cluster["example"]
                )
            
            console.print(table)
            console.print()
            
            # Top issues
            console.print("[primary]Top Issues to Address[/primary]")
            console.print()
            for i, issue in enumerate(analysis["top_issues"], 1):
                console.print(f"{i}. [error]{issue['title']}[/error]")
                console.print(f"   {issue['description']}", style="muted")
                console.print()
            
            console.print("Run [info]arc recommend[/info] for specific configuration improvements", style="muted")
            console.print()
    
    except Exception as e:
        if json_output:
            import json
            print(json.dumps({"error": str(e)}, indent=2))
        else:
            console.print(format_error(f"Analysis failed: {str(e)}"))
        raise click.Exit(1)


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


def _identify_top_issues(clusters: List[Dict]) -> List[Dict]:
    """Identify top issues from failure clusters."""
    issues = []
    
    for cluster in clusters[:3]:  # Top 3 issues
        if cluster["pattern"] == "Currency Assumption Violation":
            issues.append({
                "title": "Hard-coded Currency Assumptions",
                "description": "Agent assumes USD for all transactions. Add multi-currency support.",
                "severity": "high",
                "recommendation": "Implement currency detection and conversion logic"
            })
        elif cluster["pattern"] == "Timeout/Performance Issue":
            issues.append({
                "title": "Slow External API Calls",
                "description": "Multiple scenarios timing out on API calls. Consider caching or async calls.",
                "severity": "medium",
                "recommendation": "Add request timeouts and retry logic"
            })
        elif cluster["pattern"] == "Tool Execution Error":
            issues.append({
                "title": "Tool Configuration Issues",
                "description": "Tools failing to execute properly. Check tool definitions and parameters.",
                "severity": "medium",
                "recommendation": "Validate tool configurations and add error handling"
            })
        else:
            issues.append({
                "title": cluster["pattern"],
                "description": f"{cluster['count']} failures detected in this category",
                "severity": "low",
                "recommendation": "Review individual failures for patterns"
            })
    
    return issues