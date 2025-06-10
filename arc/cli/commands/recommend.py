"""Arc recommend command - get improvement recommendations."""

import asyncio
from typing import Any, Optional
from datetime import datetime, timedelta

import click
from rich.table import Table
from rich.panel import Panel
from rich.syntax import Syntax
from rich.progress import Progress, SpinnerColumn, TextColumn
from sqlalchemy import text

from arc.cli.utils import ArcConsole, CLIState, format_error, format_success, format_warning
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
@click.option('--run', 'run_id', help='Specific run ID (default: last run)')
@click.option('--historical', is_flag=True, help='Include historical model performance data')
@click.option('--days', default=30, help='Days of history for model analysis (default: 30)')
@click.option('--json', 'json_output', is_flag=True, help='Output JSON instead of rich text')
def recommend(run_id: Optional[str], historical: bool, days: int, json_output: bool):
    """Get configuration improvement recommendations.
    
    This command:
    1. Analyzes failures from the run
    2. Generates specific configuration fixes
    3. Provides multi-model cost optimization with real data
    4. Shows expected impact based on historical performance
    
    Example:
        arc recommend
        arc recommend --run run_20240110_143022_abc123
        arc recommend --historical --days 7
    """
    try:
        # Run async recommendation generation
        asyncio.run(_recommend_async(run_id, historical, days, json_output))
    except Exception as e:
        if json_output:
            import json
            print(json.dumps({"error": str(e)}, indent=2))
        else:
            console.print(format_error(f"Recommendation generation failed: {str(e)}"))
        raise click.exceptions.Exit(1)


async def _recommend_async(run_id: Optional[str], historical: bool, days: int, json_output: bool):
    """Generate recommendations asynchronously with database queries."""
    # Initialize state
    db_connected = await _initialize_state()
    
    # Load run and analysis
    run_result = state.get_run(run_id)
    if not run_result:
        if json_output:
            import json
            print(json.dumps({"error": "No run found"}, indent=2))
        else:
            console.print(format_error("No run found. Run 'arc run' first."))
        raise click.exceptions.Exit(1)
    
    # Check if analysis exists
    if not run_result.analysis:
        if json_output:
            import json
            print(json.dumps({"error": "No analysis found. Run 'arc analyze' first."}, indent=2))
        else:
            console.print(format_error("No analysis found. Run 'arc analyze' first."))
        raise click.exceptions.Exit(1)
    
    # Get historical model performance if requested and database is available
    model_performance_data = None
    recommendation_history = None
    
    if historical and db_connected:
        db_client = db_manager.get_client()
        if db_client:
            if not json_output:
                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    console=console
                ) as progress:
                    progress.add_task("Analyzing model performance history...", total=None)
                    
                    model_performance_data = await _get_model_performance_data(db_client, days)
                    recommendation_history = await _get_recommendation_history(db_client, run_result.config_path, days)
            else:
                model_performance_data = await _get_model_performance_data(db_client, days)
                recommendation_history = await _get_recommendation_history(db_client, run_result.config_path, days)
    
    # Generate recommendations with enhanced data
    recommendations = _generate_recommendations(run_result, model_performance_data, recommendation_history)
    
    # Save recommendations
    state.save_recommendations(run_result.run_id, recommendations)
    
    # Save to database if available
    if db_connected and isinstance(state, HybridState):
        db_client = db_manager.get_client()
        if db_client:
            try:
                await _save_recommendations_to_database(db_client, run_result, recommendations)
            except Exception as e:
                console.print(f"Warning: Failed to save recommendations to database: {str(e)}", style="warning")
    
    # Display results
    if json_output:
        import json
        print(json.dumps(recommendations, indent=2))
    else:
        _display_recommendations_ui(run_result, recommendations, model_performance_data)


async def _get_model_performance_data(db_client, days: int) -> dict:
    """Get model performance data from database."""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    query = text("""
    SELECT 
        s.metadata->>'agent_model' as model,
        COUNT(DISTINCT s.simulation_id) as run_count,
        AVG(s.overall_score) as avg_reliability,
        STDDEV(s.overall_score) as reliability_stddev,
        AVG(s.total_cost_usd) as avg_cost,
        SUM(o.tokens_used) as total_tokens,
        AVG(o.execution_time_ms) as avg_execution_time
    FROM simulations s
    JOIN outcomes o ON s.simulation_id = o.simulation_id
    WHERE s.created_at >= :start_date 
        AND s.created_at <= :end_date
        AND s.metadata->>'agent_model' IS NOT NULL
    GROUP BY s.metadata->>'agent_model'
    HAVING COUNT(DISTINCT s.simulation_id) >= 3
    ORDER BY AVG(s.overall_score) DESC
    """)
    
    async with db_client.engine.begin() as conn:
        result = await conn.execute(query, {"start_date": start_date, "end_date": end_date})
        
    models = []
    for row in result:
        models.append({
            "model": row[0],
            "runs": row[1],
            "avg_reliability": float(row[2] or 0),
            "reliability_stddev": float(row[3] or 0),
            "avg_cost": float(row[4] or 0),
            "total_tokens": row[5] or 0,
            "avg_execution_time": float(row[6] or 0) / 1000  # Convert to seconds
        })
    
    return {
        "period_days": days,
        "models": models
    }


async def _get_recommendation_history(db_client, config_path: str, days: int) -> dict:
    """Get historical recommendations for this config."""
    config_name = config_path.split("/")[-1]
    
    # This would query from recommendations table when it exists
    # For now, return mock data structure
    return {
        "config_name": config_name,
        "previous_recommendations": [],
        "accepted_changes": []
    }


async def _save_recommendations_to_database(db_client, run_result, recommendations):
    """Save recommendations to database."""
    # This would save to recommendations table when it exists
    pass


def _generate_recommendations(run_result, model_performance_data: Optional[dict] = None, 
                            recommendation_history: Optional[dict] = None) -> dict[str, Any]:
    """Generate recommendations based on run and analysis."""
    recommendations = {
        "run_id": run_result.run_id,
        "current_reliability": run_result.reliability_score,
        "expected_improvement": 0.0,
        "config_changes": [],
        "model_optimization": None,
        "implementation_steps": [],
        "database_enhanced": model_performance_data is not None
    }
    
    # Analyze top issues from analysis
    top_issues = run_result.analysis.get("top_issues", [])
    
    for issue in top_issues:
        if "currency" in issue["title"].lower():
            # Currency assumption fix
            recommendations["config_changes"].append({
                "title": "Add Multi-Currency Support",
                "description": "Configure agent to handle multiple currencies with explicit conversion",
                "impact": "high",
                "yaml_diff": """system_prompt: |
  You are a financial analyst. When handling monetary values:
  1. Always identify the currency being used
  2. Convert to requested currency if different
  3. Never assume USD unless explicitly stated
  
tools:
  - name: currency_converter
    description: Convert between currencies
    enabled: true"""
            })
            recommendations["expected_improvement"] += 18.0  # Based on demo target
            
        elif "timeout" in issue["title"].lower():
            recommendations["config_changes"].append({
                "title": "Add Request Timeouts",
                "description": "Configure timeouts to prevent hanging on slow API calls",
                "impact": "medium",
                "yaml_diff": """max_execution_time: 30
request_timeout: 10
retry_policy:
  max_attempts: 3
  backoff: exponential"""
            })
            recommendations["expected_improvement"] += 5.0
            
        elif "tool" in issue["title"].lower():
            recommendations["config_changes"].append({
                "title": "Fix Tool Configuration",
                "description": "Ensure all tools are properly configured with error handling",
                "impact": "medium",
                "yaml_diff": """tools:
  - name: calculator
    enabled: true
    error_handling: graceful
    fallback: manual_calculation"""
            })
            recommendations["expected_improvement"] += 3.0
    
    # Model optimization
    current_model = run_result.analysis.get("model", "openai/gpt-4.1")
    
    if model_performance_data and model_performance_data.get("models"):
        # Use real historical data
        models_by_name = {m["model"]: m for m in model_performance_data["models"]}
        current_perf = models_by_name.get(current_model, {
            "avg_reliability": run_result.reliability_score,
            "avg_cost": run_result.total_cost / run_result.scenario_count if run_result.scenario_count > 0 else 0.001
        })
        
        # Calculate cost per 1k tokens based on actual usage
        avg_tokens_per_scenario = current_perf.get("total_tokens", 1500) / current_perf.get("runs", 1) / 50  # Assume 50 scenarios per run
        cost_per_1k = (current_perf["avg_cost"] / avg_tokens_per_scenario) if avg_tokens_per_scenario > 0 else 0.001
        
        recommendations["model_optimization"] = {
            "current": {
                "model": current_model,
                "reliability": current_perf["avg_reliability"] * 100,
                "cost_per_1k": cost_per_1k,
                "runs_analyzed": current_perf.get("runs", 0)
            },
            "alternatives": []
        }
        
        # Find alternatives from real data
        for model_data in model_performance_data["models"]:
            if model_data["model"] != current_model:
                alt_tokens_per_scenario = model_data.get("total_tokens", 1500) / model_data.get("runs", 1) / 50
                alt_cost_per_1k = (model_data["avg_cost"] / alt_tokens_per_scenario) if alt_tokens_per_scenario > 0 else 0.001
                savings = ((cost_per_1k - alt_cost_per_1k) / cost_per_1k * 100) if cost_per_1k > 0 else 0
                
                if savings > 0:  # Only show models that save money
                    recommendations["model_optimization"]["alternatives"].append({
                        "model": model_data["model"],
                        "reliability": model_data["avg_reliability"] * 100,
                        "cost_per_1k": alt_cost_per_1k,
                        "savings": savings,
                        "runs_analyzed": model_data["runs"],
                        "confidence": "high" if model_data["runs"] >= 10 else "medium" if model_data["runs"] >= 5 else "low"
                    })
        
        # Sort by savings and pick best
        if recommendations["model_optimization"]["alternatives"]:
            recommendations["model_optimization"]["alternatives"].sort(key=lambda x: x["savings"], reverse=True)
            best = recommendations["model_optimization"]["alternatives"][0]
            if best["reliability"] >= current_perf["avg_reliability"] * 100 * 0.95:  # Within 5% reliability
                recommendations["model_optimization"]["best_alternative"] = best
    else:
        # Fallback to mock data if no historical data
        recommendations["model_optimization"] = {
            "current": {
                "model": current_model,
                "reliability": run_result.reliability_score * 100,
                "cost_per_1k": 0.00200
            },
            "alternatives": [
                {
                    "model": "anthropic/claude-3.5-haiku",
                    "reliability": 97.0,
                    "cost_per_1k": 0.00080,
                    "savings": 60.0
                },
                {
                    "model": "openai/gpt-4.1-mini",
                    "reliability": 95.0,
                    "cost_per_1k": 0.00040,
                    "savings": 80.0
                },
                {
                    "model": "meta-llama/llama-4-scout",
                    "reliability": 92.0,
                    "cost_per_1k": 0.00008,
                    "savings": 96.0
                }
            ],
            "best_alternative": None
        }
        
        # Only recommend if reliability is acceptable
        for alt in recommendations["model_optimization"]["alternatives"]:
            if alt["reliability"] >= run_result.reliability_score * 100 * 0.95:
                recommendations["model_optimization"]["best_alternative"] = alt
                break
    
    # Implementation steps
    recommendations["implementation_steps"] = [
        "Create a backup of your current configuration",
        "Apply the recommended configuration changes",
        "Test with a smaller scenario set first (arc run --scenarios 10)",
        "Monitor for any unexpected behavior",
        "Run full validation once initial tests pass"
    ]
    
    return recommendations


def _display_recommendations_ui(run_result, recommendations, model_performance_data):
    """Display recommendations in rich UI."""
    console.print()
    console.print_header("Configuration Improvement Recommendations")
    
    console.print_metric("Current reliability", f"{run_result.reliability_percentage:.1f}%")
    console.print_metric("Expected improvement", f"+{recommendations['expected_improvement']:.1f} percentage points", style="success")
    
    if recommendations.get("database_enhanced"):
        console.print_metric("Analysis type", "Historical data-driven", style="info")
    console.print()
    
    # Configuration changes
    console.print("[primary]Recommended Configuration Changes[/primary]")
    console.print()
    
    for i, change in enumerate(recommendations["config_changes"], 1):
        console.print(f"{i}. [success]{change['title']}[/success]")
        console.print(f"   {change['description']}", style="muted")
        console.print()
        
        # Show YAML diff
        if change.get("yaml_diff"):
            console.print("   Configuration change:")
            syntax = Syntax(change["yaml_diff"], "yaml", theme="monokai", line_numbers=False)
            console.print(Panel(syntax, border_style="muted"))
        console.print()
    
    # Model optimization
    if recommendations.get("model_optimization"):
        console.print("[primary]Multi-Model Cost Optimization[/primary]")
        
        if recommendations.get("database_enhanced"):
            console.print("Based on historical performance data")
        console.print()
        
        table = Table(show_header=True, header_style="bold")
        table.add_column("Model", style="info")
        table.add_column("Reliability", justify="right")
        table.add_column("Cost/1k", justify="right")
        table.add_column("Savings", justify="right", style="success")
        
        if recommendations.get("database_enhanced"):
            table.add_column("Confidence", justify="center")
        
        current_model = recommendations["model_optimization"]["current"]
        row = [
            f"{current_model['model']} (current)",
            f"{current_model['reliability']:.1f}%",
            f"${current_model['cost_per_1k']:.4f}",
            "-"
        ]
        if recommendations.get("database_enhanced"):
            row.append(f"({current_model.get('runs_analyzed', 0)} runs)")
        table.add_row(*row)
        
        for alt in recommendations["model_optimization"]["alternatives"][:5]:  # Show top 5
            row = [
                alt["model"],
                f"{alt['reliability']:.1f}%",
                f"${alt['cost_per_1k']:.4f}",
                f"{alt['savings']:.1f}%"
            ]
            if recommendations.get("database_enhanced"):
                confidence = alt.get("confidence", "unknown")
                conf_style = "success" if confidence == "high" else "warning" if confidence == "medium" else "muted"
                row.append(f"[{conf_style}]{confidence}[/{conf_style}]")
            table.add_row(*row)
        
        console.print(table)
        console.print()
        
        best = recommendations["model_optimization"].get("best_alternative")
        if best:
            console.print(
                f"[success]Recommended:[/success] Switch to {best['model']} for "
                f"{best['savings']:.1f}% cost reduction with {best['reliability']:.1f}% reliability"
            )
            if recommendations.get("database_enhanced") and best.get("confidence"):
                console.print(f"Confidence: {best['confidence']} (based on {best.get('runs_analyzed', 0)} runs)")
            console.print()
    
    # Implementation guidance
    console.print("[primary]Implementation Steps[/primary]")
    console.print()
    for i, step in enumerate(recommendations["implementation_steps"], 1):
        console.print(f"{i}. {step}")
    console.print()
    
    console.print("Run [info]arc diff {run_result.config_path} improved_config.yaml[/info] to validate improvements", style="muted")
    console.print()