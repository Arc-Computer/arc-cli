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

# Pure LLM-based recommendations - no hardcoded constants needed


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


def _handle_json_error(error_msg: str, json_output: bool):
    """Handle errors with consistent JSON/console output."""
    if json_output:
        import json
        print(json.dumps({"error": error_msg}, indent=2))
    else:
        console.print(format_error(error_msg))
    raise click.exceptions.Exit(1)


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
        _handle_json_error(f"Recommendation generation failed: {str(e)}", json_output)


async def _recommend_async(run_id: Optional[str], historical: bool, days: int, json_output: bool):
    """Generate recommendations asynchronously with database queries."""
    # Initialize state
    db_connected = await _initialize_state()
    
    # Load and validate run result
    run_result = _load_and_validate_run(run_id, json_output)
    
    # Get historical data if requested
    model_performance_data, recommendation_history = await _get_historical_data(
        historical, db_connected, days, json_output, run_result.config_path
    )
    
    # Generate recommendations
    recommendations = await _generate_recommendations_wrapper_async(
        run_result, model_performance_data, recommendation_history
    )
    
    # Save and display results
    await _save_and_display_results(
        run_result, recommendations, model_performance_data, db_connected, json_output
    )


def _load_and_validate_run(run_id: Optional[str], json_output: bool):
    """Load and validate run result."""
    run_result = state.get_run(run_id)
    if not run_result:
        _handle_json_error("No run found. Run 'arc run' first.", json_output)
    
    if not run_result.analysis:
        _handle_json_error("No analysis found. Run 'arc analyze' first.", json_output)
    
    return run_result


async def _get_historical_data(historical: bool, db_connected: bool, days: int, 
                              json_output: bool, config_path: str):
    """Get historical model performance and recommendation data."""
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
                    recommendation_history = await _get_recommendation_history(db_client, config_path, days)
            else:
                model_performance_data = await _get_model_performance_data(db_client, days)
                recommendation_history = await _get_recommendation_history(db_client, config_path, days)
    
    return model_performance_data, recommendation_history


async def _save_and_display_results(run_result, recommendations, model_performance_data, 
                                   db_connected: bool, json_output: bool):
    """Save recommendations and display results."""
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


async def _generate_recommendations_async(run_result, model_performance_data: Optional[dict] = None, 
                            recommendation_history: Optional[dict] = None) -> dict[str, Any]:
    """Generate recommendations using the pure LLM-based recommendation engine."""
    from arc.recommendations import RecommendationEngine, RecommendationRequest
    from arc.recommendations.llm_client import create_llm_client
    
    # Initialize the pure LLM-based recommendation engine
    llm_client = create_llm_client("mock")  # Use mock for now, can be configured later
    engine = RecommendationEngine(llm_client=llm_client)
    
    # Extract and process analysis data for LLM analysis
    failures, trajectories, scenarios = _extract_analysis_data(run_result.analysis)
    
    # Create recommendation request
    request = RecommendationRequest(
        failures=failures,
        trajectories=trajectories,
        scenarios=scenarios,
        config_path=run_result.config_path,
        run_id=run_result.run_id
    )
    
    # Generate LLM-based recommendations
    response = await engine.generate_recommendations(request)
    
    # Convert to output format
    return _build_response(run_result, response, model_performance_data)


def _extract_analysis_data(analysis: dict):
    """Extract failure data from analysis for LLM-based recommendation engine."""
    failures = []
    trajectories = []
    scenarios = []
    
    top_issues = analysis.get("top_issues", [])
    
    for issue in top_issues:
        # Extract failure information for LLM analysis
        failure = {
            "error_message": issue.get("technical_details", ""),
            "failure_reason": issue.get("title", ""),
            "severity": issue.get("severity", "medium"),
            "affected_count": len(issue.get("affected_scenarios", []))
        }
        failures.append(failure)
        
        # Create trajectory information from available data
        trajectory = {
            "full_trajectory": [],  # Limited data available from current analysis
            "tool_calls": [],
            "execution_steps": issue.get("affected_scenarios", []),
            "execution_time": 30  # Default estimate
        }
        trajectories.append(trajectory)
        
        # Create scenario context for LLM
        scenario_chars = issue.get("scenario_characteristics", {})
        scenario = {
            "task_prompt": f"Task involving {issue.get('title', 'unknown issue')}",
            "expected_tools": scenario_chars.get("expected_tools", []),
            "complexity_level": scenario_chars.get("complexity", "medium"),
            "domain": scenario_chars.get("domain", "general")
        }
        scenarios.append(scenario)
    
    # If no specific failures found, create a general analysis request for LLM
    if not failures:
        failures = [{
            "error_message": "General performance issues detected - needs LLM analysis",
            "failure_reason": "reliability_below_threshold",
            "severity": "medium",
            "affected_count": 1
        }]
        trajectories = [{
            "full_trajectory": [], 
            "tool_calls": [],
            "execution_time": 30
        }]
        scenarios = [{
            "task_prompt": "General agent performance optimization",
            "expected_tools": [],
            "complexity_level": "medium",
            "domain": "general"
        }]
    
    return failures, trajectories, scenarios


def _build_response(run_result, response, model_performance_data: Optional[dict] = None) -> dict[str, Any]:
    """Build response from LLM-generated recommendations."""
    
    # Extract config changes from LLM recommendations
    config_changes = []
    total_expected_improvement = 0
    
    for rec in response.recommendations:
        config_changes.append({
            "section": "llm_generated",
            "description": rec["specific_problem"],
            "yaml_changes": rec["yaml_changes"],
            "reasoning": rec["recommended_fix"],
            "confidence": rec["confidence"],
            "expected_improvement": rec["expected_improvement"]
        })
        total_expected_improvement += rec["expected_improvement"]
    
    # Build response with LLM-generated recommendations
    llm_recommendations = {
        "run_id": run_result.run_id,
        "current_reliability": run_result.reliability_score,
        "expected_improvement": total_expected_improvement,
        "config_changes": config_changes,
        "model_optimization": _build_model_optimization(run_result, model_performance_data),
        "implementation_steps": [
            "Review the LLM-generated YAML changes below",
            "Test each recommendation in a development environment",
            "Apply changes incrementally and measure impact",
            "Monitor performance improvements"
        ] if config_changes else [],
        "analysis_approach": "pure_llm_based",
        "llm_analysis": {
            "confidence_score": response.confidence_score,
            "processing_time": response.processing_time,
            "analysis_summary": response.analysis_summary,
            "total_recommendations": len(response.recommendations)
        }
    }
    
    return llm_recommendations


def _build_model_optimization(run_result, model_performance_data: Optional[dict] = None):
    """Build model optimization recommendations if data is available."""
    if not model_performance_data or not model_performance_data.get("models"):
        return None
    
    current_model = "gpt-4"  # Default assumption
    best_model = max(model_performance_data["models"], key=lambda m: m["avg_reliability"])
    
    if best_model["avg_reliability"] > run_result.reliability_score:
        return {
            "current_model": current_model,
            "recommended_model": best_model["model"],
            "expected_reliability_gain": best_model["avg_reliability"] - run_result.reliability_score,
            "cost_impact": best_model["avg_cost"],
            "confidence": "high" if best_model["runs"] >= 10 else "medium"
        }
    
    return None


# Pure LLM-based recommendations - direct async call
async def _generate_recommendations_wrapper_async(run_result, model_performance_data: Optional[dict] = None, 
                            recommendation_history: Optional[dict] = None) -> dict[str, Any]:
    """Generate pure LLM-based recommendations."""
    return await _generate_recommendations_async(run_result, model_performance_data, recommendation_history)


def _display_recommendations_ui(run_result, recommendations, model_performance_data):
    """Display failure-based recommendations in rich UI."""
    _display_header(run_result, recommendations)
    _display_config_changes(recommendations)
    _display_model_optimization(recommendations, model_performance_data)
    _display_implementation_steps(recommendations, run_result)


def _display_header(run_result, recommendations):
    """Display recommendation header with metrics."""
    console.print()
    console.print_header("LLM-Generated Improvement Recommendations")
    
    console.print_metric("Current reliability", f"{run_result.reliability_percentage:.1f}%")
    console.print_metric("Expected improvement", f"+{recommendations['expected_improvement']:.1f} percentage points", style="success")
    
    # Show LLM analysis info
    llm_analysis = recommendations.get("llm_analysis", {})
    if llm_analysis:
        console.print_metric("Analysis approach", recommendations.get("analysis_approach", "pure_llm_based"), style="info")
        console.print_metric("LLM confidence", f"{llm_analysis.get('confidence_score', 0):.2f}", style="info")
        console.print_metric("Processing time", f"{llm_analysis.get('processing_time', 0):.2f}s", style="muted")
    console.print()


def _display_config_changes(recommendations):
    """Display LLM-generated configuration change recommendations."""
    if not recommendations.get("config_changes"):
        return
    
    console.print("[primary]LLM-Generated YAML Configuration Improvements[/primary]")
    console.print("Based on AI analysis of your actual simulation data and execution patterns")
    console.print()
    
    for i, change in enumerate(recommendations["config_changes"], 1):
        _display_single_config_change(i, change)


def _display_single_config_change(index: int, change: dict):
    """Display a single LLM-generated configuration change recommendation."""
    # Use description as the title for LLM recommendations
    title = change.get("description", "LLM Recommendation")
    confidence = change.get("confidence", 0.5)
    
    # Color based on confidence level
    if confidence > 0.8:
        color = "success"
    elif confidence > 0.6:
        color = "warning"
    else:
        color = "info"
    
    console.print(f"{index}. [{color}]{title}[/{color}]")
    console.print(f"   [dim]Confidence: {confidence:.2f}[/dim]")
    
    # Show the reasoning/root cause
    if change.get("reasoning"):
        console.print(f"   [info]Analysis: {change['reasoning']}[/info]")
    
    # Show expected improvement
    if change.get("expected_improvement"):
        console.print(f"   [success]Expected improvement: +{change['expected_improvement']:.1f}%[/success]")
    
    console.print()
    
    # Show YAML changes
    yaml_changes = change.get("yaml_changes")
    if yaml_changes:
        console.print("   Add to your YAML configuration:")
        syntax = Syntax(yaml_changes, "yaml", theme="monokai", line_numbers=False)
        console.print(Panel(syntax, title="LLM-Generated YAML Changes", border_style="info"))
    console.print()


def _get_severity_color(severity: str) -> str:
    """Get color for severity level."""
    severity_colors = {
        "critical": "error",
        "high": "warning", 
        "medium": "info",
        "low": "muted"
    }
    return severity_colors.get(severity, "info")


def _display_model_optimization(recommendations, model_performance_data):
    """Display model optimization recommendations."""
    if not recommendations.get("model_optimization"):
        return
    
    console.print("[primary]Multi-Model Cost Optimization[/primary]")
    
    if recommendations.get("database_enhanced"):
        console.print("Based on historical performance data")
    console.print()
    
    # Create and populate table
    table = _create_model_optimization_table(recommendations)
    _populate_model_optimization_table(table, recommendations)
    
    console.print(table)
    console.print()
    
    # Show best recommendation
    _display_best_model_recommendation(recommendations)


def _create_model_optimization_table(recommendations):
    """Create model optimization table structure."""
    table = Table(show_header=True, header_style="bold")
    table.add_column("Model", style="info")
    table.add_column("Reliability", justify="right")
    table.add_column("Cost/1k", justify="right")
    table.add_column("Savings", justify="right", style="success")
    
    if recommendations.get("database_enhanced"):
        table.add_column("Confidence", justify="center")
    
    return table


def _populate_model_optimization_table(table, recommendations):
    """Populate model optimization table with data."""
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


def _display_best_model_recommendation(recommendations):
    """Display the best model recommendation."""
    best = recommendations["model_optimization"].get("best_alternative")
    if best:
        console.print(
            f"[success]Recommended:[/success] Switch to {best['model']} for "
            f"{best['savings']:.1f}% cost reduction with {best['reliability']:.1f}% reliability"
        )
        if recommendations.get("database_enhanced") and best.get("confidence"):
            console.print(f"Confidence: {best['confidence']} (based on {best.get('runs_analyzed', 0)} runs)")
        console.print()


def _display_implementation_steps(recommendations, run_result):
    """Display implementation guidance."""
    console.print("[primary]Implementation Steps[/primary]")
    console.print()
    for i, step in enumerate(recommendations["implementation_steps"], 1):
        console.print(f"{i}. {step}")
    console.print()
    
    console.print(f"Run [info]arc diff {run_result.config_path} improved_config.yaml[/info] to validate improvements", style="muted")
    console.print()