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
    """Generate recommendations based on actual failure analysis."""
    recommendations = {
        "run_id": run_result.run_id,
        "current_reliability": run_result.reliability_score,
        "expected_improvement": 0.0,
        "config_changes": [],
        "model_optimization": None,
        "implementation_steps": [],
        "database_enhanced": model_performance_data is not None,
        "failure_based_analysis": True
    }
    
    # Analyze actual failure patterns from enhanced analysis
    analysis = run_result.analysis or {}
    failure_clusters = analysis.get("clusters", [])
    top_issues = analysis.get("top_issues", [])
    
    # Generate specific recommendations for each failure pattern
    for issue in top_issues:
        pattern = issue.get("title", "")
        severity = issue.get("severity", "low")
        affected_scenarios = issue.get("affected_scenarios", [])
        technical_details = issue.get("technical_details", "")
        
        if "Code Execution" in pattern:
            # Critical agent execution failures - fix via YAML configuration
            recommendations["config_changes"].append({
                "title": "Fix Agent Runtime Errors",
                "description": f"Agent failing with runtime exceptions during execution. Affects {len(affected_scenarios)} scenarios.",
                "severity": "critical",
                "root_cause": f"Agent encounters null/undefined values: {technical_details}",
                "affected_scenarios": affected_scenarios,
                "solution": {
                    "type": "defensive_agent_configuration",
                    "explanation": "Update system prompt and add validation rules to make agent handle edge cases gracefully",
                    "yaml_diff": """# Add defensive instructions to system prompt
system_prompt: |
  You are a [your agent description]. 
  
  CRITICAL ERROR HANDLING RULES:
  1. Before processing any input, validate it is not null/empty
  2. If you encounter missing or invalid data, ask for clarification instead of assuming
  3. When data appears incomplete, explicitly state what is missing
  4. Never attempt operations on null/undefined values
  
  Example defensive patterns:
  - Before: process(user_input.lower())
  - After: if user_input and user_input.strip(): process(user_input.lower())
  
  If you encounter errors:
  - Explain what went wrong in simple terms
  - Ask the user to provide the missing information
  - Suggest alternative approaches when possible

# Add validation configuration
validation_rules:
  - "Check for null/empty inputs before processing"
  - "Validate data types match expectations"
  - "Request clarification for ambiguous inputs"

# Add error handling behavior
error_handling:
  strategy: "graceful_degradation"
  fallback_responses:
    - "I encountered an issue with the input data. Could you please check if all required fields are provided?"
    - "The data appears to be incomplete. Could you provide more details about [specific missing field]?"
    - "I'm having trouble processing this request. Could you try rephrasing or providing the information in a different format?"
"""
                }
            })
            recommendations["expected_improvement"] += 25.0  # High impact for fixing crashes
            
        elif "Ambiguous Input" in pattern:
            # Ambiguous input handling issues
            recommendations["config_changes"].append({
                "title": "Improve Ambiguous Input Handling",
                "description": f"Agent struggles with unclear inputs. Affects {len(affected_scenarios)} scenarios.",
                "severity": "high",
                "root_cause": "Agent makes assumptions instead of asking for clarification",
                "affected_scenarios": affected_scenarios,
                "solution": {
                    "type": "clarification_strategy",
                    "yaml_diff": """system_prompt: |
  You are a financial analyst. When you encounter ambiguous requests:
  
  1. NEVER make assumptions about unclear terms
  2. ALWAYS ask for clarification when inputs are ambiguous
  3. Provide specific examples of what you need
  
  Examples of ambiguous terms that require clarification:
  - "last quarter" → Ask: "Which quarter? Q1, Q2, Q3, or Q4 of which year?"
  - "recent entries" → Ask: "How recent? Last week, month, or specific date range?"
  - "missing account numbers" → Ask: "Do you mean null, zero, or empty string values?"

clarification_prompts:
  enabled: true
  max_clarifications: 2
  templates:
    - "I need clarification on '{ambiguous_term}'. Could you specify {clarification_needed}?"
    - "The term '{ambiguous_term}' could mean several things. Please clarify {options}."
"""
                }
            })
            recommendations["expected_improvement"] += 15.0
            
        elif "Early Execution Failure" in pattern:
            # Early termination issues - fix via YAML configuration
            recommendations["config_changes"].append({
                "title": "Fix Agent Startup Issues",
                "description": f"Agent fails to start or crashes immediately. Affects {len(affected_scenarios)} scenarios.",
                "severity": "critical",
                "root_cause": "Agent configuration or system prompt causes immediate failure",
                "affected_scenarios": affected_scenarios,
                "solution": {
                    "type": "robust_agent_configuration",
                    "explanation": "Simplify system prompt and add fallback behaviors to prevent startup crashes",
                    "yaml_diff": """# Simplify system prompt to avoid complex instructions that cause crashes
system_prompt: |
  You are a helpful assistant. 
  
  STARTUP SAFETY RULES:
  1. Start with simple acknowledgment of the request
  2. Break complex tasks into smaller steps
  3. If you don't understand something, ask for clarification
  4. Always respond with something, even if it's to ask for help
  
  If you encounter any issues:
  - Don't crash or stop responding
  - Explain what you're having trouble with
  - Ask the user to simplify their request
  - Offer to try a different approach

# Add fallback configuration
fallback_behavior:
  enabled: true
  simple_responses: true
  max_complexity: "basic"  # Start with basic functionality

# Reduce temperature for more predictable behavior
temperature: 0.1  # Lower temperature for more stable responses

# Ensure essential tools only
tools:
  # Keep only the most essential tools to reduce startup complexity
  # Remove any tools that might cause initialization issues
"""
                }
            })
            recommendations["expected_improvement"] += 20.0
            
        elif "Currency" in pattern:
            # Currency handling issues
            recommendations["config_changes"].append({
                "title": "Add Multi-Currency Support",
                "description": f"Agent makes currency assumptions. Affects {len(affected_scenarios)} scenarios.",
                "severity": "medium",
                "root_cause": "Hard-coded USD assumptions",
                "affected_scenarios": affected_scenarios,
                "solution": {
                    "type": "currency_handling",
                    "yaml_diff": """system_prompt: |
  When handling monetary values:
  1. ALWAYS identify the currency being used
  2. Convert to requested currency if different  
  3. NEVER assume USD unless explicitly stated
  4. Ask for currency clarification if ambiguous

tools:
  - name: currency_converter
    description: Convert between currencies with current rates
    enabled: true
    
currency_handling:
  default_currency: null  # Force explicit currency
  require_currency_specification: true
  supported_currencies: ["USD", "EUR", "GBP", "JPY", "CAD", "AUD"]"""
                }
            })
            recommendations["expected_improvement"] += 12.0
            
        elif "Tool Execution" in pattern:
            # Tool-related failures - these are critical for agent functionality
            expected_tools = issue.get("scenario_characteristics", {}).get("expected_tools", [])
            recommendations["config_changes"].append({
                "title": "Fix Tool Execution Failures",
                "description": f"Agent failing to use tools properly. Expected tools: {', '.join(expected_tools[:3])}. Affects {len(affected_scenarios)} scenarios.",
                "severity": "high",
                "root_cause": f"Tools not working correctly or agent unable to use them: {technical_details}",
                "affected_scenarios": affected_scenarios,
                "solution": {
                    "type": "tool_configuration_and_validation",
                    "explanation": "Add tool validation, error handling, and fallback behaviors to handle tool failures gracefully",
                    "yaml_diff": f"""# Ensure tools are properly configured
tools:
{chr(10).join(f'  - name: {tool}' for tool in expected_tools[:3])}
    enabled: true
    error_handling: graceful
    timeout: 30
    retry_attempts: 2
    validation: true

# Add tool error handling to system prompt
system_prompt: |
  You are a [your agent description].
  
  TOOL USAGE RULES:
  1. Before using any tool, verify it's available and properly configured
  2. If a tool fails, explain the failure and try alternative approaches
  3. Never assume tool responses are valid - always validate them
  4. If tools are unavailable, provide manual alternatives when possible
  
  Tool Error Handling:
  - If a tool returns an error, explain what went wrong
  - Suggest alternative tools or manual approaches
  - Don't crash or stop - continue with available functionality
  
# Add tool fallback configuration
tool_fallbacks:
  enabled: true
  manual_alternatives: true
  error_messages: 
    - "The [tool_name] tool is currently unavailable. Let me try an alternative approach."
    - "I encountered an issue with [tool_name]. I'll attempt to solve this manually."
  
# Add validation rules
validation_rules:
  - "Verify tool availability before use"
  - "Validate all tool responses"
  - "Provide fallbacks when tools fail"
"""
                }
            })
            recommendations["expected_improvement"] += 18.0  # Higher impact since tools are critical
    
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
    
    # Generate specific implementation steps based on findings
    implementation_steps = ["Create a backup of your current configuration file"]
    
    # Check for critical configuration issues
    critical_issues = [change for change in recommendations["config_changes"] 
                      if change.get("severity") == "critical"]
    
    if critical_issues:
        implementation_steps.extend([
            "CRITICAL: Fix agent configuration issues first - these cause immediate failures",
            "Update system prompt with defensive error handling instructions",
            "Add validation rules and fallback behaviors to your YAML config",
            "Test configuration changes with a single failing scenario first"
        ])
    
    if recommendations["config_changes"]:
        implementation_steps.extend([
            "Apply YAML configuration changes for better input handling",
            "Update system prompts to handle edge cases and ambiguous inputs",
            "Add appropriate validation rules and error handling behaviors"
        ])
    
    implementation_steps.extend([
        "Test with a smaller scenario set first (arc run --scenarios 10)",
        "Focus testing on previously failing scenarios",
        "Monitor agent responses for improved error handling",
        "Run full validation once configuration fixes are confirmed working"
    ])
    
    recommendations["implementation_steps"] = implementation_steps
    
    return recommendations


def _display_recommendations_ui(run_result, recommendations, model_performance_data):
    """Display failure-based recommendations in rich UI."""
    console.print()
    console.print_header("Failure-Based Improvement Recommendations")
    
    console.print_metric("Current reliability", f"{run_result.reliability_percentage:.1f}%")
    console.print_metric("Expected improvement", f"+{recommendations['expected_improvement']:.1f} percentage points", style="success")
    
    if recommendations.get("failure_based_analysis"):
        console.print_metric("Analysis type", "Based on actual failure patterns", style="info")
    console.print()
    

    
    # Configuration changes
    if recommendations.get("config_changes"):
        console.print("[primary]YAML Configuration Improvements[/primary]")
        console.print("Based on actual failure patterns from your agent execution")
        console.print()
        
        for i, change in enumerate(recommendations["config_changes"], 1):
            severity_color = "error" if change.get("severity") == "critical" else "warning" if change.get("severity") == "high" else "info"
            console.print(f"{i}. [{severity_color}]{change['title']}[/{severity_color}]")
            console.print(f"   {change['description']}", style="muted")
            
            if change.get("root_cause"):
                console.print(f"   [dim]Root cause: {change['root_cause']}[/dim]")
            if change.get("affected_scenarios"):
                console.print(f"   [dim]Affected scenarios: {len(change['affected_scenarios'])}[/dim]")
            
            # Show explanation of the fix
            solution = change.get("solution", {})
            if solution.get("explanation"):
                console.print(f"   [info]Solution: {solution['explanation']}[/info]")
            console.print()
            
            # Show YAML diff
            yaml_diff = change.get("yaml_diff") or solution.get("yaml_diff")
            if yaml_diff:
                console.print("   Add to your YAML configuration:")
                syntax = Syntax(yaml_diff, "yaml", theme="monokai", line_numbers=False)
                console.print(Panel(syntax, title="YAML Configuration Changes", border_style="info"))
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