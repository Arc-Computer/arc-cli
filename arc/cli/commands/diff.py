"""Arc diff command - compare two agent configurations."""

import asyncio
from typing import Any, Dict, Optional, List
from datetime import datetime
from uuid import uuid4

import click
from rich.table import Table
from rich.panel import Panel
import numpy as np
from scipy.stats import chi2_contingency
from sqlalchemy import text
try:
    from statsmodels.stats.power import ttest_power
except ImportError:
    ttest_power = None

from arc.cli.utils import ArcConsole, CLIState, format_error, format_success, format_warning
from arc.cli.utils import db_manager, HybridState
from arc.cli.commands.run import _generate_scenarios_async, _estimate_cost, _simulate_execution, _execute_with_modal, _check_modal_available
from arc.ingestion.parser import AgentConfigParser
from arc.ingestion.normalizer import ConfigNormalizer

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
@click.argument('config1', type=click.Path(exists=True))
@click.argument('config2', type=click.Path(exists=True))
@click.option('--scenarios', '-s', default=50, help='Number of scenarios per config (default: 50)')
@click.option('--historical', is_flag=True, help='Include historical A/B test comparisons')
@click.option('--modal/--no-modal', default=True, help='Use Modal for execution (default: True)')
@click.option('--json', 'json_output', is_flag=True, help='Output JSON instead of rich text')
def diff(config1: str, config2: str, scenarios: int, historical: bool, modal: bool, json_output: bool):
    """Compare two agent configurations with A/B testing.
    
    This command:
    1. Runs both configurations on the same scenarios
    2. Performs statistical significance testing
    3. Shows improvement metrics
    4. Validates claimed improvements
    5. Stores results in database for historical tracking
    
    Example:
        arc diff finance_agent_v1.yaml finance_agent_v2.yaml
        arc diff config_a.yaml config_b.yaml --scenarios 100 --historical
    """
    try:
        # Run async diff
        asyncio.run(_diff_async(config1, config2, scenarios, historical, modal, json_output))
    except Exception as e:
        if json_output:
            import json
            print(json.dumps({"error": str(e)}, indent=2))
        else:
            console.print(format_error(f"Comparison failed: {str(e)}"))
        raise click.Exit(1)


async def _diff_async(config1: str, config2: str, scenarios: int, historical: bool, modal: bool, json_output: bool):
    """Perform async diff with database integration."""
    # Initialize state
    db_connected = await _initialize_state()
    
    if not json_output:
        console.print()
        console.print_header("A/B Configuration Comparison")
    
    # Parse both configurations
    parser = AgentConfigParser()
    normalizer = ConfigNormalizer()
    
    config1_parsed = parser.parse(config1)
    config1_normalized = normalizer.normalize(config1_parsed)
    
    config2_parsed = parser.parse(config2)
    config2_normalized = normalizer.normalize(config2_parsed)
    
    if not json_output:
        console.print_metric("Config A", config1)
        console.print_metric("Config B", config2)
        console.print_metric("Scenarios per config", scenarios)
        console.print_metric("Execution mode", "Modal" if modal and _check_modal_available() else "Local simulation")
        console.print()
    
    # Get historical comparisons if requested
    historical_comparisons = None
    if historical and db_connected:
        db_client = db_manager.get_client()
        if db_client:
            historical_comparisons = await _get_historical_comparisons(
                db_client, 
                config1.split("/")[-1], 
                config2.split("/")[-1]
            )
    
    # Generate scenarios (same for both configs)
    if not json_output:
        console.print("Generating test scenarios...", style="muted")
    
    test_scenarios = await _generate_scenarios_async(
        config_path=config1,
        count=scenarios,
        pattern_ratio=0.7,
        capabilities=parser.extract_capabilities(config1_parsed),
        json_output=json_output
    )
    
    # Create unique diff session ID
    diff_id = f"diff_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid4().hex[:8]}"
    
    # Execute both configurations
    use_modal = modal and _check_modal_available()
    
    if not json_output:
        console.print()
        console.print("[primary]Running Config A[/primary]")
    
    if use_modal:
        results_a, time_a, cost_a = _execute_with_modal(test_scenarios, config1_normalized, json_output)
    else:
        results_a, time_a = _simulate_execution(test_scenarios, json_output)
        cost_a = _estimate_cost(scenarios, config1_normalized.get("model", "unknown"))
    
    success_a = sum(1 for r in results_a if r["success"])
    reliability_a = success_a / len(results_a)
    
    if not json_output:
        console.print(format_success(f"Config A: {reliability_a:.1%} reliability"))
        console.print()
        console.print("[primary]Running Config B[/primary]")
    
    if use_modal:
        results_b, time_b, cost_b = _execute_with_modal(test_scenarios, config2_normalized, json_output)
    else:
        results_b, time_b = _simulate_execution(test_scenarios, json_output)
        cost_b = _estimate_cost(scenarios, config2_normalized.get("model", "unknown"))
    
    success_b = sum(1 for r in results_b if r["success"])
    reliability_b = success_b / len(results_b)
    
    if not json_output:
        console.print(format_success(f"Config B: {reliability_b:.1%} reliability"))
        console.print()
    
    # Perform statistical analysis
    analysis = _perform_statistical_analysis(results_a, results_b)
        
    
    # Create diff result
    diff_result = {
        "diff_id": diff_id,
        "timestamp": datetime.now().isoformat(),
        "config_a": {
            "path": config1,
            "model": config1_normalized.get("model", "unknown"),
            "reliability": reliability_a,
            "successes": success_a,
            "failures": len(results_a) - success_a,
            "cost": cost_a,
            "execution_time": time_a
        },
        "config_b": {
            "path": config2,
            "model": config2_normalized.get("model", "unknown"),
            "reliability": reliability_b,
            "successes": success_b,
            "failures": len(results_b) - success_b,
            "cost": cost_b,
            "execution_time": time_b
        },
        "comparison": {
            "improvement": reliability_b - reliability_a,
            "improvement_percentage": (reliability_b - reliability_a) * 100,
            "relative_improvement": ((reliability_b - reliability_a) / reliability_a * 100) if reliability_a > 0 else 0,
            "cost_difference": cost_b - cost_a
        },
        "statistical_analysis": analysis,
        "scenario_count": scenarios,
        "execution_mode": "modal" if use_modal else "simulation"
    }
    
    # Save diff result
    state.save_diff(diff_id, diff_result)
    
    # Save to database if available
    if db_connected and isinstance(state, HybridState):
        db_client = db_manager.get_client()
        if db_client:
            try:
                await _save_diff_to_database(db_client, diff_result, results_a, results_b)
            except Exception as e:
                console.print(f"Warning: Failed to save diff to database: {str(e)}", style="warning")
    
    # Display results
    if json_output:
        import json
        if historical_comparisons:
            diff_result["historical_comparisons"] = historical_comparisons
        print(json.dumps(diff_result, indent=2))
    else:
            # Results comparison table
            console.print_header("Results Comparison")
            
            table = Table(show_header=True, header_style="bold")
            table.add_column("Metric", style="muted")
            table.add_column("Config A", justify="right")
            table.add_column("Config B", justify="right")
            table.add_column("Difference", justify="right")
            
            # Reliability
            rel_diff = reliability_b - reliability_a
            rel_style = "success" if rel_diff > 0 else "error" if rel_diff < 0 else "muted"
            table.add_row(
                "Reliability",
                f"{reliability_a:.1%}",
                f"{reliability_b:.1%}",
                f"{rel_diff:+.1%}"
            )
            
            # Success/Failure counts
            table.add_row(
                "Successes",
                str(success_a),
                str(success_b),
                f"{success_b - success_a:+d}"
            )
            
            table.add_row(
                "Failures",
                str(len(results_a) - success_a),
                str(len(results_b) - success_b),
                f"{(len(results_b) - success_b) - (len(results_a) - success_a):+d}"
            )
            
            console.print(table)
            console.print()
            
            # Statistical validation
            console.print_header("Statistical Validation")
            
            # Sample size warning if applicable
            if analysis.get("sample_size_warning"):
                console.print(format_warning(analysis["sample_size_warning"]))
                console.print()
            
            if analysis["significant"]:
                console.print(format_success(f"✓ Difference is statistically significant (p = {analysis['p_value']:.4f})"))
            else:
                console.print(format_warning(f"Difference is NOT statistically significant (p = {analysis['p_value']:.4f})"))
            
            console.print_metric("Test type", analysis.get("test_type", "chi-square"))
            console.print_metric("Effect size (Cohen's h)", f"{analysis['effect_size']:.2f} ({analysis['interpretation']})") 
            console.print_metric("Confidence interval", f"[{analysis['ci_lower']:.1%}, {analysis['ci_upper']:.1%}]")
            if analysis.get("power") is not None:
                console.print_metric("Statistical power", f"{analysis['power']:.2f}")
            console.print()
            
            # Interpretation
            if analysis["significant"] and rel_diff > 0:
                relative_improvement = (rel_diff/reliability_a*100) if reliability_a > 0 else float('inf')
                console.print(Panel.fit(
                    f"[success]Config B shows a {rel_diff*100:.1f} percentage point improvement[/success]\n"
                    f"This represents a {relative_improvement:.1f}% relative improvement" if reliability_a > 0 else "Config A had 0% reliability",
                    title="Conclusion",
                    border_style="success"
                ))
            elif analysis["significant"] and rel_diff < 0:
                relative_decrease = abs(rel_diff/reliability_a*100) if reliability_a > 0 else float('inf')
                console.print(Panel.fit(
                    f"[error]Config B shows a {abs(rel_diff)*100:.1f} percentage point regression[/error]\n"
                    f"This represents a {relative_decrease:.1f}% relative decrease" if reliability_a > 0 else "Config A had 0% reliability",
                    title="Conclusion",
                    border_style="error"
                ))
            else:
                console.print(Panel.fit(
                    "[warning]No significant difference between configurations[/warning]\n"
                    "Consider running with more scenarios for higher statistical power",
                    title="Conclusion",
                    border_style="warning"
                ))
            console.print()
            
            # Show historical comparisons if available
            if historical_comparisons and historical_comparisons.get("previous_diffs"):
                console.print_header("Historical Context")
                console.print(f"Found {len(historical_comparisons['previous_diffs'])} previous comparisons")
                
                hist_table = Table(show_header=True, header_style="bold")
                hist_table.add_column("Date", style="muted")
                hist_table.add_column("Config A Reliability", justify="right")
                hist_table.add_column("Config B Reliability", justify="right") 
                hist_table.add_column("Improvement", justify="right")
                hist_table.add_column("Significant", justify="center")
                
                for diff in historical_comparisons["previous_diffs"][:5]:
                    hist_table.add_row(
                        diff["date"],
                        f"{diff['reliability_a']:.1%}",
                        f"{diff['reliability_b']:.1%}",
                        f"{diff['improvement']:+.1%}",
                        "✓" if diff["significant"] else "✗"
                    )
                
                console.print(hist_table)
                console.print()


async def _get_historical_comparisons(db_client, config1_name: str, config2_name: str) -> dict:
    """Get historical A/B test comparisons from database."""
    query = text("""
    SELECT 
        cd.created_at,
        cd.config_a_reliability,
        cd.config_b_reliability,
        cd.improvement_percentage,
        cd.is_significant,
        cd.p_value,
        cd.effect_size,
        cd.scenario_count
    FROM config_diffs cd
    WHERE (cd.config_a_name = :config1 AND cd.config_b_name = :config2)
       OR (cd.config_a_name = :config2 AND cd.config_b_name = :config1)
    ORDER BY cd.created_at DESC
    LIMIT 10
    """)
    
    async with db_client.engine.begin() as conn:
        result = await conn.execute(query, {"config1": config1_name, "config2": config2_name})
    
    previous_diffs = []
    for row in result:
        previous_diffs.append({
            "date": row[0].strftime("%Y-%m-%d"),
            "reliability_a": float(row[1] or 0),
            "reliability_b": float(row[2] or 0),
            "improvement": float(row[3] or 0) / 100,
            "significant": row[4],
            "p_value": float(row[5] or 0),
            "effect_size": float(row[6] or 0),
            "scenarios": row[7]
        })
    
    return {
        "previous_diffs": previous_diffs,
        "comparison_count": len(previous_diffs)
    }


async def _save_diff_to_database(db_client, diff_result: dict, results_a: list, results_b: list):
    """Save A/B test comparison to database."""
    # This would save to config_diffs table
    # For now, we'll create the necessary data structure
    config_a_name = diff_result["config_a"]["path"].split("/")[-1]
    config_b_name = diff_result["config_b"]["path"].split("/")[-1]
    
    # Would execute INSERT to config_diffs table here
    pass


def _perform_statistical_analysis(results_a: list, results_b: list) -> Dict[str, Any]:
    """Perform statistical analysis on A/B test results with proper validation."""
    # Validate sample sizes
    n_a = len(results_a)
    n_b = len(results_b)
    
    min_sample_size = 30  # Minimum for reliable statistics
    sample_size_warning = None
    
    if n_a < min_sample_size or n_b < min_sample_size:
        sample_size_warning = f"Small sample size detected (A: {n_a}, B: {n_b}). Results may not be reliable. Consider running with at least {min_sample_size} scenarios."
    
    # Convert to binary success arrays
    successes_a = [1 if r["success"] else 0 for r in results_a]
    successes_b = [1 if r["success"] else 0 for r in results_b]
    
    # Calculate proportions
    p_a = sum(successes_a) / n_a if n_a > 0 else 0
    p_b = sum(successes_b) / n_b if n_b > 0 else 0
    
    # Use chi-square test for proportions (more appropriate than t-test for binary data)
    from scipy.stats import chi2_contingency
    contingency_table = [
        [sum(successes_a), n_a - sum(successes_a)],
        [sum(successes_b), n_b - sum(successes_b)]
    ]
    chi2, p_value, dof, expected = chi2_contingency(contingency_table)
    
    # Calculate effect size (Cohen's h for proportions)
    # h = 2 * (arcsin(sqrt(p1)) - arcsin(sqrt(p2)))
    import numpy as np
    h = 2 * (np.arcsin(np.sqrt(p_b)) - np.arcsin(np.sqrt(p_a)))
    
    # Calculate confidence interval for difference in proportions
    pooled_p = (sum(successes_a) + sum(successes_b)) / (n_a + n_b)
    se = np.sqrt(pooled_p * (1 - pooled_p) * (1/n_a + 1/n_b))
    diff = p_b - p_a
    z_critical = 1.96  # 95% confidence
    ci_lower = diff - z_critical * se
    ci_upper = diff + z_critical * se
    
    # Calculate statistical power
    if ttest_power is not None:
        try:
            power = ttest_power(abs(h), n_a, alpha=0.05, alternative='two-sided')
        except:
            power = None
    else:
        power = None
    
    return {
        "p_value": p_value,
        "significant": p_value < 0.05,
        "effect_size": h,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "interpretation": _interpret_effect_size(h),
        "sample_size_a": n_a,
        "sample_size_b": n_b,
        "power": power,
        "sample_size_warning": sample_size_warning,
        "test_type": "chi-square",
        "reliability_a": p_a,
        "reliability_b": p_b
    }


def _interpret_effect_size(h: float) -> str:
    """Interpret Cohen's h effect size for proportions."""
    abs_h = abs(h)
    if abs_h < 0.2:
        return "negligible"
    elif abs_h < 0.5:
        return "small"
    elif abs_h < 0.8:
        return "medium"
    else:
        return "large"