"""Arc diff command - compare two agent configurations."""

import asyncio
from typing import Dict, Any

import click
from rich.table import Table
from rich.panel import Panel
from scipy import stats

from arc.cli.utils import ArcConsole, CLIState, format_error, format_success, format_warning
from arc.cli.commands.run import _generate_scenarios_async, _estimate_cost, _simulate_execution
from arc.ingestion.parser import AgentConfigParser
from arc.ingestion.normalizer import ConfigNormalizer

console = ArcConsole()
state = CLIState()


@click.command()
@click.argument('config1', type=click.Path(exists=True))
@click.argument('config2', type=click.Path(exists=True))
@click.option('--scenarios', '-s', default=50, help='Number of scenarios per config (default: 50)')
@click.option('--json', 'json_output', is_flag=True, help='Output JSON instead of rich text')
def diff(config1: str, config2: str, scenarios: int, json_output: bool):
    """Compare two agent configurations with A/B testing.
    
    This command:
    1. Runs both configurations on the same scenarios
    2. Performs statistical significance testing
    3. Shows improvement metrics
    4. Validates claimed improvements
    
    Example:
        arc diff finance_agent_v1.yaml finance_agent_v2.yaml
    """
    try:
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
            console.print()
        
        # Generate scenarios (same for both configs)
        if not json_output:
            console.print("Generating test scenarios...", style="muted")
        
        test_scenarios = asyncio.run(
            _generate_scenarios_async(
                config_path=config1,
                count=scenarios,
                pattern_ratio=0.7,
                capabilities=parser.extract_capabilities(config1_parsed),
                json_output=json_output
            )
        )
        
        # Run both configurations
        if not json_output:
            console.print()
            console.print("[primary]Running Config A[/primary]")
        
        results_a, time_a = _simulate_execution(test_scenarios, json_output)
        success_a = sum(1 for r in results_a if r["success"])
        reliability_a = success_a / len(results_a)
        
        if not json_output:
            console.print(format_success(f"Config A: {reliability_a:.1%} reliability"))
            console.print()
            console.print("[primary]Running Config B[/primary]")
        
        results_b, time_b = _simulate_execution(test_scenarios, json_output)
        success_b = sum(1 for r in results_b if r["success"])
        reliability_b = success_b / len(results_b)
        
        if not json_output:
            console.print(format_success(f"Config B: {reliability_b:.1%} reliability"))
            console.print()
        
        # Perform statistical analysis
        analysis = _perform_statistical_analysis(results_a, results_b)
        
        # Display results
        if json_output:
            import json
            output = {
                "config_a": {
                    "path": config1,
                    "reliability": reliability_a,
                    "successes": success_a,
                    "failures": len(results_a) - success_a
                },
                "config_b": {
                    "path": config2,
                    "reliability": reliability_b,
                    "successes": success_b,
                    "failures": len(results_b) - success_b
                },
                "comparison": {
                    "improvement": reliability_b - reliability_a,
                    "improvement_percentage": (reliability_b - reliability_a) * 100,
                    "relative_improvement": ((reliability_b - reliability_a) / reliability_a * 100) if reliability_a > 0 else 0
                },
                "statistical_analysis": analysis
            }
            print(json.dumps(output, indent=2))
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
            
            if analysis["significant"]:
                console.print(format_success(f"✓ Improvement is statistically significant (p = {analysis['p_value']:.4f})"))
            else:
                console.print(format_warning(f"! Improvement is NOT statistically significant (p = {analysis['p_value']:.4f})"))
            
            console.print_metric("Effect size (Cohen's d)", f"{analysis['effect_size']:.2f}")
            console.print_metric("Confidence interval", f"[{analysis['ci_lower']:.1%}, {analysis['ci_upper']:.1%}]")
            console.print()
            
            # Interpretation
            if analysis["significant"] and rel_diff > 0:
                console.print(Panel.fit(
                    f"[success]Config B shows a {rel_diff*100:.1f} percentage point improvement[/success]\n"
                    f"This represents a {(rel_diff/reliability_a*100):.1f}% relative improvement",
                    title="Conclusion",
                    border_style="success"
                ))
            elif analysis["significant"] and rel_diff < 0:
                console.print(Panel.fit(
                    f"[error]Config B shows a {abs(rel_diff)*100:.1f} percentage point regression[/error]\n"
                    f"This represents a {abs(rel_diff/reliability_a*100):.1f}% relative decrease",
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
    
    except Exception as e:
        if json_output:
            import json
            print(json.dumps({"error": str(e)}, indent=2))
        else:
            console.print(format_error(f"Comparison failed: {str(e)}"))
        raise click.Exit(1)


def _perform_statistical_analysis(results_a: list, results_b: list) -> Dict[str, Any]:
    """Perform statistical analysis on A/B test results."""
    # Convert to binary success arrays
    successes_a = [1 if r["success"] else 0 for r in results_a]
    successes_b = [1 if r["success"] else 0 for r in results_b]
    
    # Perform t-test
    t_stat, p_value = stats.ttest_ind(successes_b, successes_a)
    
    # Calculate effect size (Cohen's d)
    mean_a = sum(successes_a) / len(successes_a)
    mean_b = sum(successes_b) / len(successes_b)
    std_a = stats.tstd(successes_a)
    std_b = stats.tstd(successes_b)
    pooled_std = ((std_a ** 2 + std_b ** 2) / 2) ** 0.5
    
    effect_size = (mean_b - mean_a) / pooled_std if pooled_std > 0 else 0
    
    # Calculate confidence interval
    diff = mean_b - mean_a
    se = pooled_std * (2 / len(successes_a)) ** 0.5
    ci_lower = diff - 1.96 * se
    ci_upper = diff + 1.96 * se
    
    return {
        "p_value": p_value,
        "significant": p_value < 0.05,
        "effect_size": effect_size,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "interpretation": _interpret_effect_size(effect_size)
    }


def _interpret_effect_size(d: float) -> str:
    """Interpret Cohen's d effect size."""
    abs_d = abs(d)
    if abs_d < 0.2:
        return "negligible"
    elif abs_d < 0.5:
        return "small"
    elif abs_d < 0.8:
        return "medium"
    else:
        return "large"