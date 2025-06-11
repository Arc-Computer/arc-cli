"""Arc diff command - compare two agent run results."""

import asyncio
from typing import Any, Dict, Optional, List, Tuple
from datetime import datetime
from pathlib import Path

import click
from rich.table import Table
from rich.panel import Panel
import numpy as np
from scipy.stats import chi2_contingency, fisher_exact
try:
    from statsmodels.stats.power import ttest_power
except ImportError:
    ttest_power = None

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


def _resolve_to_run_id(identifier: str) -> Tuple[Optional[str], Optional[str]]:
    """Resolve an identifier to a run ID.
    
    Args:
        identifier: Either a run ID or a config file path
        
    Returns:
        Tuple of (run_id, config_name)
    """
    # Check if it's already a run ID (format: run_YYYYMMDD_HHMMSS_XXXXXXXX)
    if identifier.startswith('run_'):
        return identifier, None
    
    # Otherwise, treat as config path and find most recent run
    config_path = Path(identifier)
    if not config_path.exists():
        # Maybe it's just a config name without path
        config_name = identifier
    else:
        config_name = config_path.name
    
    # Find most recent run for this config
    runs = state.list_runs(limit=100)  # Look through recent runs
    for run in runs:
        if run['config_path'].endswith(config_name) or run['config_path'] == identifier:
            return run['run_id'], config_name
    
    return None, config_name


def _find_common_scenarios(run1, run2) -> List[str]:
    """Find scenarios that appear in both runs.
    
    Returns list of scenario IDs that are in both runs.
    """
    scenarios1 = {s.get('scenario_id', s.get('id', f'scenario_{i}')) 
                  for i, s in enumerate(run1.scenarios)}
    scenarios2 = {s.get('scenario_id', s.get('id', f'scenario_{i}')) 
                  for i, s in enumerate(run2.scenarios)}
    
    return list(scenarios1.intersection(scenarios2))


def _filter_results_to_scenarios(results: List[Dict], scenario_ids: List[str]) -> List[Dict]:
    """Filter results to only include specified scenarios."""
    scenario_set = set(scenario_ids)
    return [r for r in results if r.get('scenario_id') in scenario_set]


@click.command()
@click.argument('identifier1')
@click.argument('identifier2')
@click.option('--common-only', is_flag=True, help='Compare only scenarios that appear in both runs')
@click.option('--json', 'json_output', is_flag=True, help='Output JSON instead of rich text')
def diff(identifier1: str, identifier2: str, common_only: bool, json_output: bool):
    """Compare two agent run results.
    
    This command compares the results of two previous runs to determine
    which configuration performs better. You can specify either run IDs
    or configuration file paths.
    
    Examples:
        arc diff run_20241105_143022 run_20241105_145510
        arc diff finance_agent_v1.yaml finance_agent_v2.yaml
        arc diff old_config.yaml new_config.yaml --common-only
    """
    try:
        asyncio.run(_diff_async(identifier1, identifier2, common_only, json_output))
    except Exception as e:
        if json_output:
            import json
            print(json.dumps({"error": str(e)}, indent=2))
        else:
            console.print(format_error(f"Comparison failed: {str(e)}"))
        raise click.exceptions.Exit(1)


async def _diff_async(identifier1: str, identifier2: str, common_only: bool, json_output: bool):
    """Perform async diff analysis."""
    # Initialize state
    db_connected = await _initialize_state()
    
    if not json_output:
        console.print()
        console.print(Panel.fit(
            "[primary]Arc Run Comparison[/primary]",
            border_style="primary"
        ))
        console.print()
    
    # Resolve identifiers to run IDs
    run_id1, config_name1 = _resolve_to_run_id(identifier1)
    run_id2, config_name2 = _resolve_to_run_id(identifier2)
    
    if not run_id1:
        raise ValueError(f"No run found for '{identifier1}'")
    if not run_id2:
        raise ValueError(f"No run found for '{identifier2}'")
    
    if not json_output:
        console.print(f"Comparing runs:")
        console.print(f"  Run 1: {run_id1}" + (f" ({config_name1})" if config_name1 else ""))
        console.print(f"  Run 2: {run_id2}" + (f" ({config_name2})" if config_name2 else ""))
        console.print()
    
    # Load run results
    run1 = state.get_run(run_id1)
    run2 = state.get_run(run_id2)
    
    if not run1:
        raise ValueError(f"Run {run_id1} not found")
    if not run2:
        raise ValueError(f"Run {run_id2} not found")
    
    # Check scenario compatibility
    common_scenarios = _find_common_scenarios(run1, run2)
    
    if not json_output:
        console.print(f"Run 1 scenarios: {len(run1.scenarios)}")
        console.print(f"Run 2 scenarios: {len(run2.scenarios)}")
        console.print(f"Common scenarios: {len(common_scenarios)}")
        console.print()
    
    # Filter to common scenarios if requested or if sets are very different
    if common_only or len(common_scenarios) < min(len(run1.scenarios), len(run2.scenarios)) * 0.8:
        if not common_scenarios:
            raise ValueError("No common scenarios found between runs")
        
        if not json_output and not common_only:
            console.print(format_warning(
                f"Only {len(common_scenarios)} scenarios in common. Comparing only common scenarios."
            ))
            console.print()
        
        # Filter results to common scenarios
        results1 = _filter_results_to_scenarios(run1.results, common_scenarios)
        results2 = _filter_results_to_scenarios(run2.results, common_scenarios)
    else:
        results1 = run1.results
        results2 = run2.results
    
    # Perform statistical analysis
    analysis = _perform_statistical_analysis(results1, results2)
    
    # Create comparison summary
    comparison = {
        "run1": {
            "run_id": run_id1,
            "config_path": run1.config_path,
            "timestamp": run1.timestamp.isoformat() if hasattr(run1.timestamp, 'isoformat') else str(run1.timestamp),
            "total_scenarios": len(run1.scenarios),
            "compared_scenarios": len(results1),
            "reliability": run1.reliability_score,
            "successes": sum(1 for r in results1 if r.get('success', False)),
            "failures": sum(1 for r in results1 if not r.get('success', False)),
        },
        "run2": {
            "run_id": run_id2,
            "config_path": run2.config_path,
            "timestamp": run2.timestamp.isoformat() if hasattr(run2.timestamp, 'isoformat') else str(run2.timestamp),
            "total_scenarios": len(run2.scenarios),
            "compared_scenarios": len(results2),
            "reliability": run2.reliability_score,
            "successes": sum(1 for r in results2 if r.get('success', False)),
            "failures": sum(1 for r in results2 if not r.get('success', False)),
        },
        "comparison": {
            "scenarios_compared": len(results1),
            "improvement": analysis['reliability_b'] - analysis['reliability_a'],
            "improvement_percentage": (analysis['reliability_b'] - analysis['reliability_a']) * 100,
            "relative_improvement": ((analysis['reliability_b'] - analysis['reliability_a']) / analysis['reliability_a'] * 100) 
                                   if analysis['reliability_a'] > 0 else 0,
        },
        "statistical_analysis": analysis
    }
    
    # Find scenario-level differences
    scenario_differences = _analyze_scenario_differences(results1, results2)
    comparison['scenario_differences'] = scenario_differences
    
    # Display results
    if json_output:
        import json
        print(json.dumps(comparison, indent=2))
    else:
        _display_comparison(comparison, analysis, scenario_differences)


def _analyze_scenario_differences(results1: List[Dict], results2: List[Dict]) -> Dict:
    """Analyze differences at the scenario level."""
    # Create result maps by scenario ID
    results1_map = {r['scenario_id']: r for r in results1}
    results2_map = {r['scenario_id']: r for r in results2}
    
    improved = []
    regressed = []
    unchanged = []
    
    for scenario_id in results1_map:
        if scenario_id in results2_map:
            success1 = results1_map[scenario_id].get('success', False)
            success2 = results2_map[scenario_id].get('success', False)
            
            if not success1 and success2:
                improved.append(scenario_id)
            elif success1 and not success2:
                regressed.append(scenario_id)
            else:
                unchanged.append(scenario_id)
    
    return {
        "improved": improved,
        "regressed": regressed,
        "unchanged": unchanged,
        "improvement_count": len(improved),
        "regression_count": len(regressed),
        "unchanged_count": len(unchanged)
    }


def _display_comparison(comparison: Dict, analysis: Dict, scenario_differences: Dict):
    """Display comparison results in rich format."""
    # Results comparison table
    console.print_header("Results Comparison")
    
    table = Table(show_header=True, header_style="bold")
    table.add_column("Metric", style="muted")
    table.add_column("Run 1", justify="right")
    table.add_column("Run 2", justify="right")
    table.add_column("Difference", justify="right")
    
    # Reliability
    rel1 = comparison['run1']['reliability']
    rel2 = comparison['run2']['reliability']
    rel_diff = rel2 - rel1
    
    table.add_row(
        "Reliability",
        f"{rel1:.1%}",
        f"{rel2:.1%}",
        f"{rel_diff:+.1%}"
    )
    
    # Success/Failure counts
    table.add_row(
        "Successes",
        str(comparison['run1']['successes']),
        str(comparison['run2']['successes']),
        f"{comparison['run2']['successes'] - comparison['run1']['successes']:+d}"
    )
    
    table.add_row(
        "Failures",
        str(comparison['run1']['failures']),
        str(comparison['run2']['failures']),
        f"{comparison['run2']['failures'] - comparison['run1']['failures']:+d}"
    )
    
    table.add_row(
        "Scenarios compared",
        str(comparison['comparison']['scenarios_compared']),
        str(comparison['comparison']['scenarios_compared']),
        "-"
    )
    
    console.print(table)
    console.print()
    
    # Scenario-level changes
    if scenario_differences['improvement_count'] > 0 or scenario_differences['regression_count'] > 0:
        console.print_header("Scenario-Level Changes")
        
        if scenario_differences['improvement_count'] > 0:
            console.print(f"[success]✓ {scenario_differences['improvement_count']} scenarios improved[/success]")
        
        if scenario_differences['regression_count'] > 0:
            console.print(f"[error]✗ {scenario_differences['regression_count']} scenarios regressed[/error]")
        
        console.print(f"[muted]• {scenario_differences['unchanged_count']} scenarios unchanged[/muted]")
        console.print()
    
    # Statistical validation
    console.print_header("Statistical Validation")
    
    # Display statistical warnings
    statistical_warnings = analysis.get("statistical_warnings", [])
    if statistical_warnings:
        for warning in statistical_warnings:
            console.print(format_warning(warning))
        console.print()
    
    # Statistical significance
    if analysis.get("valid_statistical_test", False):
        if analysis["significant"]:
            console.print(format_success(f"✓ Difference is statistically significant (p = {analysis['p_value']:.4f})"))
        else:
            console.print(format_warning(f"Difference is NOT statistically significant (p = {analysis['p_value']:.4f})"))
    else:
        console.print(format_warning("Statistical significance test could not be performed"))
    
    console.print_metric("Test type", analysis.get("test_type", "unknown"))
    console.print_metric("Effect size (Cohen's h)", f"{analysis['effect_size']:.2f} ({analysis['interpretation']})")
    console.print_metric("Confidence interval", f"[{analysis['ci_lower']:.1%}, {analysis['ci_upper']:.1%}]")
    
    if analysis.get("power") is not None:
        console.print_metric("Statistical power", f"{analysis['power']:.2f}")
    console.print()
    
    # Interpretation
    if analysis.get("significant") and rel_diff > 0:
        relative_improvement = (rel_diff/rel1*100) if rel1 > 0 else float('inf')
        console.print(Panel.fit(
            f"[success]Run 2 shows a {rel_diff*100:.1f} percentage point improvement[/success]\n"
            f"This represents a {relative_improvement:.1f}% relative improvement" if rel1 > 0 else "Run 1 had 0% reliability",
            title="Conclusion",
            border_style="success"
        ))
    elif analysis.get("significant") and rel_diff < 0:
        relative_decrease = abs(rel_diff/rel1*100) if rel1 > 0 else float('inf')
        console.print(Panel.fit(
            f"[error]Run 2 shows a {abs(rel_diff)*100:.1f} percentage point regression[/error]\n"
            f"This represents a {relative_decrease:.1f}% relative decrease" if rel1 > 0 else "Run 1 had 0% reliability",
            title="Conclusion",
            border_style="error"
        ))
    else:
        console.print(Panel.fit(
            "[warning]No significant difference between runs[/warning]\n"
            "Consider running with more scenarios for higher statistical power",
            title="Conclusion",
            border_style="warning"
        ))
    console.print()
    
    # Run information
    console.print_header("Run Information")
    console.print(f"Run 1: {comparison['run1']['config_path']} at {comparison['run1']['timestamp']}")
    console.print(f"Run 2: {comparison['run2']['config_path']} at {comparison['run2']['timestamp']}")


def _perform_statistical_analysis(results_a: list, results_b: list) -> Dict[str, Any]:
    """Perform statistical analysis on run results with proper validation."""
    # Validate sample sizes
    n_a = len(results_a)
    n_b = len(results_b)
    
    min_sample_size = 30  # Minimum for reliable statistics
    sample_size_warning = None
    
    if n_a < min_sample_size or n_b < min_sample_size:
        sample_size_warning = f"Small sample size detected (Run 1: {n_a}, Run 2: {n_b}). Results may not be reliable. Consider running with at least {min_sample_size} scenarios."
    
    # Convert to binary success arrays
    successes_a = [1 if r.get("success", False) else 0 for r in results_a]
    successes_b = [1 if r.get("success", False) else 0 for r in results_b]
    
    # Calculate proportions
    p_a = sum(successes_a) / n_a if n_a > 0 else 0
    p_b = sum(successes_b) / n_b if n_b > 0 else 0
    
    # Use chi-square test for proportions
    contingency_table = [
        [sum(successes_a), n_a - sum(successes_a)],
        [sum(successes_b), n_b - sum(successes_b)]
    ]
    
    # Validate chi-square test assumptions
    try:
        chi2, p_value_chi2, dof, expected = chi2_contingency(contingency_table)
        
        # Check if all expected frequencies are >= 5
        min_expected = np.min(expected)
        if min_expected < 5:
            # Use Fisher's exact test for small samples
            if len(contingency_table) == 2 and len(contingency_table[0]) == 2:
                oddsratio, p_value = fisher_exact(contingency_table)
                test_type = "fisher_exact"
                chi2 = None
            else:
                p_value = None
                test_type = "descriptive_only"
                chi2 = None
        else:
            p_value = p_value_chi2
            test_type = "chi_square"
            
    except (ValueError, ZeroDivisionError):
        p_value = None
        test_type = "insufficient_data"
        chi2 = None
    
    # Calculate effect size (Cohen's h for proportions)
    h = 2 * (np.arcsin(np.sqrt(p_b)) - np.arcsin(np.sqrt(p_a)))
    
    # Calculate confidence interval for difference in proportions
    pooled_p = (sum(successes_a) + sum(successes_b)) / (n_a + n_b) if (n_a + n_b) > 0 else 0
    se = np.sqrt(pooled_p * (1 - pooled_p) * (1/n_a + 1/n_b)) if pooled_p > 0 and pooled_p < 1 else 0
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
    
    # Determine significance
    significant = p_value < 0.05 if p_value is not None else None
    
    # Add warnings
    statistical_warnings = []
    if sample_size_warning:
        statistical_warnings.append(sample_size_warning)
    
    if test_type == "fisher_exact":
        statistical_warnings.append("Used Fisher's exact test due to small expected frequencies.")
    elif test_type == "descriptive_only":
        statistical_warnings.append("Statistical test not performed due to insufficient data.")
    elif test_type == "insufficient_data":
        statistical_warnings.append("Insufficient data for statistical analysis.")
    
    return {
        "p_value": p_value,
        "significant": significant,
        "effect_size": h,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "interpretation": _interpret_effect_size(h),
        "sample_size_a": n_a,
        "sample_size_b": n_b,
        "power": power,
        "sample_size_warning": sample_size_warning,
        "statistical_warnings": statistical_warnings,
        "test_type": test_type,
        "reliability_a": p_a,
        "reliability_b": p_b,
        "valid_statistical_test": p_value is not None
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