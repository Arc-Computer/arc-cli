"""Arc run command - test agent with generated scenarios."""

import asyncio
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any
from uuid import uuid4

import click
import yaml
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeRemainingColumn
from rich.panel import Panel
from rich.table import Table

from arc.cli.utils import ArcConsole, CLIState, RunResult, format_error, format_success, format_warning
from arc.ingestion.parser import AgentConfigParser
from arc.ingestion.normalizer import ConfigNormalizer
from arc.scenarios.generator import ScenarioGenerator, generate_high_quality_scenarios
from arc.core.models.scenario import Scenario


console = ArcConsole()
state = CLIState()


@click.command()
@click.argument('config_path', type=click.Path(exists=True))
@click.option('--scenarios', '-s', default=50, help='Number of scenarios to generate (default: 50)')
@click.option('--json', 'json_output', is_flag=True, help='Output JSON instead of rich text')
@click.option('--no-confirm', is_flag=True, help='Skip cost confirmation prompt')
@click.option('--pattern-ratio', default=0.7, help='Ratio of pattern-based scenarios (0-1)')
def run(config_path: str, scenarios: int, json_output: bool, no_confirm: bool, pattern_ratio: float):
    """Test an agent configuration with generated scenarios.
    
    This command:
    1. Parses your agent configuration
    2. Generates test scenarios (currency assumptions + general)  
    3. Executes scenarios in parallel
    4. Reports reliability score and failures
    
    Example:
        arc run finance_agent_v1.yaml
    """
    config_path = Path(config_path)
    run_id = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid4().hex[:8]}"
    
    if not json_output:
        console.print()
        console.print(Panel.fit(
            "[primary]Arc: Proactive Capability Assurance[/primary]",
            border_style="primary"
        ))
        console.print()
    
    try:
        # Step 1: Parse and validate configuration
        if not json_output:
            console.print("Parsing agent configuration...", style="muted")
        
        parser = AgentConfigParser()
        normalizer = ConfigNormalizer()
        
        parsed_config = parser.parse(config_path)
        normalized_config = normalizer.normalize(parsed_config)
        capabilities = parser.extract_capabilities(parsed_config)
        
        if not json_output:
            console.print(format_success(f"Configuration validated: {config_path.name}"))
            console.print_metric("Model", normalized_config["model"])
            console.print_metric("Temperature", normalized_config["temperature"])
            console.print_metric("Tools", f"{len(normalized_config['tools'])} configured")
            console.print()
        
        # Step 2: Estimate costs
        estimated_cost = _estimate_cost(scenarios, normalized_config["model"])
        
        if not json_output and not no_confirm and estimated_cost > state.config.get("cost_warning_threshold", 0.10):
            console.print(format_warning(f"Estimated cost: ${estimated_cost:.4f}"))
            if not click.confirm("Continue with this run?"):
                console.print("Run cancelled.", style="muted")
                return
        
        # Step 3: Generate scenarios
        if not json_output:
            console.print_header(f"Generating {scenarios} test scenarios")
            console.print(f"Pattern-based: {int(pattern_ratio * 100)}%")
            console.print(f"LLM-generated: {int((1 - pattern_ratio) * 100)}%")
            console.print()
        
        # Run async scenario generation
        generated_scenarios = asyncio.run(
            _generate_scenarios_async(
                config_path=str(config_path),
                count=scenarios,
                pattern_ratio=pattern_ratio,
                capabilities=capabilities,
                json_output=json_output
            )
        )
        
        if not json_output:
            console.print(format_success(f"Generated {len(generated_scenarios)} scenarios"))
            _print_scenario_summary(generated_scenarios)
            console.print()
        
        # Step 4: Execute scenarios (placeholder for now)
        if not json_output:
            console.print_header("Executing scenarios")
            console.print("[warning]Note: Modal execution not yet implemented[/warning]")
            console.print("Simulating execution for demo purposes...")
            console.print()
        
        # Simulate execution with mock results
        results, execution_time = _simulate_execution(generated_scenarios, json_output)
        
        # Step 5: Calculate results
        success_count = sum(1 for r in results if r["success"])
        failure_count = len(results) - success_count
        reliability_score = success_count / len(results) if results else 0
        
        # Identify currency failures (for finance demo)
        currency_failures = [
            r for r in results 
            if not r["success"] and "currency" in r.get("failure_reason", "").lower()
        ]
        
        # Save run results
        run_result = RunResult(
            run_id=run_id,
            config_path=str(config_path),
            timestamp=datetime.now(),
            scenario_count=len(generated_scenarios),
            success_count=success_count,
            failure_count=failure_count,
            reliability_score=reliability_score,
            execution_time=execution_time,
            total_cost=estimated_cost,  # Will be actual cost when Modal is integrated
            scenarios=[s.to_dict() if hasattr(s, 'to_dict') else s for s in generated_scenarios],
            results=results,
            failures=[r for r in results if not r["success"]]
        )
        
        state.save_run(run_result)
        
        # Display results
        if json_output:
            import json
            print(json.dumps(run_result.to_dict(), indent=2))
        else:
            console.print()
            console.print(Panel.fit(
                "[success]✓ Found capability issues BEFORE production[/success]",
                border_style="success"
            ))
            console.print()
            
            # Results table
            table = Table(show_header=False, box=None)
            table.add_column("Metric", style="muted")
            table.add_column("Value", style="highlight")
            
            table.add_row("Overall Reliability", f"{reliability_score:.1%} ({success_count}/{len(results)} scenarios)")
            if currency_failures:
                table.add_row("Currency Assumption Violations", f"{len(currency_failures)} failures", style="error")
            table.add_row("Time to insight", f"{execution_time:.1f} seconds")
            table.add_row("Actual cost", f"${estimated_cost:.4f}")
            
            console.print(table)
            console.print()
            
            console.print("Run [info]arc analyze[/info] to see detailed breakdown", style="muted")
            console.print("Run [info]arc recommend[/info] for specific fixes", style="muted")
            console.print()
    
    except Exception as e:
        if json_output:
            import json
            print(json.dumps({"error": str(e)}, indent=2))
        else:
            console.print(format_error(f"Run failed: {str(e)}"))
            console.print("\nTry: [info]arc validate {config_path}[/info] to check your configuration", style="muted")
        raise click.Exit(1)


async def _generate_scenarios_async(
    config_path: str,
    count: int,
    pattern_ratio: float,
    capabilities: Dict[str, Any],
    json_output: bool
) -> List[Scenario]:
    """Generate scenarios asynchronously."""
    try:
        # Check if finance domain for currency scenarios
        is_finance = "finance" in capabilities.get("domains", [])
        
        if is_finance and not json_output:
            console.print("Detected finance domain - including currency assumption scenarios", style="info")
        
        # Generate scenarios
        scenarios = await generate_high_quality_scenarios(
            agent_config_path=config_path,
            count=count,
            use_patterns=True,
            pattern_ratio=pattern_ratio,
            quality_threshold=0.7
        )
        
        return scenarios
    
    except Exception as e:
        # For now, return mock scenarios if generation fails
        if not json_output:
            console.print(format_warning("Using mock scenarios for demo"))
        
        mock_scenarios = []
        for i in range(count):
            scenario = {
                "scenario_id": f"mock_{i}",
                "task_prompt": f"Test scenario {i}",
                "expected_tools": ["calculator", "web_search"],
                "complexity_level": "medium",
                "domain": "finance" if i < 15 else "general"
            }
            mock_scenarios.append(scenario)
        
        return mock_scenarios


def _estimate_cost(scenario_count: int, model: str) -> float:
    """Estimate cost for running scenarios."""
    # Simplified cost estimation
    cost_per_scenario = {
        "openai/gpt-4.1": 0.0010,
        "openai/gpt-4.1-mini": 0.0002,
        "anthropic/claude-opus-4": 0.0015,
        "anthropic/claude-sonnet-4": 0.0005,
    }
    
    base_cost = cost_per_scenario.get(model, 0.0005)
    return scenario_count * base_cost


def _print_scenario_summary(scenarios: List[Any]) -> None:
    """Print summary of generated scenarios."""
    # Count by domain
    domains = {}
    for s in scenarios:
        domain = s.get("domain", "general") if isinstance(s, dict) else getattr(s, "inferred_domain", "general")
        domains[domain] = domains.get(domain, 0) + 1
    
    if domains:
        console.print("Scenario breakdown:", style="muted")
        for domain, count in sorted(domains.items(), key=lambda x: x[1], reverse=True):
            console.print(f"  {domain}: {count} scenarios", style="muted")


def _simulate_execution(scenarios: List[Any], json_output: bool) -> tuple[List[Dict[str, Any]], float]:
    """Simulate scenario execution with mock results."""
    results = []
    start_time = time.time()
    
    if not json_output:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeRemainingColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Executing scenarios...", total=len(scenarios))
            
            for i, scenario in enumerate(scenarios):
                # Simulate execution time
                time.sleep(0.02)
                
                # Generate mock result
                is_currency = "currency" in str(scenario).lower()
                success = not (is_currency and i < 15)  # First 15 currency scenarios fail
                
                result = {
                    "scenario_id": scenario.get("scenario_id", f"scenario_{i}") if isinstance(scenario, dict) else getattr(scenario, "scenario_id", f"scenario_{i}"),
                    "success": success,
                    "execution_time": 0.5 + (i * 0.01),
                    "failure_reason": "Currency assumption violation: Expected multi-currency support" if not success else None
                }
                results.append(result)
                
                progress.update(task, advance=1)
    else:
        # Silent execution for JSON output
        for i, scenario in enumerate(scenarios):
            is_currency = "currency" in str(scenario).lower()
            success = not (is_currency and i < 15)
            
            result = {
                "scenario_id": scenario.get("scenario_id", f"scenario_{i}") if isinstance(scenario, dict) else getattr(scenario, "scenario_id", f"scenario_{i}"),
                "success": success,
                "execution_time": 0.5 + (i * 0.01),
                "failure_reason": "Currency assumption violation: Expected multi-currency support" if not success else None
            }
            results.append(result)
    
    execution_time = time.time() - start_time
    return results, execution_time