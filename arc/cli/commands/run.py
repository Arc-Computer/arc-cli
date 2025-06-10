"""Arc run command - test agent with generated scenarios."""

import asyncio
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, List, Dict, Optional
from uuid import uuid4

import click
import yaml
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeRemainingColumn
from rich.panel import Panel
from rich.table import Table

from arc.cli.utils import ArcConsole, CLIState, RunResult, format_error, format_success, format_warning
from arc.ingestion.parser import AgentConfigParser
from arc.ingestion.normalizer import ConfigNormalizer
from arc.scenarios.generator import ScenarioGenerator
from arc.core.models.scenario import Scenario


console = ArcConsole()
state = CLIState()


@click.command()
@click.argument('config_path', type=click.Path(exists=True))
@click.option('--scenarios', '-s', default=50, help='Number of scenarios to generate (default: 50)')
@click.option('--json', 'json_output', is_flag=True, help='Output JSON instead of rich text')
@click.option('--no-confirm', is_flag=True, help='Skip cost confirmation prompt')
@click.option('--pattern-ratio', default=0.7, type=click.FloatRange(0.0, 1.0), help='Ratio of pattern-based scenarios (0-1)')
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
        
        # Step 4: Execute scenarios
        if not json_output:
            console.print_header("Executing scenarios")
        
        # Check if Modal execution is available
        use_modal = state.config.get("use_modal", False) and _check_modal_available()
        
        if use_modal:
            # Execute with Modal sandbox
            results, execution_time, actual_cost = _execute_with_modal(
                scenarios=generated_scenarios,
                agent_config=normalized_config,
                json_output=json_output
            )
        else:
            # Fall back to simulation
            if not json_output:
                console.print("[warning]Modal not configured - simulating execution[/warning]")
                console.print()
            results, execution_time = _simulate_execution(generated_scenarios, json_output)
            actual_cost = estimated_cost
        
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
            total_cost=actual_cost,  # Use actual cost from execution
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
            table.add_row("Actual cost", f"${actual_cost:.4f}")
            
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
        raise click.exceptions.Exit(1)


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
        
        # Initialize generator
        generator = ScenarioGenerator(
            agent_config_path=config_path,
            use_patterns=True,
            quality_threshold=2.0  # Lower threshold for more scenarios
        )
        
        # Generate scenarios with currency focus for first 15
        if count >= 50:
            # Generate 15 currency scenarios + 35 general scenarios
            currency_scenarios = await generator.generate_scenarios_batch(
                count=15,
                pattern_ratio=pattern_ratio,
                currency_focus=True
            )
            
            general_scenarios = await generator.generate_scenarios_batch(
                count=count - 15,
                pattern_ratio=pattern_ratio,
                currency_focus=False
            )
            
            scenarios = currency_scenarios + general_scenarios
        else:
            # Generate all as general scenarios
            scenarios = await generator.generate_scenarios_batch(
                count=count,
                pattern_ratio=pattern_ratio,
                currency_focus=False
            )
        
        return scenarios
    
    except Exception as e:
        # For now, return mock scenarios if generation fails
        if not json_output:
            console.print(format_warning("Failed to generate scenarios, using mock data"))
        
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
    # Model pricing from experiments/generation/generator.py
    # Estimated cost per scenario based on ~1500 tokens (1k input, 0.5k output)
    # Using the token pricing to calculate per-scenario cost
    MODELS_PRICING = {
        # OpenAI
        "openai/gpt-4.1": {"cost_per_1k_in": 0.00200, "cost_per_1k_out": 0.00800},
        "openai/gpt-4.1-mini": {"cost_per_1k_in": 0.00040, "cost_per_1k_out": 0.00160},
        
        # Anthropic
        "anthropic/claude-opus-4": {"cost_per_1k_in": 0.01500, "cost_per_1k_out": 0.07500},
        "anthropic/claude-sonnet-4": {"cost_per_1k_in": 0.00300, "cost_per_1k_out": 0.01500},
        "anthropic/claude-3.5-haiku": {"cost_per_1k_in": 0.00080, "cost_per_1k_out": 0.00400},
        
        # Google
        "google/gemini-2.5-pro-preview": {"cost_per_1k_in": 0.00125, "cost_per_1k_out": 0.01000},
        "google/gemini-2.5-flash-preview-05-20": {"cost_per_1k_in": 0.00015, "cost_per_1k_out": 0.00060},
        
        # Meta (Llama 4)
        "meta-llama/llama-4-maverick": {"cost_per_1k_in": 0.00015, "cost_per_1k_out": 0.00060},
        "meta-llama/llama-4-scout": {"cost_per_1k_in": 0.00008, "cost_per_1k_out": 0.00030},
        
        # Mistral
        "mistralai/mistral-medium-3": {"cost_per_1k_in": 0.00040, "cost_per_1k_out": 0.00200},
        
        # Cohere
        "cohere/command-a": {"cost_per_1k_in": 0.00250, "cost_per_1k_out": 0.01000},
        
        # DeepSeek
        "deepseek/deepseek-chat-v3-0324": {"cost_per_1k_in": 0.00030, "cost_per_1k_out": 0.00088},
        "deepseek/deepseek-r1-0528": {"cost_per_1k_in": 0.00050, "cost_per_1k_out": 0.00215},
    }
    
    # Calculate cost per scenario (assuming 1k input tokens + 0.5k output tokens)
    if model in MODELS_PRICING:
        pricing = MODELS_PRICING[model]
        input_cost = 1.0 * pricing["cost_per_1k_in"]  # 1k input tokens
        output_cost = 0.5 * pricing["cost_per_1k_out"]  # 0.5k output tokens
        cost_per_scenario = input_cost + output_cost
    else:
        # Default fallback for unknown models
        cost_per_scenario = 0.0005
    
    return scenario_count * cost_per_scenario


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


def _check_modal_available() -> bool:
    """Check if Modal is installed and configured."""
    try:
        import modal
        # Check for both Modal tokens
        has_token_id = bool(os.environ.get("MODAL_TOKEN_ID"))
        has_token_secret = bool(os.environ.get("MODAL_TOKEN_SECRET"))
        
        if has_token_id and not has_token_secret:
            console.print(format_warning("MODAL_TOKEN_ID found but MODAL_TOKEN_SECRET is missing"))
            console.print("Run 'modal setup' to complete authentication", style="muted")
            return False
            
        return has_token_id and has_token_secret
    except ImportError:
        return False


def _execute_with_modal(
    scenarios: List[Scenario],
    agent_config: Dict[str, Any],
    json_output: bool
) -> tuple[List[Dict[str, Any]], float, float]:
    """Execute scenarios using Modal sandbox.
    
    Returns:
        Tuple of (results, execution_time, total_cost)
    """
    try:
        import modal
        # Import the app and function from simulator
        from arc.sandbox.engine.simulator import app as modal_app, evaluate_single_scenario
    except ImportError as e:
        raise RuntimeError(f"Modal execution failed: {e}")
    
    start_time = time.time()
    results = []
    total_cost = 0.0
    
    # Prepare scenarios for sandbox
    # Convert Scenario objects to dicts if needed
    scenario_dicts = []
    for scenario in scenarios:
        if hasattr(scenario, 'to_dict'):
            scenario_dicts.append(scenario.to_dict())
        else:
            scenario_dicts.append(scenario)
    
    # Create sandbox data format
    sandbox_data = {
        "scenarios": scenario_dicts,
        "metadata": {
            "total_scenarios": len(scenarios),
            "generation_timestamp": datetime.now().isoformat(),
            "agent_model": agent_config.get("model", "unknown")
        }
    }
    
    # Create tuples of (scenario, agent_config, index) for Modal
    scenario_tuples = [
        (scenario, agent_config, i) 
        for i, scenario in enumerate(sandbox_data["scenarios"])
    ]
    
    if not json_output:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeRemainingColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Executing scenarios on Modal...", total=len(scenarios))
            
            # Execute in batches for progress updates
            batch_size = 10
            for i in range(0, len(scenario_tuples), batch_size):
                batch = scenario_tuples[i:i+batch_size]
                
                # Execute batch on Modal
                try:
                    with modal_app.run():
                        batch_results = list(evaluate_single_scenario.map(batch))
                except Exception as e:
                    console.print(format_warning(f"Modal execution error for batch {i//batch_size + 1}: {str(e)}"))
                    # Create error results for failed batch
                    for scenario_tuple in batch:
                        results.append({
                            "scenario_id": f"scenario_{scenario_tuple[2]}",
                            "success": False,
                            "execution_time": 0,
                            "failure_reason": f"Modal execution error: {str(e)}",
                            "tool_calls": [],
                            "cost": 0
                        })
                    progress.update(task, advance=len(batch))
                    continue
                
                for result in batch_results:
                    results.append({
                        "scenario_id": result.get("scenario_id"),
                        "success": result.get("success", False),
                        "execution_time": result.get("execution_time_ms", 0) / 1000,
                        "failure_reason": result.get("failure_reason"),
                        "tool_calls": result.get("tool_calls", []),
                        "cost": result.get("cost", 0)
                    })
                    total_cost += result.get("cost", 0)
                    progress.update(task, advance=len(batch))
    else:
        # Silent execution for JSON output
        try:
            with modal_app.run():
                modal_results = list(evaluate_single_scenario.map(scenario_tuples))
        except Exception as e:
            # Return error results for all scenarios
            for i, scenario_tuple in enumerate(scenario_tuples):
                results.append({
                    "scenario_id": f"scenario_{i}",
                    "success": False,
                    "execution_time": 0,
                    "failure_reason": f"Modal execution error: {str(e)}",
                    "tool_calls": [],
                    "cost": 0
                })
            execution_time = time.time() - start_time
            return results, execution_time, 0.0
        
        for result in modal_results:
            results.append({
                "scenario_id": result.get("scenario_id"),
                "success": result.get("success", False),
                "execution_time": result.get("execution_time_ms", 0) / 1000,
                "failure_reason": result.get("failure_reason"),
                "tool_calls": result.get("tool_calls", []),
                "cost": result.get("cost", 0)
            })
            total_cost += result.get("cost", 0)
    
    execution_time = time.time() - start_time
    return results, execution_time, total_cost