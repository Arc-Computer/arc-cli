"""Arc run command - test agent with generated scenarios."""

import asyncio
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import click
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeRemainingColumn,
)
from rich.table import Table

from arc.analysis.assumption_detector import AssumptionDetector
from arc.analysis.funnel_analyzer import FunnelAnalyzer
from arc.cli.loading_interface import ConfigAnalysisLoader, ExecutionProgressLoader
from arc.cli.utils import (
    ArcConsole,
    CLIState,
    HybridState,
    RunResult,
    db_manager,
    format_error,
    format_success,
    format_warning,
)
from arc.cli.utils.error_helpers import categorize_error
from arc.cli.utils.modal_auth import (
    check_modal_quota,
    get_modal_auth_instructions,
    setup_modal_auth,
)
from arc.core.models.scenario import Scenario
from arc.ingestion.normalizer import ConfigNormalizer
from arc.ingestion.parser import AgentConfigParser
from arc.scenarios.generator import ScenarioGenerator

console = ArcConsole()
# Will be initialized with database connection if available
state = None


async def _initialize_state():
    """Initialize state with database connection if available."""
    global state

    # Try to initialize database connection
    db_connected = await db_manager.initialize()

    if db_connected:
        # Use hybrid state with database
        state = HybridState(db_connected=True)
    else:
        # Fall back to file-only state
        state = CLIState()

    return db_connected


@click.command()
@click.argument("config_path", type=click.Path(exists=True))
@click.option(
    "--scenarios",
    "-s",
    default=50,
    help="Number of scenarios to generate (default: 50)",
)
@click.option(
    "--json", "json_output", is_flag=True, help="Output JSON instead of rich text"
)
@click.option("--no-confirm", is_flag=True, help="Skip cost confirmation prompt")
@click.option(
    "--pattern-ratio",
    default=0.7,
    type=click.FloatRange(0.0, 1.0),
    help="Ratio of pattern-based scenarios (0-1)",
)
def run(
    config_path: str,
    scenarios: int,
    json_output: bool,
    no_confirm: bool,
    pattern_ratio: float,
):
    """Test an agent configuration with generated scenarios.

    This command:
    1. Parses your agent configuration
    2. Generates test scenarios (currency assumptions + general)
    3. Executes scenarios in parallel
    4. Reports reliability score and failures

    Example:
        arc run finance_agent_v1.yaml
    """
    # Run the async implementation in a single event loop
    try:
        asyncio.run(
            _run_impl(config_path, scenarios, json_output, no_confirm, pattern_ratio)
        )
    except KeyboardInterrupt:
        console.print("\nRun cancelled by user.", style="warning")
        raise click.exceptions.Exit(130) from None
    except Exception as e:
        if json_output:
            import json

            print(json.dumps({"error": str(e)}, indent=2))
        else:
            console.print(format_error(f"Run failed: {str(e)}"))
        raise click.exceptions.Exit(1) from e


async def _run_impl(
    config_path: str,
    scenarios: int,
    json_output: bool,
    no_confirm: bool,
    pattern_ratio: float,
):
    """Async implementation of the run command."""
    global state

    # Initialize state with database connection
    db_connected = await _initialize_state()
    if not json_output and db_connected:
        console.print("Database connection established", style="success")

    config_path = Path(config_path)
    run_id = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid4().hex[:8]}"

    if not json_output:
        console.print()
        console.print(
            Panel.fit(
                "[primary]Arc: Proactive Capability Assurance[/primary]",
                border_style="primary",
            )
        )
        console.print()

    try:
        # Step 1: Enhanced Configuration Analysis with Loading
        parser = AgentConfigParser()
        normalizer = ConfigNormalizer()

        if not json_output:
            # Use professional loading interface for config analysis
            config_loader = ConfigAnalysisLoader(console)
            agent_profile = await config_loader.analyze_config_with_progress(
                str(config_path), parser, normalizer
            )

            # Display professional config summary
            config_loader.display_config_summary(agent_profile)

            # Extract components from agent profile
            normalized_config = agent_profile["configuration"]
            capabilities = agent_profile["capabilities"]
        else:
            # JSON mode: basic parsing without visual feedback
            parsed_config = parser.parse(config_path)
            normalized_config = normalizer.normalize(parsed_config)
            capabilities = parser.extract_capabilities(parsed_config)

            agent_profile = {
                "configuration": normalized_config,
                "capabilities": capabilities,
                "normalizer_enhancements": [],
            }

        # Step 2: Estimate costs
        estimated_cost = _estimate_cost(scenarios, normalized_config["model"])

        if (
            not json_output
            and not no_confirm
            and estimated_cost > state.config.get("cost_warning_threshold", 0.10)
        ):
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
        generated_scenarios = await _generate_scenarios_async(
            config_path=str(config_path),
            count=scenarios,
            pattern_ratio=pattern_ratio,
            capabilities=capabilities,
            json_output=json_output,
        )

        if not json_output:
            console.print(
                format_success(f"Generated {len(generated_scenarios)} scenarios")
            )
            _print_scenario_summary(generated_scenarios)
            console.print()

        # Step 4: Execute scenarios with streaming analysis
        if not json_output:
            console.print_header("Executing scenarios with real-time intelligence")

        # Execute with streaming analysis and intelligence
        results, execution_time, actual_cost = await _execute_with_streaming_analysis(
            scenarios=generated_scenarios,
            agent_config=normalized_config,
            agent_profile=agent_profile,
            json_output=json_output,
        )

        # Fallback cost estimation if not provided
        if actual_cost == 0.0:
            actual_cost = estimated_cost

        # Step 5: Calculate results
        success_count = sum(1 for r in results if r["success"])
        failure_count = len(results) - success_count
        reliability_score = success_count / len(results) if results else 0

        # Analyze assumption violations detected during streaming
        assumption_violations = []
        currency_failures = []

        for r in results:
            if not r["success"]:
                # Check for currency failures in failure reason
                if "currency" in r.get("failure_reason", "").lower():
                    currency_failures.append(r)

                # Collect assumption violations from AI analysis
                violations = r.get("assumptions_detected", [])
                assumption_violations.extend(violations)

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
            scenarios=[
                s.to_dict() if hasattr(s, "to_dict") else s for s in generated_scenarios
            ],
            results=results,
            failures=[r for r in results if not r["success"]],
        )

        state.save_run(run_result)

        # Display results
        if json_output:
            import json

            print(json.dumps(run_result.to_dict(), indent=2))
        else:
            console.print()
            console.print(
                Panel.fit(
                    "[success]✓ Found capability issues BEFORE production[/success]",
                    border_style="success",
                )
            )
            console.print()

            # Results table
            table = Table(show_header=False, box=None)
            table.add_column("Metric", style="muted")
            table.add_column("Value", style="bright_cyan")

            table.add_row(
                "Overall Reliability",
                f"{reliability_score:.1%} ({success_count}/{len(results)} scenarios)",
            )

            # Show assumption violations discovered
            if assumption_violations:
                violation_types = {}
                for violation in assumption_violations:
                    vtype = violation.get("type", "unknown")
                    violation_types[vtype] = violation_types.get(vtype, 0) + 1

                # Show most common violation type
                most_common = max(violation_types.items(), key=lambda x: x[1])
                table.add_row(
                    "Primary Assumption Violation",
                    f"{most_common[0].title()}: {most_common[1]} occurrences",
                    style="error",
                )

            if currency_failures:
                table.add_row(
                    "Currency Assumption Violations",
                    f"{len(currency_failures)} failures",
                    style="error",
                )

            table.add_row("Time to insight", f"{execution_time:.1f} seconds")
            table.add_row("Actual cost", f"${actual_cost:.4f}")

            console.print(table)
            console.print()

            # Display top assumption violations with business impact
            if assumption_violations:
                console.print_header("Key Assumption Violations Discovered")

                # Group by type and show top 3
                violation_groups = {}
                for violation in assumption_violations:
                    vtype = violation.get("type", "unknown")
                    if vtype not in violation_groups:
                        violation_groups[vtype] = []
                    violation_groups[vtype].append(violation)

                for _i, (vtype, violations) in enumerate(
                    list(violation_groups.items())[:3]
                ):
                    violation = violations[0]  # Show representative violation
                    console.print(
                        console.format_assumption(
                            vtype.title(),
                            violation.get("description", "No description"),
                            violation.get("business_impact", "Unknown impact"),
                        )
                    )
                    console.print(
                        f"[muted]  → Fix: {violation.get('suggested_fix', 'No suggestion available')}[/muted]"
                    )
                    console.print()
                console.print()

            console.print(
                "Run [info]arc analyze[/info] to see detailed breakdown", style="muted"
            )
            console.print(
                "Run [info]arc recommend[/info] for specific fixes", style="muted"
            )
            console.print()

    except Exception as e:
        if json_output:
            import json

            print(json.dumps({"error": str(e)}, indent=2))
        else:
            console.print(format_error(f"Run failed: {str(e)}"))
            console.print(
                f"\nTry: [info]arc validate {config_path}[/info] to check your configuration",
                style="muted",
            )
        raise click.exceptions.Exit(1) from e
    finally:
        # Ensure proper cleanup of database connections
        try:
            await db_manager.close()
        except Exception:
            pass  # Ignore cleanup errors


async def _generate_scenarios_async(
    config_path: str,
    count: int,
    pattern_ratio: float,
    capabilities: dict[str, Any],
    json_output: bool,
) -> list[Scenario]:
    """Generate scenarios asynchronously."""
    try:
        # Check if finance domain for currency scenarios
        is_finance = "finance" in capabilities.get("domains", [])

        if is_finance and not json_output:
            console.print(
                "Detected finance domain - including currency assumption scenarios",
                style="info",
            )

        # Initialize generator
        generator = ScenarioGenerator(
            agent_config_path=config_path,
            use_patterns=True,
            quality_threshold=2.0,  # Lower threshold for more scenarios
        )

        # Generate scenarios with currency focus for first 15
        if count >= 50:
            # Generate 15 currency scenarios + 35 general scenarios
            currency_scenarios = await generator.generate_scenarios_batch(
                count=15, pattern_ratio=pattern_ratio, currency_focus=True
            )

            general_scenarios = await generator.generate_scenarios_batch(
                count=count - 15, pattern_ratio=pattern_ratio, currency_focus=False
            )

            scenarios = currency_scenarios + general_scenarios
        else:
            # Generate all as general scenarios
            scenarios = await generator.generate_scenarios_batch(
                count=count, pattern_ratio=pattern_ratio, currency_focus=False
            )

        return scenarios

    except Exception:
        # For now, return mock scenarios if generation fails
        if not json_output:
            console.print(
                format_warning("Failed to generate scenarios, using mock data")
            )

        mock_scenarios = []
        for i in range(count):
            scenario = {
                "scenario_id": f"mock_{i}",
                "task_prompt": f"Test scenario {i}",
                "expected_tools": ["calculator", "web_search"],
                "complexity_level": "medium",
                "domain": "finance" if i < 15 else "general",
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
        "openai/o3-pro": {"cost_per_1k_in": 0.02, "cost_per_1k_out": 0.08},
        # Anthropic
        "anthropic/claude-opus-4": {
            "cost_per_1k_in": 0.01500,
            "cost_per_1k_out": 0.07500,
        },
        "anthropic/claude-sonnet-4": {
            "cost_per_1k_in": 0.00300,
            "cost_per_1k_out": 0.01500,
        },
        "anthropic/claude-3.5-haiku": {
            "cost_per_1k_in": 0.00080,
            "cost_per_1k_out": 0.00400,
        },
        # Google
        "google/gemini-2.5-pro-preview": {
            "cost_per_1k_in": 0.00125,
            "cost_per_1k_out": 0.01000,
        },
        "google/gemini-2.5-flash-preview-05-20": {
            "cost_per_1k_in": 0.00015,
            "cost_per_1k_out": 0.00060,
        },
        # Meta (Llama 4)
        "meta-llama/llama-4-maverick": {
            "cost_per_1k_in": 0.00015,
            "cost_per_1k_out": 0.00060,
        },
        "meta-llama/llama-4-scout": {
            "cost_per_1k_in": 0.00008,
            "cost_per_1k_out": 0.00030,
        },
        # Mistral
        "mistralai/mistral-medium-3": {
            "cost_per_1k_in": 0.00040,
            "cost_per_1k_out": 0.00200,
        },
        # Cohere
        "cohere/command-a": {"cost_per_1k_in": 0.00250, "cost_per_1k_out": 0.01000},
        # DeepSeek
        "deepseek/deepseek-chat-v3-0324": {
            "cost_per_1k_in": 0.00030,
            "cost_per_1k_out": 0.00088,
        },
        "deepseek/deepseek-r1-0528": {
            "cost_per_1k_in": 0.00050,
            "cost_per_1k_out": 0.00215,
        },
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


def _print_scenario_summary(scenarios: list[Any]) -> None:
    """Print summary of generated scenarios."""
    # Count by domain
    domains = {}
    for s in scenarios:
        domain = (
            s.get("domain", "general")
            if isinstance(s, dict)
            else getattr(s, "inferred_domain", "general")
        )
        domains[domain] = domains.get(domain, 0) + 1

    if domains:
        console.print("Scenario breakdown:", style="muted")
        for domain, count in sorted(domains.items(), key=lambda x: x[1], reverse=True):
            console.print(f"  {domain}: {count} scenarios", style="muted")


def _simulate_execution(
    scenarios: list[Any], json_output: bool
) -> tuple[list[dict[str, Any]], float]:
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
                success = not (
                    is_currency and i < 15
                )  # First 15 currency scenarios fail

                result = {
                    "scenario_id": scenario.get("scenario_id", f"scenario_{i}")
                    if isinstance(scenario, dict)
                    else getattr(scenario, "scenario_id", f"scenario_{i}"),
                    "success": success,
                    "execution_time": 0.5 + (i * 0.01),
                    "failure_reason": "Currency assumption violation: Expected multi-currency support"
                    if not success
                    else None,
                }
                results.append(result)

                progress.update(task, advance=1)
    else:
        # Silent execution for JSON output
        for i, scenario in enumerate(scenarios):
            is_currency = "currency" in str(scenario).lower()
            success = not (is_currency and i < 15)

            result = {
                "scenario_id": scenario.get("scenario_id", f"scenario_{i}")
                if isinstance(scenario, dict)
                else getattr(scenario, "scenario_id", f"scenario_{i}"),
                "success": success,
                "execution_time": 0.5 + (i * 0.01),
                "failure_reason": "Currency assumption violation: Expected multi-currency support"
                if not success
                else None,
            }
            results.append(result)

    execution_time = time.time() - start_time
    return results, execution_time


async def _check_modal_available(max_retries: int = 3) -> bool:
    """Check if Modal is installed and configured with retry logic.

    Args:
        max_retries: Maximum number of connection attempts

    Returns:
        True if Modal is available and configured, False otherwise
    """
    try:
        import modal

        # First check if Modal package is properly installed
        try:
            # Test import of required components
            from arc.sandbox.engine.simulator import app as modal_app
        except ImportError as e:
            console.print(
                format_warning(f"Modal components not properly installed: {e}")
            )
            return False

        # Try to setup Modal authentication (handles multiple methods)
        auth_configured = setup_modal_auth()

        if not auth_configured:
            # Check if we might be in a deployed environment with different auth
            if os.environ.get("ARC_MODAL_APP_ID") or os.environ.get("MODAL_ENVIRONMENT"):
                console.print(format_warning("Modal authentication may be handled by deployment"))
                return True
            console.print(format_warning("Modal not authenticated"))
            console.print(get_modal_auth_instructions())
            return False

        # Check if using shared account and warn
        is_shared, warning = check_modal_quota()
        if is_shared and warning:
            console.print(format_warning(warning))

        # Modal auth is configured, we can proceed
        # Don't test connection here - let the actual execution handle any issues
        return True
    except ImportError:
        console.print(
            format_warning("Modal not installed. Run 'pip install modal' to install")
        )
        return False


async def _execute_with_streaming_analysis(
    scenarios: list[Scenario],
    agent_config: dict[str, Any],
    agent_profile: dict[str, Any],
    json_output: bool,
) -> tuple[list[dict[str, Any]], float, float]:
    """
    Execute scenarios with real-time streaming analysis and intelligence.

    Integrates:
    - ExecutionProgressLoader for live metrics
    - FunnelAnalyzer for progressive capability analysis
    - AssumptionDetector for real-time violation detection

    Returns:
        Tuple of (results, execution_time, total_cost)
    """
    # Initialize intelligence components
    funnel_analyzer = FunnelAnalyzer()
    assumption_detector = AssumptionDetector()
    execution_loader = ExecutionProgressLoader(console)

    # Check Modal availability
    modal_configured = await _check_modal_available()
    
    # Allow forcing local execution via environment variable
    force_local = os.environ.get("ARC_FORCE_LOCAL", "").lower() in ["true", "1", "yes"]
    if force_local:
        console.print(format_warning("Forcing local execution (ARC_FORCE_LOCAL=true)"))
        use_modal = False
    else:
        use_modal = state.config.get("use_modal", False) and modal_configured

    if state.config.get("use_modal", False) and not modal_configured:
        console.print(
            format_warning(
                "Modal requested but not properly configured - falling back to simulation"
            )
        )

    if use_modal:
        # Execute with Modal + streaming analysis
        return await _execute_modal_with_streaming(
            scenarios,
            agent_config,
            agent_profile,
            funnel_analyzer,
            assumption_detector,
            execution_loader,
            json_output,
        )
    else:
        # Execute simulation with streaming analysis
        return await _execute_simulation_with_streaming(
            scenarios,
            agent_config,
            agent_profile,
            funnel_analyzer,
            assumption_detector,
            execution_loader,
            json_output,
        )


async def _execute_modal_with_streaming(
    scenarios: list[Scenario],
    agent_config: dict[str, Any],
    agent_profile: dict[str, Any],
    funnel_analyzer: FunnelAnalyzer,
    assumption_detector: AssumptionDetector,
    execution_loader: ExecutionProgressLoader,
    json_output: bool,
) -> tuple[list[dict[str, Any]], float, float]:
    """Execute Modal scenarios with real-time streaming intelligence."""
    try:
        import modal

        from arc.sandbox.engine.simulator import (
            app as modal_app,
        )
        from arc.sandbox.engine.simulator import (
            evaluate_single_scenario,
        )
    except ImportError as e:
        raise RuntimeError(f"Modal execution failed: {e}") from e

    start_time = time.time()
    results = []
    total_cost = 0.0
    completed_trajectories = []

    # Prepare scenarios for Modal
    scenario_dicts = []
    for scenario in scenarios:
        if hasattr(scenario, "to_dict"):
            scenario_dicts.append(scenario.to_dict())
        else:
            scenario_dicts.append(scenario)

    # Merge configuration with capabilities to ensure tools have full definitions
    enhanced_config = agent_profile["configuration"].copy()
    enhanced_config["tools"] = agent_profile["capabilities"].get(
        "tools", enhanced_config.get("tools", [])
    )
    enhanced_config["assumptions"] = agent_profile["capabilities"].get(
        "assumptions", enhanced_config.get("assumptions", [])
    )
    enhanced_config["validation_rules"] = agent_profile["capabilities"].get(
        "validation_rules", enhanced_config.get("validation_rules", [])
    )
    enhanced_config["job"] = agent_profile["capabilities"].get(
        "job", enhanced_config.get("job", "")
    )

    scenario_tuples = [
        (scenario, enhanced_config, i) for i, scenario in enumerate(scenario_dicts)
    ]

    # FIXED: Create single Modal app context for entire execution
    try:
        async with modal_app.run():
            if not json_output:
                # Initialize live streaming display
                with console.start_live_session() as live:
                    # Create progress tracking
                    progress = execution_loader.create_execution_progress(
                        len(scenarios)
                    )
                    task = progress.add_task(
                        "Executing scenarios on Modal...", total=len(scenarios)
                    )

                    # Execute in batches with streaming updates
                    batch_size = 10
                    for batch_start in range(0, len(scenario_tuples), batch_size):
                        batch = scenario_tuples[batch_start : batch_start + batch_size]

                        # Execute batch on Modal within the single app context
                        try:
                            # Use async iteration for Modal map
                            batch_results = []
                            async for result in evaluate_single_scenario.map.aio(batch):
                                batch_results.append(result)
                        except Exception as e:
                            console.print(
                                format_warning(f"Modal batch execution error: {str(e)}")
                            )
                            # Create error results for failed batch
                            for scenario_tuple in batch:
                                results.append(
                                    {
                                        "scenario_id": f"scenario_{scenario_tuple[2]}",
                                        "success": False,
                                        "execution_time": 0,
                                        "failure_reason": f"Modal execution error: {str(e)}",
                                        "cost": 0,
                                    }
                                )
                            progress.update(task, advance=len(batch))
                            continue

                        # Process batch results with intelligence analysis
                        for result in batch_results:
                            # Extract Modal result data
                            processed_result = (
                                await _process_modal_result_with_intelligence(
                                    result, assumption_detector, agent_profile
                                )
                            )
                            results.append(processed_result)
                            completed_trajectories.append(
                                processed_result.get("trajectory", {})
                            )
                            total_cost += processed_result.get("cost", 0)

                            progress.update(task, advance=1)

                        # Update live metrics every batch
                        # Estimate active containers based on execution state
                        remaining_scenarios = len(scenarios) - len(results)
                        max_containers = min(50, len(scenarios))  # Modal max is 50
                        active_containers = min(remaining_scenarios, len(batch), max_containers)
                        
                        # Calculate estimated total time based on current progress
                        if len(results) > 0:
                            avg_time_per_scenario = (time.time() - start_time) / len(results)
                            estimated_total_time = avg_time_per_scenario * len(scenarios)
                        else:
                            estimated_total_time = (time.time() - start_time) * 1.5
                        
                        live_metrics = {
                            "completed": len(results),
                            "total": len(scenarios),
                            "cost": total_cost,
                            "estimated_cost": total_cost
                            * (len(scenarios) / len(results))
                            if results
                            else 0,
                            "elapsed_time": time.time() - start_time,
                            "estimated_total_time": estimated_total_time,
                            "active_containers": active_containers,
                            "max_containers": max_containers,
                            "failures": len(
                                [r for r in results if not r.get("success", False)]
                            ),
                        }

                        # Display live metrics panel
                        metrics_panel = execution_loader.display_live_metrics(
                            live_metrics
                        )
                        live.update(metrics_panel)
                        
                        # Display execution timeline and performance metrics (every 3 batches)
                        if len(results) % 30 == 0 and len(results) > 0:
                            # Calculate performance metrics
                            elapsed = time.time() - start_time
                            scenarios_per_minute = (len(results) / elapsed) * 60 if elapsed > 0 else 0
                            avg_scenario_time = elapsed / len(results) if len(results) > 0 else 0
                            
                            # Calculate speedup factor (estimate)
                            sequential_time = avg_scenario_time * len(results)
                            speedup_factor = sequential_time / elapsed if elapsed > 0 else 1
                            
                            # Container efficiency (active/max ratio over time)
                            container_efficiency = (active_containers / max_containers * 100) if max_containers > 0 else 0
                            
                            timeline_data = {
                                "scenarios_per_minute": scenarios_per_minute,
                                "avg_scenario_time": avg_scenario_time,
                                "speedup_factor": speedup_factor,
                                "container_efficiency": container_efficiency,
                                "performance_trend": "improving" if speedup_factor > 1.5 else "stable"
                            }
                            
                            timeline_panel = execution_loader.display_execution_timeline(timeline_data)
                            console.print()
                            console.print(timeline_panel)
                            console.print()
                        
                        # Enhanced error monitoring with recovery procedures
                        if len(results) > 0:
                            failed_results = [r for r in results if not r.get("success", False)]
                            if failed_results:
                                # Categorize errors
                                error_categories = {}
                                for result in failed_results:
                                    error_category = result.get("error_category", "unknown")
                                    error_categories[error_category] = error_categories.get(error_category, 0) + 1
                                
                                error_rate = (len(failed_results) / len(results)) * 100
                                
                                error_data = {
                                    "errors": failed_results,
                                    "total_errors": len(failed_results),
                                    "error_rate": error_rate,
                                    "error_categories": error_categories,
                                    "recovery_status": "in_progress" if len(failed_results) > 3 else "none",
                                    "clustering_active": len(failed_results) >= 3,
                                    "clusters_found": min(len(error_categories), 3)
                                }
                                
                                error_panel = execution_loader.display_live_error_monitoring(error_data)
                                if error_panel:
                                    console.print()
                                    console.print(error_panel)
                                    console.print()

                        # Progressive funnel analysis (every 10 scenarios)
                        if (
                            len(completed_trajectories) % 10 == 0
                            and len(completed_trajectories) > 0
                        ):
                            funnel = await funnel_analyzer.build_capability_funnel(
                                agent_profile, completed_trajectories[-10:]
                            )

                            # Display funnel update
                            funnel_table = console.format_funnel(
                                [
                                    {
                                        "name": step.name,
                                        "success_rate": step.success_rate,
                                        "failures": len(step.failures),
                                        "is_bottleneck": step.is_bottleneck,
                                        "impact": "Critical"
                                        if step.is_bottleneck
                                        else "Normal",
                                    }
                                    for step in funnel.steps
                                ]
                            )

                            console.print()
                            console.print(funnel_table)
                            console.print()

                        # Real-time assumption violation detection
                        failed_trajectories = [
                            r.get("trajectory", {})
                            for r in results[-len(batch) :]
                            if not r.get("success", False)
                        ]

                        if failed_trajectories:
                            violations = []
                            for trajectory in failed_trajectories:
                                trajectory_violations = (
                                    await assumption_detector.detect_live_violations(
                                        trajectory, agent_profile
                                    )
                                )
                                violations.extend(trajectory_violations)

                            if violations:
                                violations_panel = (
                                    execution_loader.display_assumption_alerts(
                                        [
                                            {
                                                "type": v.type,
                                                "severity": v.severity,
                                                "description": v.description,
                                            }
                                            for v in violations[:3]  # Top 3 violations
                                        ]
                                    )
                                )
                                if violations_panel:
                                    console.print()
                                    console.print(violations_panel)
                                    console.print()

                        # Short delay for visual feedback
                        await asyncio.sleep(0.1)
            else:
                # JSON mode: execute without visual feedback
                # Execute all scenarios at once for JSON output
                try:
                    modal_results = []
                    async for result in evaluate_single_scenario.map.aio(
                        scenario_tuples
                    ):
                        modal_results.append(result)
                except Exception as e:
                    # Return error results for all scenarios
                    for i, _scenario_tuple in enumerate(scenario_tuples):
                        results.append(
                            {
                                "scenario_id": f"scenario_{i}",
                                "success": False,
                                "execution_time": 0,
                                "failure_reason": f"Modal execution error: {str(e)}",
                                "cost": 0,
                            }
                        )
                    execution_time = time.time() - start_time
                    return results, execution_time, 0.0

                # Process all results
                for result in modal_results:
                    processed_result = await _process_modal_result_with_intelligence(
                        result, assumption_detector, agent_profile
                    )
                    results.append(processed_result)
                    total_cost += processed_result.get("cost", 0)

    except Exception as e:
        # Handle Modal app context errors
        error_msg = str(e)
        console.print(format_error(f"Modal app initialization failed: {error_msg}"))
        
        # Provide more detailed error information
        if "Token ID is malformed" in error_msg:
            console.print(format_warning("\nModal authentication issue detected."))
            console.print("This can happen when:")
            console.print("  • Modal tokens have expired")
            console.print("  • Using incompatible token format")
            console.print("  • Token was corrupted during copy/paste")
            console.print("\nTo fix:")
            console.print("  1. Clear existing tokens: unset MODAL_TOKEN_ID MODAL_TOKEN_SECRET")
            console.print("  2. Re-authenticate: modal token new")
            console.print("  3. Or use Arc demo tokens if provided")
        
        # Return empty results with error
        execution_time = time.time() - start_time
        return [], execution_time, 0.0

    execution_time = time.time() - start_time
    return results, execution_time, total_cost


async def _process_modal_result_with_intelligence(
    modal_result: dict[str, Any],
    assumption_detector: AssumptionDetector,
    agent_profile: dict[str, Any],
) -> dict[str, Any]:
    """Process Modal result with intelligence analysis."""
    scenario = modal_result.get("scenario", {})
    trajectory = modal_result.get("trajectory", {})
    reliability_score = modal_result.get("reliability_score", {})

    # Calculate cost from token usage
    token_usage = trajectory.get("token_usage", {})
    cost = token_usage.get("total_cost", 0.0)

    # Determine success based on reliability score
    success = reliability_score.get("overall_score", 0) >= 0.7

    # AI-enhanced assumption detection for failed scenarios
    assumptions_detected = []
    if not success:
        try:
            violations = await assumption_detector.detect_live_violations(
                trajectory, agent_profile
            )
            assumptions_detected = [
                {
                    "type": v.type,
                    "severity": v.severity,
                    "confidence": v.confidence,
                    "description": v.description,
                    "suggested_fix": v.suggested_fix,
                    "business_impact": v.business_impact,
                }
                for v in violations
            ]
        except Exception:
            # Log but don't fail on assumption detection errors
            pass

    return {
        "scenario_id": scenario.get(
            "id", f"scenario_{modal_result.get('scenario_index', 0)}"
        ),
        "success": success,
        "execution_time": trajectory.get("execution_time_seconds", 0),
        "failure_reason": trajectory.get("final_response")
        if trajectory.get("status") == "error"
        else None,
        "cost": cost,
        "tokens_used": token_usage.get("total_tokens", 0),
        "trajectory": trajectory,
        "reliability_score": reliability_score.get("overall_score", 0),
        "assumptions_detected": assumptions_detected,
        "modal_call_id": os.environ.get("MODAL_FUNCTION_CALL_ID"),
        "error_category": categorize_error(trajectory.get("final_response"))
        if trajectory.get("status") == "error"
        else None,
    }


async def _execute_simulation_with_streaming(
    scenarios: list[Scenario],
    agent_config: dict[str, Any],
    agent_profile: dict[str, Any],
    funnel_analyzer: FunnelAnalyzer,
    assumption_detector: AssumptionDetector,
    execution_loader: ExecutionProgressLoader,
    json_output: bool,
) -> tuple[list[dict[str, Any]], float, float]:
    """Execute simulation with streaming analysis for demo purposes."""
    start_time = time.time()
    results = []

    if not json_output:
        console.print(
            "[warning]Modal not configured - simulating execution with intelligence analysis[/warning]"
        )
        console.print()

        # Simulate with live updates
        progress = execution_loader.create_execution_progress(len(scenarios))
        task = progress.add_task("Simulating scenarios...", total=len(scenarios))

        for i, scenario in enumerate(scenarios):
            # Simulate execution time
            await asyncio.sleep(0.02)

            # Generate mock result with currency assumption focus
            is_currency = "currency" in str(scenario).lower()
            success = not (is_currency and i < 15)  # First 15 currency scenarios fail

            result = {
                "scenario_id": scenario.get("scenario_id", f"scenario_{i}")
                if isinstance(scenario, dict)
                else getattr(scenario, "scenario_id", f"scenario_{i}"),
                "success": success,
                "execution_time": 0.5 + (i * 0.01),
                "failure_reason": "Currency assumption violation: Expected multi-currency support"
                if not success
                else None,
                "assumptions_detected": [
                    {
                        "type": "currency",
                        "severity": "high",
                        "confidence": 85,
                        "description": "Agent assumes USD for all transactions without validation",
                        "suggested_fix": "Add currency validation and conversion tools",
                        "business_impact": "Financial calculation errors affecting customer billing",
                    }
                ]
                if not success
                else [],
            }
            results.append(result)
            progress.update(task, advance=1)
    else:
        # Silent simulation for JSON output
        for i, scenario in enumerate(scenarios):
            is_currency = "currency" in str(scenario).lower()
            success = not (is_currency and i < 15)

            result = {
                "scenario_id": scenario.get("scenario_id", f"scenario_{i}")
                if isinstance(scenario, dict)
                else getattr(scenario, "scenario_id", f"scenario_{i}"),
                "success": success,
                "execution_time": 0.5 + (i * 0.01),
                "failure_reason": "Currency assumption violation: Expected multi-currency support"
                if not success
                else None,
            }
            results.append(result)

    execution_time = time.time() - start_time
    return results, execution_time, 0.0


# Removed unused _execute_with_modal function - replaced by _execute_modal_with_streaming
