"""
Modal Orchestrator for cost-aware, persistent, and resilient scenario execution.

This module provides the ModalOrchestrator class, which wraps Modal's execution
engine to provide several key features for the Arc platform:

1.  **Cost Transparency**: Estimates execution costs before a run and tracks
    actual costs in real-time.
2.  **Persistent Results**: Integrates with a database client to save all
    simulation results, statuses, and trajectories.
3.  **Real-time Monitoring**: Yields progress updates that can be displayed
    in a CLI to monitor container scaling, completion status, and costs.
4.  **Resilience**: Implements fallback mechanisms to handle potential issues
    with Modal or the database connection, ensuring that evaluations can
    still run.
5.  **Simplified Interface**: Abstracts away the direct complexities of Modal,
    providing a clean, high-level API for running evaluation suites.
"""
import asyncio
import logging
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, AsyncGenerator, Dict, List, Optional
from uuid import UUID

import modal
from rich.panel import Panel

from arc.core.models.scenario import Scenario
from arc.core.models.config import AgentConfig
from arc.database.client import ArcDBClient
from arc.database.api import ArcAPI
from arc.database.batch_processor import BatchExecutionRecorder, BatchConfig, batch_processor_context
from arc.database.utils import normalize_modal_result
from arc.cli.utils import ArcConsole
from arc.sandbox.engine.simulator import app as modal_app, evaluate_single_scenario

logger = logging.getLogger(__name__)


@dataclass
class ProgressUpdate:
    """Represents a real-time update on the execution progress."""
    completed: int
    total: int
    current_cost: float
    estimated_cost: float
    active_containers: int = 0
    status_message: str = "Executing..."
    latest_result: Optional[Dict[str, Any]] = None
    batch_metrics: Optional[Dict[str, Any]] = None  # New: batch processing metrics

    @property
    def progress_pct(self) -> float:
        """Returns the completion percentage."""
        return self.completed / self.total if self.total > 0 else 0


class ModalOrchestrator:
    """
    Orchestrates the execution of agent evaluation scenarios using Modal,
    with built-in support for cost estimation, database persistence, and
    real-time progress monitoring.
    """
    def __init__(self, db_client: Optional[ArcDBClient] = None, console: Optional[ArcConsole] = None):
        """
        Initializes the ModalOrchestrator.

        Args:
            db_client: An optional client for database interactions. If provided,
                       results and progress will be persisted.
            console: An optional console object for logging output.
        """
        self.db_client = db_client
        self.console = console or ArcConsole()
        self.modal_app_name = "arc-eval-sandbox"
        
        # Batch processing configuration based on environment
        env = os.getenv('ARC_ENV', 'development').lower()
        if env == 'production':
            self.batch_config = BatchConfig.for_production()
        elif env == 'testing':
            self.batch_config = BatchConfig.for_testing()
        else:
            self.batch_config = BatchConfig.for_development()
        
        logger.info(f"Using batch configuration for {env} environment: "
                   f"batch_size={self.batch_config.max_batch_size}, "
                   f"flush_interval={self.batch_config.flush_interval_seconds}s")
        
        # Verify Modal version compatibility
        self._verify_modal_version()

    def _verify_modal_version(self) -> None:
        """
        Checks if the installed Modal version is compatible.
        Raises an error if a potentially breaking version is detected.
        """
        version = modal.__version__
        if not version.startswith("0."):
            self.console.print(
                Panel(
                    f"[bold yellow]Warning:[/bold yellow] You are using Modal version {version}. "
                    "This version may have breaking changes.\n"
                    "For stability, please pin to a version `<1.0`.",
                    title="Compatibility Alert",
                    border_style="yellow",
                )
            )

    async def execute_with_persistence(
        self,
        agent_config: AgentConfig,
        scenarios: List[Scenario],
        run_id: str,
    ) -> AsyncGenerator[ProgressUpdate, None]:
        """
        Executes a suite of scenarios on Modal with real-time progress updates
        and database persistence.

        This is the core method that implements:
        1. Pre-execution cost estimation
        2. Real-time CLI monitoring 
        3. Persistent results in TimescaleDB
        4. Container scaling awareness
        """
        total_scenarios = len(scenarios)
        estimated_cost = self._estimate_execution_cost(total_scenarios, agent_config.model)
        completed_count = 0
        total_cost = 0.0
        active_containers = 0
        
        self.console.print(f"[bold blue]Estimated cost: ${estimated_cost:.4f}[/bold blue]")
        
        # Create simulation record and database API
        simulation_id = None
        db_api = None
        if self.db_client:
            try:
                db_api = ArcAPI(self.db_client)
                
                # Create a proper config version first
                import uuid
                config_version_id = await self.db_client.create_configuration(
                    name=f"Agent Config {run_id}",
                    user_id=str(uuid.uuid4()),  # Generate proper UUID for system user
                    initial_config=agent_config.model_dump()
                )
                
                scenario_ids = [scenario.id for scenario in scenarios]
                simulation_id = await self.db_client.create_simulation(
                    config_version_id=config_version_id,
                    scenario_set=scenario_ids,
                    simulation_name=f"Modal Run {run_id}",
                    modal_app_id=self.modal_app_name
                )
                self.console.print(f"[bold green]Database tracking enabled[/bold green] - Simulation ID: {simulation_id[:8]}...")
                self.console.print(f"[bold cyan]Batch processing enabled[/bold cyan] - Batch size: {self.batch_config.max_batch_size}")
            except Exception as e:
                self.console.print(f"[bold yellow]Warning:[/bold yellow] Database tracking disabled: {e}")
                simulation_id = None
                db_api = None
        
        # Initial progress update
        yield ProgressUpdate(
            completed=0,
            total=total_scenarios,
            current_cost=0,
            estimated_cost=estimated_cost,
            active_containers=0,
            status_message="Initializing Modal execution with batch processing..."
        )

        # Batch processing context
        batch_processor = None
        if db_api and simulation_id:
            def on_batch_complete(batch_size: int, processing_time: float):
                """Callback for batch completion events."""
                records_per_sec = batch_size / processing_time if processing_time > 0 else 0
                self.console.print(f"[dim]Batch processed: {batch_size} records in {processing_time:.2f}s ({records_per_sec:.1f} rec/s)[/dim]")

            async with batch_processor_context(
                db_api, 
                self.batch_config, 
                on_batch_complete
            ) as batch_proc:
                batch_processor = batch_proc
                
                # Execute scenarios with batch processing
                async for update in self._execute_scenarios_with_batching(
                    agent_config, scenarios, simulation_id, batch_processor,
                    estimated_cost, total_scenarios
                ):
                    yield update
        else:
            # Execute without batch processing (fallback)
            async for update in self._execute_scenarios_without_batching(
                agent_config, scenarios, estimated_cost, total_scenarios
            ):
                yield update

    async def _execute_scenarios_with_batching(
        self,
        agent_config: AgentConfig,
        scenarios: List[Scenario],
        simulation_id: str,
        batch_processor: BatchExecutionRecorder,
        estimated_cost: float,
        total_scenarios: int
    ) -> AsyncGenerator[ProgressUpdate, None]:
        """Execute scenarios with batch processing enabled."""
        completed_count = 0
        total_cost = 0.0
        active_containers = 0

        try:
            # Check if Modal is available, otherwise simulate
            modal_available = True
            try:
                # Check for deployed app first
                if os.environ.get("ARC_USE_DEPLOYED_APP"):
                    try:
                        from arc.modal.public_api import ArcModalAPI
                        modal_available = ArcModalAPI.is_available()
                        if modal_available:
                            self.console.print("[bold green]Using deployed Modal web endpoint[/bold green]")
                        else:
                            self.console.print("[bold yellow]Modal web endpoint not available - falling back to simulation[/bold yellow]")
                    except ImportError:
                        modal_available = False
                        self.console.print("[bold yellow]Modal public API not available - falling back to simulation[/bold yellow]")
                else:
                    # Check for local Modal authentication
                    try:
                        import modal
                        # Try to authenticate - this will fail if no token
                        modal.Function.from_name("test", "test")
                    except Exception:
                        modal_available = False
                        self.console.print("[bold yellow]Modal authentication not available - falling back to simulation[/bold yellow]")
            except Exception as e:
                modal_available = False
                self.console.print(f"[bold yellow]Modal check failed: {e} - falling back to simulation[/bold yellow]")

            if modal_available:
                # Check if using deployed app
                if os.environ.get("ARC_USE_DEPLOYED_APP"):
                    # Execute with deployed Modal app
                    self.console.print("[bold green]Executing on deployed Modal app with batch processing...[/bold green]")
                    
                    from arc.modal.public_api import ArcModalAPI
                    
                    # Convert scenarios to format expected by deployed app
                    scenario_dicts = []
                    for scenario in scenarios:
                        if hasattr(scenario, 'model_dump'):
                            scenario_dicts.append(scenario.model_dump())
                        elif hasattr(scenario, 'to_dict'):
                            scenario_dicts.append(scenario.to_dict())
                        else:
                            scenario_dicts.append(scenario)

                    active_containers = min(len(scenario_dicts), 50)  # Max containers
                    
                    yield ProgressUpdate(
                        completed=0,
                        total=total_scenarios,
                        current_cost=0,
                        estimated_cost=estimated_cost,
                        active_containers=active_containers,
                        status_message=f"Running {active_containers} Modal containers with batch processing...",
                        batch_metrics=batch_processor.get_status()
                    )
                    
                    # Execute scenarios using deployed app with parallel batches
                    async for result in ArcModalAPI.run_scenarios(
                        scenarios=scenario_dicts,
                        agent_config=agent_config.model_dump(),
                        batch_size=20  # Process 20 scenarios in parallel per batch
                    ):
                        completed_count += 1
                        
                        # Calculate cost for this scenario
                        scenario_cost = 0.001  # Default cost
                        if not isinstance(result, Exception):
                            trajectory = result.get("trajectory", {})
                            token_usage = trajectory.get("token_usage", {})
                            scenario_cost = token_usage.get("total_cost", 0.001)
                        
                        total_cost += scenario_cost
                        
                        # Update active containers (simulate scaling down)
                        remaining = total_scenarios - completed_count
                        active_containers = min(remaining, 50) if remaining > 0 else 0
                        
                        # Add to batch processor instead of immediate database write
                        try:
                            # Transform the result to ensure 0-1 scale reliability scores for database
                            transformed_result = normalize_modal_result(result)
                            await batch_processor.add_outcome(
                                simulation_id=simulation_id,
                                scenario_result=transformed_result,
                                modal_call_id=f"deployed_modal_{completed_count}",
                                sandbox_id=f"deployed_sandbox_{completed_count}"
                            )
                        except Exception as e:
                            # Log error but don't stop execution - batch processor handles retries
                            logger.error(f"Failed to add outcome to batch processor: {e}")
                            self.console.print(f"[bold red]Batch Processing Error:[/bold red] {e}")
                            # For critical errors, we might want to fall back to individual processing
                            if "ResourceLimitError" in str(type(e)) or "ValueError" in str(type(e)):
                                logger.critical(f"Critical batch processing error: {e}")
                                raise
                        
                        # Yield real-time progress update with batch metrics
                        yield ProgressUpdate(
                            completed=completed_count,
                            total=total_scenarios,
                            current_cost=total_cost,
                            estimated_cost=estimated_cost,
                            active_containers=active_containers,
                            status_message=f"Completed {completed_count}/{total_scenarios} scenarios",
                            latest_result=result if not isinstance(result, Exception) else {"error": str(result)},
                            batch_metrics=batch_processor.get_status()
                        )
                else:
                    # Execute with authenticated Modal
                    self.console.print("[bold green]Executing on Modal with batch processing...[/bold green]")
                    
                    # Convert scenarios to Modal format
                    scenario_dicts = []
                    for scenario in scenarios:
                        if hasattr(scenario, 'model_dump'):
                            scenario_dicts.append(scenario.model_dump())
                        elif hasattr(scenario, 'to_dict'):
                            scenario_dicts.append(scenario.to_dict())
                        else:
                            scenario_dicts.append(scenario)

                    # Create scenario tuples for Modal
                    scenario_tuples = [
                        (scenario_dict, agent_config.model_dump(), i) 
                        for i, scenario_dict in enumerate(scenario_dicts)
                    ]

                    # Execute using Modal with progress tracking and batch processing
                    with modal_app.run():
                        active_containers = min(len(scenario_tuples), 50)  # Max containers
                        
                        yield ProgressUpdate(
                            completed=0,
                            total=total_scenarios,
                            current_cost=0,
                            estimated_cost=estimated_cost,
                            active_containers=active_containers,
                            status_message=f"Running {active_containers} Modal containers with batch processing...",
                            batch_metrics=batch_processor.get_status()
                        )
                        
                        # Execute scenarios and track progress with batching
                        for i, result in enumerate(evaluate_single_scenario.map(scenario_tuples, return_exceptions=True)):
                            completed_count += 1
                            
                            # Calculate cost for this scenario
                            scenario_cost = 0.001  # Default cost
                            if not isinstance(result, Exception):
                                trajectory = result.get("trajectory", {})
                                token_usage = trajectory.get("token_usage", {})
                                scenario_cost = token_usage.get("total_cost", 0.001)
                            
                            total_cost += scenario_cost
                            
                            # Update active containers (simulate scaling down)
                            remaining = total_scenarios - completed_count
                            active_containers = min(remaining, 50) if remaining > 0 else 0
                            
                            # Add to batch processor instead of immediate database write
                            try:
                                # Transform the result to ensure 0-1 scale reliability scores for database
                                transformed_result = normalize_modal_result(result)
                                await batch_processor.add_outcome(
                                    simulation_id=simulation_id,
                                    scenario_result=transformed_result,
                                    modal_call_id=f"modal_call_{i}",
                                    sandbox_id=f"sandbox_{i}"
                                )
                            except Exception as e:
                                # Log error but don't stop execution - batch processor handles retries
                                logger.error(f"Failed to add outcome to batch processor: {e}")
                                self.console.print(f"[bold red]Batch Processing Error:[/bold red] {e}")
                                # For critical errors, we might want to fall back to individual processing
                                if "ResourceLimitError" in str(type(e)) or "ValueError" in str(type(e)):
                                    logger.critical(f"Critical batch processing error: {e}")
                                    raise
                            
                            # Yield real-time progress update with batch metrics
                            yield ProgressUpdate(
                                completed=completed_count,
                                total=total_scenarios,
                                current_cost=total_cost,
                                estimated_cost=estimated_cost,
                                active_containers=active_containers,
                                status_message=f"Completed {completed_count}/{total_scenarios} scenarios",
                                latest_result=result if not isinstance(result, Exception) else {"error": str(result)},
                                batch_metrics=batch_processor.get_status()
                            )
            else:
                # Fallback to simulation mode
                self.console.print("[bold yellow]Modal not available - using simulation mode[/bold yellow]")
                for i in range(total_scenarios):
                    await asyncio.sleep(0.01)  # Simulate processing time
                    completed_count += 1
                    
                    # Simulate result
                    mock_result = {
                        "scenario": {"id": f"scenario_{i}", "name": f"Simulated Scenario {i}"},
                        "trajectory": {"status": "success", "execution_time_seconds": 0.5},
                        "reliability_score": {"overall_score": 0.8}
                    }
                    
                    # Add to batch processor
                    await batch_processor.add_outcome(
                        simulation_id=simulation_id,
                        scenario_result=mock_result,
                        modal_call_id=f"sim_call_{i}",
                        sandbox_id=f"sim_sandbox_{i}"
                    )
                    
                    yield ProgressUpdate(
                        completed=completed_count,
                        total=total_scenarios,
                        current_cost=total_cost,
                        estimated_cost=estimated_cost,
                        active_containers=0,
                        status_message=f"Simulated {completed_count}/{total_scenarios} scenarios",
                        batch_metrics=batch_processor.get_status()
                    )

        except Exception as e:
            self.console.print(f"[bold red]Execution Error:[/bold red] {e}")
            raise

    async def _execute_scenarios_without_batching(
        self,
        agent_config: AgentConfig,
        scenarios: List[Scenario],
        estimated_cost: float,
        total_scenarios: int
    ) -> AsyncGenerator[ProgressUpdate, None]:
        """Execute scenarios without batch processing (fallback mode)."""
        completed_count = 0
        total_cost = 0.0
        
        self.console.print("[bold yellow]Executing without batch processing (fallback mode)[/bold yellow]")
        
        # Simple simulation for fallback
        for i in range(total_scenarios):
            await asyncio.sleep(0.01)
            completed_count += 1
            
            yield ProgressUpdate(
                completed=completed_count,
                total=total_scenarios,
                current_cost=total_cost,
                estimated_cost=estimated_cost,
                active_containers=0,
                status_message=f"Processed {completed_count}/{total_scenarios} scenarios (no batching)"
            )

    def _prepare_outcome_data(self, result, scenario, simulation_id: str, cost: float) -> dict:
        """Prepare outcome data for database insertion."""
        if isinstance(result, Exception):
            return {
                "simulation_id": simulation_id,
                "scenario_id": scenario.get("id", "unknown"),
                "execution_time": datetime.now(),
                "status": "error",
                "reliability_score": 0.0,
                "execution_time_ms": 0,
                "tokens_used": 0,
                "cost_usd": 0.0,
                "trajectory": {
                    "start_time": datetime.now().isoformat(),
                    "status": "error",
                    "error": str(result)
                },
                "error_code": str(result),
                "error_category": "system_error"
            }
        
        trajectory = result.get("trajectory", {})
        reliability_score = result.get("reliability_score", {})
        
        return {
            "simulation_id": simulation_id,
            "scenario_id": scenario.get("id", "unknown"),
            "execution_time": datetime.now(),
            "status": trajectory.get("status", "success"),
            "reliability_score": reliability_score.get("overall_score", 0.5) if isinstance(reliability_score, dict) else reliability_score,
            "execution_time_ms": int(trajectory.get("execution_time_seconds", 0) * 1000),
            "tokens_used": trajectory.get("token_usage", {}).get("total_tokens", 0),
            "cost_usd": cost,
            "trajectory": trajectory,
            "modal_call_id": result.get("modal_call_id"),
            "sandbox_id": result.get("sandbox_id")
        }

    def _estimate_execution_cost(self, scenario_count: int, model: str) -> float:
        """
        Estimates the total cost for running a number of scenarios with a given model.
        This is a simplified estimation and should be updated with more accurate pricing.
        """
        # This pricing map should be centralized
        MODELS_PRICING = {
            "openai/gpt-4.1": {"cost_per_1k_in": 0.00200, "cost_per_1k_out": 0.00800},
            "default": {"cost_per_1k_in": 0.001, "cost_per_1k_out": 0.002},
        }
        
        pricing = MODELS_PRICING.get(model, MODELS_PRICING["default"])
        # Assuming ~1k input tokens and 0.5k output tokens per scenario
        cost_per_scenario = (1.0 * pricing["cost_per_1k_in"]) + (0.5 * pricing["cost_per_1k_out"])
        
        return scenario_count * cost_per_scenario


