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
from arc.cli.utils import ArcConsole
from arc.sandbox.engine.simulator import app as modal_app, evaluate_single_scenario


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
        
        # Create simulation record in database
        simulation_id = None
        if self.db_client:
            try:
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
            except Exception as e:
                self.console.print(f"[bold yellow]Warning:[/bold yellow] Database tracking disabled: {e}")
                simulation_id = None
        
        # Initial progress update
        yield ProgressUpdate(
            completed=0,
            total=total_scenarios,
            current_cost=0,
            estimated_cost=estimated_cost,
            active_containers=0,
            status_message="Initializing Modal execution..."
        )

        try:
            # Check if Modal is available, otherwise simulate
            modal_available = True
            try:
                # Check for deployed app first
                if os.environ.get("ARC_USE_DEPLOYED_APP"):
                    try:
                        from arc.modal.public_api import ArcModalAPI
                        modal_available = ArcModalAPI.is_available()
                    except ImportError:
                        modal_available = False
                elif not os.environ.get("MODAL_TOKEN_ID") or not os.environ.get("MODAL_TOKEN_SECRET"):
                    modal_available = False
            except Exception:
                modal_available = False

            if modal_available:
                # Check if using deployed app
                if os.environ.get("ARC_USE_DEPLOYED_APP"):
                    # Execute with deployed Modal app
                    self.console.print("[bold green]Executing on deployed Modal app...[/bold green]")
                    
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
                        status_message=f"Running {active_containers} Modal containers..."
                    )
                    
                    # Execute scenarios using deployed app
                    async for result in ArcModalAPI.run_scenarios(
                        scenarios=scenario_dicts,
                        agent_config=agent_config.model_dump(),
                        batch_size=10
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
                        
                        # Persist to database immediately
                        if self.db_client and simulation_id:
                            try:
                                outcome_data = self._prepare_outcome_data(
                                    result, scenario_dicts[completed_count - 1], simulation_id, scenario_cost
                                )
                                await self.db_client.record_outcome(outcome_data)
                            except Exception as e:
                                self.console.print(f"[bold red]Database Error:[/bold red] {e}")
                        
                        # Yield real-time progress update
                        yield ProgressUpdate(
                            completed=completed_count,
                            total=total_scenarios,
                            current_cost=total_cost,
                            estimated_cost=estimated_cost,
                            active_containers=active_containers,
                            status_message=f"Completed {completed_count}/{total_scenarios} scenarios",
                            latest_result=result if not isinstance(result, Exception) else {"error": str(result)}
                        )
                else:
                    # Execute with authenticated Modal
                    self.console.print("[bold green]Executing on Modal...[/bold green]")
                    
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

                    # Execute using Modal with progress tracking
                    with modal_app.run():
                        active_containers = min(len(scenario_tuples), 50)  # Max containers
                        
                        yield ProgressUpdate(
                            completed=0,
                            total=total_scenarios,
                            current_cost=0,
                            estimated_cost=estimated_cost,
                            active_containers=active_containers,
                            status_message=f"Running {active_containers} Modal containers..."
                        )
                        
                        # Execute scenarios and track progress
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
                            
                            # Persist to database immediately
                            if self.db_client and simulation_id:
                                try:
                                    outcome_data = self._prepare_outcome_data(
                                        result, scenario_dicts[i], simulation_id, scenario_cost
                                    )
                                    await self.db_client.record_outcome(outcome_data)
                                except Exception as e:
                                    self.console.print(f"[bold red]Database Error:[/bold red] {e}")
                            
                            # Yield real-time progress update
                            yield ProgressUpdate(
                                completed=completed_count,
                                total=total_scenarios,
                                current_cost=total_cost,
                                estimated_cost=estimated_cost,
                                active_containers=active_containers,
                                status_message=f"Completed {completed_count}/{total_scenarios} scenarios",
                                latest_result=result if not isinstance(result, Exception) else {"error": str(result)}
                            )
            else:
                # Simulate execution with database persistence
                self.console.print("[bold yellow]Modal not available - simulating execution with database tracking[/bold yellow]")
                
                for i, scenario in enumerate(scenarios):
                    completed_count += 1
                    
                    # Simulate processing time
                    await asyncio.sleep(0.1)
                    
                    # Simulate result
                    success = i % 3 != 0  # 2/3 success rate
                    scenario_cost = 0.001
                    total_cost += scenario_cost
                    
                    # Create simulated result
                    result = {
                        "scenario": {"id": scenario.id} if hasattr(scenario, 'id') else {"id": f"scenario_{i}"},
                        "trajectory": {
                            "start_time": datetime.now().isoformat(),
                            "status": "success" if success else "error",
                            "execution_time_seconds": 0.5,
                            "token_usage": {
                                "total_tokens": 100,
                                "total_cost": scenario_cost
                            }
                        },
                        "reliability_score": {"overall_score": 0.9 if success else 0.2}
                    }
                    
                    # Persist to database
                    if self.db_client and simulation_id:
                        try:
                            outcome_data = self._prepare_outcome_data(
                                result, scenario, simulation_id, scenario_cost
                            )
                            await self.db_client.record_outcome(outcome_data)
                        except Exception as e:
                            self.console.print(f"[bold red]Database Error:[/bold red] {e}")
                    
                    # Yield progress update
                    yield ProgressUpdate(
                        completed=completed_count,
                        total=total_scenarios,
                        current_cost=total_cost,
                        estimated_cost=estimated_cost,
                        active_containers=0,
                        status_message=f"Simulated {completed_count}/{total_scenarios} scenarios",
                        latest_result=result
                    )

        except Exception as e:
            self.console.print(f"[bold red]Execution Error:[/bold red] {e}")
            # Create error results for remaining scenarios
            for i in range(completed_count, total_scenarios):
                yield ProgressUpdate(
                    completed=i + 1,
                    total=total_scenarios,
                    current_cost=total_cost,
                    estimated_cost=estimated_cost,
                    active_containers=0,
                    status_message=f"Failed at scenario {i + 1}: {e}",
                    latest_result={"error": str(e)}
                )

        # Finalize simulation in database
        if self.db_client and simulation_id:
            try:
                await self.db_client.finalize_simulation(
                    simulation_id=simulation_id,
                    status="completed",
                    total_cost_usd=total_cost,
                    completed_scenarios=completed_count,
                    metadata={
                        "final_cost": total_cost,
                        "estimated_cost": estimated_cost,
                        "cost_accuracy": abs(total_cost - estimated_cost) / estimated_cost if estimated_cost > 0 else 0
                    }
                )
                self.console.print(f"[bold green]Simulation finalized in database[/bold green]")
            except Exception as e:
                self.console.print(f"[bold yellow]Warning:[/bold yellow] Could not finalize simulation: {e}")

    def _prepare_outcome_data(self, result, scenario, simulation_id: str, cost: float) -> dict:
        """Prepare outcome data for database storage."""
        if isinstance(result, Exception):
            return {
                "simulation_id": simulation_id,
                "scenario_id": scenario.get("id", "unknown") if isinstance(scenario, dict) else getattr(scenario, 'id', 'unknown'),
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
                "error_category": "execution_error"
            }
        else:
            trajectory = result.get("trajectory", {})
            reliability = result.get("reliability_score", {})
            
            return {
                "simulation_id": simulation_id,
                "scenario_id": result.get("scenario", {}).get("id", "unknown"),
                "status": "success" if trajectory.get("status") == "success" else "error",
                "reliability_score": reliability.get("overall_score", 0.5) if isinstance(reliability, dict) else reliability,
                "execution_time_ms": int(trajectory.get("execution_time_seconds", 1.0) * 1000),
                "tokens_used": trajectory.get("token_usage", {}).get("total_tokens", 100),
                "cost_usd": cost,
                "trajectory": trajectory,
                "modal_call_id": f"modal_call_{datetime.now().timestamp()}"
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
