"""
Arc Database API for Modal Sandbox Integration

This module provides high-level API methods for the sandbox team to record
evaluation data. It's designed to align perfectly with the Modal sandbox
data structures and execution patterns.
"""

import asyncio
import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Any, List, Optional, Tuple, Dict
from enum import Enum

from sqlalchemy import text

from arc.database.client import ArcDBClient, DatabaseError, RetryableError
from arc.database.utils import convert_row_to_dict, normalize_reliability_score

logger = logging.getLogger(__name__)


class SimulationStatus(Enum):
    """Simulation status enum matching database constraints."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class OutcomeStatus(Enum):
    """Outcome status enum matching database constraints."""
    SUCCESS = "success"
    ERROR = "error"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"


class ArcAPI:
    """
    High-level API for Arc database operations.
    
    This API is designed specifically for Modal sandbox integration,
    providing methods that align with the sandbox's data structures
    and execution patterns.
    """
    
    def __init__(self, db_client: ArcDBClient):
        """
        Initialize the API with a database client.
        
        Args:
            db_client: Initialized ArcDBClient instance
        """
        self.db = db_client
        logger.info("ArcAPI initialized")
    
    # ==================== Simulation Lifecycle ====================
    
    async def start_simulation(
        self,
        config_version_id: str,
        scenarios: List[dict[str, Any]],
        simulation_name: Optional[str] = None,
        modal_app_id: Optional[str] = None,
        modal_environment: str = "production",
        sandbox_instances: int = 1,
        metadata: Optional[dict[str, Any]] = None
    ) -> dict[str, Any]:
        """
        Start a new simulation run.
        
        This method creates a simulation record and prepares for scenario execution.
        It's designed to be called at the beginning of a Modal function execution.
        
        Args:
            config_version_id: ID of the configuration version to use
            scenarios: List of scenario dictionaries from the sandbox
            simulation_name: Optional human-readable name
            modal_app_id: Modal application ID (auto-detected if not provided)
            modal_environment: Modal environment (dev/staging/production)
            sandbox_instances: Number of parallel sandbox instances
            metadata: Additional metadata to store
            
        Returns:
            Dict containing:
                - simulation_id: UUID of the created simulation
                - scenario_count: Number of scenarios to run
                - status: Current status (will be 'pending')
                - created_at: Timestamp of creation
        """
        try:
            # Convert scenarios to dict format if needed and extract IDs
            scenario_dicts = []
            scenario_ids = []
            
            for i, scenario in enumerate(scenarios):
                if hasattr(scenario, 'to_dict'):
                    scenario_dict = scenario.to_dict()
                elif isinstance(scenario, dict):
                    scenario_dict = scenario
                else:
                    scenario_dict = {"id": f"scenario_{i}", "task": str(scenario)}
                
                scenario_dicts.append(scenario_dict)
                scenario_ids.append(scenario_dict.get("id", f"scenario_{i}"))
            
            # Ensure scenarios exist before creating simulation
            for i, scenario_dict in enumerate(scenario_dicts):
                await self.db.ensure_scenario_exists(
                    scenario_dict.get("id", f"scenario_{i}"),
                    scenario_dict
                )
            
            # Create simulation record
            simulation_id = await self.db.create_simulation(
                config_version_id=config_version_id,
                scenario_set=scenario_ids,
                simulation_name=simulation_name,
                modal_app_id=modal_app_id
            )
            
            # Update with additional fields
            async with self.db.engine.begin() as conn:
                await conn.execute(text("""
                    UPDATE simulations 
                    SET 
                        modal_environment = :modal_environment,
                        sandbox_instances = :sandbox_instances,
                        metadata = :metadata,
                        status = 'running',
                        started_at = NOW()
                    WHERE simulation_id = :simulation_id
                """), {
                    "simulation_id": simulation_id,
                    "modal_environment": modal_environment,
                    "sandbox_instances": sandbox_instances,
                    "metadata": json.dumps(metadata or {})
                })
                
                # Create simulation-scenario mappings
                for i, scenario_id in enumerate(scenario_ids):
                    await conn.execute(text("""
                        INSERT INTO simulations_scenarios (
                            simulation_id, scenario_id, execution_order, status
                        ) VALUES (:simulation_id, :scenario_id, :order, 'pending')
                    """), {
                        "simulation_id": simulation_id,
                        "scenario_id": scenario_id,
                        "order": i + 1
                    })
            
            logger.info(f"Started simulation {simulation_id} with {len(scenarios)} scenarios")
            
            return {
                "simulation_id": simulation_id,
                "scenario_count": len(scenarios),
                "status": SimulationStatus.RUNNING.value,
                "created_at": datetime.now(timezone.utc).isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to start simulation: {e}")
            raise DatabaseError(f"Failed to start simulation: {e}") from e
    
    async def record_scenario_outcome(
        self,
        simulation_id: str,
        scenario_result: dict[str, Any],
        modal_call_id: Optional[str] = None,
        sandbox_id: Optional[str] = None
    ) -> str:
        """
        Record the outcome of a single scenario execution.
        
        This method is designed to be called after each scenario completes
        in the Modal sandbox. It extracts all relevant data from the
        scenario result structure.
        
        Args:
            simulation_id: ID of the parent simulation
            scenario_result: Complete result dict from evaluate_single_scenario
            modal_call_id: Modal function call ID
            sandbox_id: Sandbox instance ID
            
        Returns:
            outcome_id: UUID of the recorded outcome
        """
        try:
            # Extract data from scenario result structure
            scenario = scenario_result.get("scenario", {})
            trajectory = scenario_result.get("trajectory", {})
            
            # Handle reliability_score that could be dict or float
            rs = scenario_result.get("reliability_score", {})
            if isinstance(rs, dict):
                overall_score = rs.get("overall_score", 0.0)
                dimension_scores = rs.get("dimension_scores", {})
                grade = rs.get("grade", "N/A")
            else:  # float or other type
                overall_score = float(rs) if rs else 0.0
                dimension_scores = {}
                grade = "N/A"
            
            detailed_trajectory = scenario_result.get("detailed_trajectory", {})
            
            # Prepare outcome data
            outcome_data = {
                "simulation_id": simulation_id,
                "scenario_id": scenario.get("id", "unknown"),
                "scenario_data": {
                    "name": scenario.get("name", f"Scenario {scenario.get('id', 'unknown')}"),
                    "task_prompt": trajectory.get("task_prompt", scenario.get("task_prompt", "Auto-generated scenario"))
                },
                "execution_time": datetime.now(timezone.utc),
                "status": trajectory.get("status", "error"),
                "reliability_score": overall_score,
                "execution_time_ms": int(trajectory.get("execution_time_seconds", 0) * 1000),
                "tokens_used": trajectory.get("token_usage", {}).get("total_tokens", 0),
                "cost_usd": trajectory.get("token_usage", {}).get("total_cost", 0.0),
                "trajectory": {
                    "start_time": trajectory.get("start_time", datetime.now(timezone.utc).isoformat()),
                    "status": trajectory.get("status", "error"),
                    "task_prompt": trajectory.get("task_prompt"),
                    "final_response": trajectory.get("final_response"),
                    "full_trajectory": trajectory.get("full_trajectory", []),
                    "detailed_trajectory": detailed_trajectory,
                    "reliability_dimensions": dimension_scores
                },
                "modal_call_id": modal_call_id,
                "sandbox_id": sandbox_id,
                "error_code": trajectory.get("error") if trajectory.get("status") == "error" else None,
                "error_category": self._categorize_error(trajectory.get("error")),
                "metrics": {
                    "prompt_tokens": trajectory.get("token_usage", {}).get("prompt_tokens", 0),
                    "completion_tokens": trajectory.get("token_usage", {}).get("completion_tokens", 0),
                    "trajectory_event_count": trajectory.get("trajectory_event_count", 0),
                    "reliability_grade": grade,
                    **dimension_scores
                }
            }
            
            # Record the outcome
            outcome_id = await self.db.record_outcome(outcome_data)
            
            # Update simulation-scenario status
            async with self.db.engine.begin() as conn:
                await conn.execute(text("""
                    UPDATE simulations_scenarios
                    SET 
                        status = :status,
                        modal_call_id = :modal_call_id,
                        started_at = COALESCE(started_at, NOW()),
                        completed_at = NOW()
                    WHERE simulation_id = :simulation_id AND scenario_id = :scenario_id
                """), {
                    "simulation_id": simulation_id,
                    "scenario_id": scenario.get("id", "unknown"),
                    "status": "completed" if trajectory.get("status") == "success" else "failed",
                    "modal_call_id": modal_call_id
                })
                
                # Update simulation completed count
                await conn.execute(text("""
                    UPDATE simulations
                    SET completed_scenarios = completed_scenarios + 1
                    WHERE simulation_id = :simulation_id
                """), {"simulation_id": simulation_id})
            
            logger.debug(f"Recorded outcome {outcome_id} for scenario {scenario.get('id')}")
            return outcome_id
            
        except Exception as e:
            logger.error(f"Failed to record scenario outcome: {e}")
            raise DatabaseError(f"Failed to record outcome: {e}") from e
    
    async def record_batch_outcomes(
        self,
        simulation_id: str,
        scenario_results: List[dict[str, Any]],
        modal_call_ids: Optional[List[str]] = None
    ) -> List[str]:
        """
        Record multiple scenario outcomes in a single batch operation.
        
        This is optimized for high-throughput recording when running
        many scenarios in parallel with Modal's .map() function.
        
        Args:
            simulation_id: ID of the parent simulation
            scenario_results: List of complete result dicts from evaluate_single_scenario
            modal_call_ids: Optional list of Modal call IDs (one per scenario)
            
        Returns:
            List of outcome_ids for the recorded outcomes
        """
        try:
            outcomes = []
            
            for i, result in enumerate(scenario_results):
                scenario = result.get("scenario", {})
                trajectory = result.get("trajectory", {})
                
                # Use centralized reliability score normalization
                rs = result.get("reliability_score", {})
                
                # Debug logging to understand what we're getting
                logger.debug(f"Processing result {i}: reliability_score type={type(rs)}, value={rs}")
                
                # Extract metadata for trajectory
                if isinstance(rs, dict):
                    dimension_scores = rs.get("dimension_scores", {})
                    grade = rs.get("grade", "N/A")
                else:
                    dimension_scores = {}
                    grade = "N/A"
                
                # Use centralized normalization (handles all cases and defaults)
                normalized_score = normalize_reliability_score(rs)
                
                # Additional debug logging
                logger.debug(f"Normalized reliability score for scenario {i}: {normalized_score}")
                
                detailed_trajectory = result.get("detailed_trajectory", {})
                
                outcomes.append({
                    "simulation_id": simulation_id,
                    "scenario_id": scenario.get("id", f"scenario_{i}"),
                    "scenario_data": {
                        "name": scenario.get("name", f"Scenario {scenario.get('id', f'scenario_{i}')}"),
                        "task_prompt": trajectory.get("task_prompt", scenario.get("task_prompt", "Auto-generated scenario"))
                    },
                    "execution_time": datetime.now(timezone.utc),
                    "status": trajectory.get("status", "error"),
                    "reliability_score": normalized_score,
                    "execution_time_ms": int(trajectory.get("execution_time_seconds", 0) * 1000),
                    "tokens_used": trajectory.get("token_usage", {}).get("total_tokens", 0),
                    "cost_usd": trajectory.get("token_usage", {}).get("total_cost", 0.0),
                    "trajectory": {
                        "start_time": trajectory.get("start_time", datetime.now(timezone.utc).isoformat()),
                        "status": trajectory.get("status", "error"),
                        "task_prompt": trajectory.get("task_prompt"),
                        "final_response": trajectory.get("final_response"),
                        "full_trajectory": trajectory.get("full_trajectory", []),
                        "detailed_trajectory": detailed_trajectory,
                        "reliability_dimensions": dimension_scores
                    },
                    "modal_call_id": modal_call_ids[i] if modal_call_ids else None,
                    "error_code": trajectory.get("error") if trajectory.get("status") == "error" else None,
                    "error_category": self._categorize_error(trajectory.get("error")),
                    "metrics": {
                        "prompt_tokens": trajectory.get("token_usage", {}).get("prompt_tokens", 0),
                        "completion_tokens": trajectory.get("token_usage", {}).get("completion_tokens", 0),
                        "trajectory_event_count": trajectory.get("trajectory_event_count", 0),
                        "reliability_grade": grade,
                        **dimension_scores
                    }
                })
            
            # Batch insert
            outcome_ids = await self.db.record_outcomes_batch(outcomes)
            
            # Update simulation-scenario statuses and count in a transaction
            async with self.db.engine.begin() as conn:
                # First, check which scenarios are already completed to avoid double-counting
                scenario_ids = [scenario.get("id", f"scenario_{i}") for i, scenario in enumerate([result.get("scenario", {}) for result in scenario_results])]
                
                logger.debug(f"Checking status for scenarios: {scenario_ids}")
                
                # Check current status of scenarios in this simulation
                # Use IN clause instead of ANY for better PostgreSQL compatibility
                placeholders = ','.join([f':scenario_id_{i}' for i in range(len(scenario_ids))])
                scenario_params = {f'scenario_id_{i}': scenario_id for i, scenario_id in enumerate(scenario_ids)}
                
                check_result = await conn.execute(text(f"""
                    SELECT scenario_id, status FROM simulations_scenarios
                    WHERE simulation_id = :simulation_id AND scenario_id IN ({placeholders})
                """), {
                    "simulation_id": simulation_id,
                    **scenario_params
                })
                
                existing_statuses = {row[0]: row[1] for row in check_result.fetchall()}
                logger.debug(f"Existing scenario statuses: {existing_statuses}")
                
                # Also check current simulation completion count
                sim_check = await conn.execute(text("""
                    SELECT completed_scenarios, total_scenarios FROM simulations
                    WHERE simulation_id = :simulation_id
                """), {"simulation_id": simulation_id})
                
                sim_row = sim_check.fetchone()
                if sim_row:
                    current_completed = sim_row[0] or 0
                    total_scenarios = sim_row[1] or 0
                    logger.debug(f"Current simulation state: {current_completed}/{total_scenarios} completed")
                else:
                    current_completed = 0
                    total_scenarios = 0
                    logger.warning(f"Could not find simulation {simulation_id}")
                
                scenarios_to_update = []
                new_completions = 0
                
                # Update scenario statuses and count only new completions
                for i, result in enumerate(scenario_results):
                    scenario = result.get("scenario", {})
                    trajectory = result.get("trajectory", {})
                    scenario_id = scenario.get("id", f"scenario_{i}")
                    new_status = "completed" if trajectory.get("status") == "success" else "failed"
                    
                    # Only count as new completion if status is changing from pending/running
                    current_status = existing_statuses.get(scenario_id, "pending")
                    if current_status in ["pending", "running"] and new_status in ["completed", "failed"]:
                        new_completions += 1
                        logger.debug(f"Scenario {scenario_id}: {current_status} -> {new_status} (new completion)")
                    else:
                        logger.debug(f"Scenario {scenario_id}: {current_status} -> {new_status} (already processed)")
                    
                    scenarios_to_update.append({
                        "simulation_id": simulation_id,
                        "scenario_id": scenario_id,
                        "status": new_status,
                        "modal_call_id": modal_call_ids[i] if modal_call_ids else None
                    })
                
                logger.debug(f"New completions to add: {new_completions}")
                
                # Check if adding new completions would violate constraint
                projected_total = current_completed + new_completions
                if projected_total > total_scenarios:
                    logger.error(f"Constraint violation: {current_completed} + {new_completions} = {projected_total} > {total_scenarios}")
                    # Don't throw error, just skip the update to avoid retry loop
                    new_completions = 0
                    logger.warning("Skipping completion count update to avoid constraint violation")
                
                # Update scenario statuses
                for scenario_update in scenarios_to_update:
                    await conn.execute(text("""
                        UPDATE simulations_scenarios
                        SET 
                            status = :status,
                            modal_call_id = :modal_call_id,
                            started_at = COALESCE(started_at, NOW()),
                            completed_at = NOW()
                        WHERE simulation_id = :simulation_id AND scenario_id = :scenario_id
                    """), scenario_update)
                
                # Only update simulation completed count if there are actual new completions
                if new_completions > 0:
                    await conn.execute(text("""
                        UPDATE simulations
                        SET completed_scenarios = completed_scenarios + :count
                        WHERE simulation_id = :simulation_id
                    """), {
                        "simulation_id": simulation_id,
                        "count": new_completions
                    })
                    logger.info(f"Updated simulation {simulation_id} with {new_completions} new completions ({current_completed} -> {current_completed + new_completions})")
                else:
                    logger.info(f"No new completions for simulation {simulation_id} - scenarios already processed")
            
            logger.info(f"Recorded {len(outcome_ids)} outcomes in batch for simulation {simulation_id}")
            return outcome_ids
            
        except Exception as e:
            logger.error(f"Failed to record batch outcomes: {e}")
            raise DatabaseError(f"Failed to record batch outcomes: {e}") from e

    async def record_modal_result(
        self,
        simulation_id: str,
        scenario: Any,
        modal_result: dict[str, Any],
        call_index: int = 0
    ) -> str:
        """
        Record a Modal execution result for a scenario.
        
        This is the production function that handles the raw Modal result
        and transforms it into the proper database format automatically.
        
        Args:
            simulation_id: ID of the simulation
            scenario: Scenario object or dict
            modal_result: Raw result from _execute_with_modal
            call_index: Index for generating call IDs
            
        Returns:
            outcome_id: UUID of the recorded outcome
        """
        try:
            # Extract scenario ID from dict or object
            if isinstance(scenario, dict):
                scenario_id = scenario.get("id", f"scenario_{call_index}")
            elif hasattr(scenario, 'id'):
                scenario_id = scenario.id
            else:
                scenario_id = f"scenario_{call_index}"
            
            # Prepare trajectory with required fields
            execution_time = datetime.now(timezone.utc)
            status = "success" if modal_result.get("success", False) else "error"
            
            # Ensure trajectory has required fields
            trajectory = modal_result.copy() if isinstance(modal_result, dict) else {}
            if "start_time" not in trajectory:
                trajectory["start_time"] = execution_time.isoformat()
            if "status" not in trajectory:
                trajectory["status"] = status
            
            # Transform Modal result to outcome format
            outcome_data = {
                "simulation_id": simulation_id,
                "scenario_id": scenario_id,
                "execution_time": execution_time,
                "status": status,
                "reliability_score": modal_result.get("reliability_score", 0.5),
                "execution_time_ms": int(modal_result.get("execution_time", 0) * 1000),
                "tokens_used": modal_result.get("tokens_used", 100),
                "cost_usd": modal_result.get("cost", 0.001),
                "trajectory": trajectory,
                "modal_call_id": f"modal_call_{call_index}",
                "sandbox_id": f"sandbox_{call_index}"
            }
            
            # Record using existing function
            outcome_id = await self.db.record_outcome(outcome_data)
            
            logger.debug(f"Recorded Modal result for scenario {scenario_id}: {outcome_id}")
            return outcome_id
            
        except Exception as e:
            logger.error(f"Failed to record Modal result: {e}")
            raise DatabaseError(f"Failed to record Modal result: {e}") from e

    async def complete_simulation_from_results(
        self,
        simulation_id: str,
        scenarios: List[Any],
        modal_results: List[dict[str, Any]],
        execution_time: float,
        total_cost: float
    ) -> dict[str, Any]:
        """
        Complete a simulation using raw Modal results.
        
        This is the production function that handles completion automatically
        based on the Modal execution results.
        
        Args:
            simulation_id: ID of the simulation to complete
            scenarios: List of scenario objects
            modal_results: Raw results from _execute_with_modal
            execution_time: Total execution time in seconds
            total_cost: Total cost in USD
            
        Returns:
            Dict with completion status
        """
        try:
            # Calculate aggregate metrics from Modal results
            successful_runs = sum(1 for r in modal_results if r.get("success", False))
            
            # Calculate average reliability only from results with explicit scores
            results_with_scores = [r for r in modal_results if "reliability_score" in r]
            if results_with_scores:
                avg_reliability = sum(r["reliability_score"] for r in results_with_scores) / len(results_with_scores)
            else:
                # Default to 0.5 if no results have reliability scores
                avg_reliability = 0.5 if modal_results else 0
            
            # Create suite result automatically
            suite_result = {
                "scenarios_run": len(scenarios),
                "successful_runs": successful_runs,
                "average_reliability_score": avg_reliability,
                "total_cost_usd": total_cost,
                "parallel_execution_time": execution_time,
                "results": modal_results
            }
            
            # Use existing complete_simulation function
            return await self.complete_simulation(simulation_id, suite_result)
            
        except Exception as e:
            logger.error(f"Failed to complete simulation from results: {e}")
            raise DatabaseError(f"Failed to complete simulation from results: {e}") from e
    
    async def complete_simulation(
        self,
        simulation_id: str,
        suite_result: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Mark a simulation as completed and record aggregate metrics.
        
        This should be called after all scenarios have been evaluated,
        typically at the end of run_evaluation_suite_parallel.
        
        Args:
            simulation_id: ID of the simulation to complete
            suite_result: Complete result dict from run_evaluation_suite_parallel
            
        Returns:
            Dict with completion status and summary
        """
        try:
            # Extract aggregate metrics
            overall_score = suite_result.get("average_reliability_score", 0.0)
            total_cost = suite_result.get("total_cost_usd", 0.0)
            execution_time_ms = int(suite_result.get("parallel_execution_time", 0) * 1000)
            
            # Update simulation record
            async with self.db.engine.begin() as conn:
                await conn.execute(text("""
                    UPDATE simulations
                    SET 
                        status = 'completed',
                        completed_at = NOW(),
                        overall_score = :overall_score,
                        total_cost_usd = :total_cost,
                        execution_time_ms = :execution_time_ms,
                        metadata = metadata || :additional_metadata
                    WHERE simulation_id = :simulation_id
                """), {
                    "simulation_id": simulation_id,
                    "overall_score": overall_score,
                    "total_cost": total_cost,
                    "execution_time_ms": execution_time_ms,
                    "additional_metadata": json.dumps({
                        "successful_runs": suite_result.get("successful_runs", 0),
                        "total_tokens_used": suite_result.get("total_tokens_used", 0),
                        "average_execution_time": suite_result.get("average_execution_time", 0),
                        "average_dimension_scores": suite_result.get("average_dimension_scores", {}),
                        "speedup_factor": suite_result.get("speedup_factor", 1.0)
                    })
                })
            
            logger.info(f"Completed simulation {simulation_id} with score {overall_score:.2f}")
            
            return {
                "simulation_id": simulation_id,
                "status": SimulationStatus.COMPLETED.value,
                "overall_score": overall_score,
                "total_cost_usd": total_cost,
                "execution_time_seconds": suite_result.get("parallel_execution_time", 0),
                "completed_at": datetime.now(timezone.utc).isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to complete simulation: {e}")
            raise DatabaseError(f"Failed to complete simulation: {e}") from e
    
    # ==================== Query Methods ====================
    
    async def get_simulation_status(self, simulation_id: str) -> dict[str, Any]:
        """
        Get current status and progress of a simulation.
        
        Args:
            simulation_id: ID of the simulation to check
            
        Returns:
            Dict with simulation status and progress information
        """
        try:
            async with self.db.engine.begin() as conn:
                result = await conn.execute(text("""
                    SELECT 
                        s.simulation_id,
                        s.status,
                        s.total_scenarios,
                        s.completed_scenarios,
                        s.overall_score,
                        s.total_cost_usd,
                        s.execution_time_ms,
                        s.created_at,
                        s.started_at,
                        s.completed_at,
                        s.modal_app_id,
                        s.metadata,
                        COUNT(CASE WHEN o.status = 'success' THEN 1 END) as successful_outcomes,
                        COUNT(CASE WHEN o.status = 'error' THEN 1 END) as failed_outcomes
                    FROM simulations s
                    LEFT JOIN outcomes o ON s.simulation_id = o.simulation_id
                    WHERE s.simulation_id = :simulation_id
                    GROUP BY s.simulation_id
                """), {"simulation_id": simulation_id})
                
                row = result.first()
                if not row:
                    raise ValueError(f"Simulation {simulation_id} not found")
                
                data = convert_row_to_dict(row)
                
                # Calculate progress percentage
                progress = (data["completed_scenarios"] / data["total_scenarios"] * 100) if data["total_scenarios"] > 0 else 0
                
                return {
                    "simulation_id": data["simulation_id"],
                    "status": data["status"],
                    "progress_percentage": progress,
                    "total_scenarios": data["total_scenarios"],
                    "completed_scenarios": data["completed_scenarios"],
                    "successful_outcomes": data["successful_outcomes"],
                    "failed_outcomes": data["failed_outcomes"],
                    "overall_score": data["overall_score"],
                    "total_cost_usd": data["total_cost_usd"],
                    "execution_time_ms": data["execution_time_ms"],
                    "modal_app_id": data["modal_app_id"],
                    "metadata": data["metadata"],
                    "created_at": data["created_at"].isoformat() if data["created_at"] else None,
                    "started_at": data["started_at"].isoformat() if data["started_at"] else None,
                    "completed_at": data["completed_at"].isoformat() if data["completed_at"] else None
                }
                
        except Exception as e:
            logger.error(f"Failed to get simulation status: {e}")
            raise DatabaseError(f"Failed to get simulation status: {e}") from e
    
    async def get_scenario_outcomes(
        self,
        simulation_id: str,
        scenario_id: Optional[str] = None,
        status_filter: Optional[str] = None
    ) -> List[dict[str, Any]]:
        """
        Get outcomes for scenarios in a simulation.
        
        Args:
            simulation_id: ID of the simulation
            scenario_id: Optional specific scenario ID to filter by
            status_filter: Optional status to filter by (success/error/timeout)
            
        Returns:
            List of outcome records with trajectory data
        """
        try:
            query = """
                SELECT 
                    o.outcome_id,
                    o.scenario_id,
                    o.execution_time,
                    o.status,
                    o.reliability_score,
                    o.execution_time_ms,
                    o.tokens_used,
                    o.cost_usd,
                    o.trajectory,
                    o.modal_call_id,
                    o.error_code,
                    o.error_category,
                    o.metrics,
                    s.name as scenario_name
                FROM outcomes o
                LEFT JOIN scenarios s ON o.scenario_id = s.scenario_id
                WHERE o.simulation_id = :simulation_id
            """
            
            params = {"simulation_id": simulation_id}
            
            if scenario_id:
                query += " AND o.scenario_id = :scenario_id"
                params["scenario_id"] = scenario_id
            
            if status_filter:
                query += " AND o.status = :status"
                params["status"] = status_filter
            
            query += " ORDER BY o.execution_time DESC"
            
            async with self.db.engine.begin() as conn:
                result = await conn.execute(text(query), params)
                
                outcomes = []
                for row in result:
                    data = convert_row_to_dict(row)
                    
                    # Parse JSON fields
                    data["trajectory"] = json.loads(data["trajectory"]) if data["trajectory"] else {}
                    data["metrics"] = json.loads(data["metrics"]) if data["metrics"] else {}
                    data["execution_time"] = data["execution_time"].isoformat() if data["execution_time"] else None
                    outcomes.append(data)
                
                return outcomes
                
        except Exception as e:
            logger.error(f"Failed to get scenario outcomes: {e}")
            raise DatabaseError(f"Failed to get scenario outcomes: {e}") from e
    
    # ==================== Helper Methods ====================
    
    def _categorize_error(self, error_message: Optional[str]) -> Optional[str]:
        """
        Categorize errors based on error message patterns.
        
        Valid categories from database constraint:
        'timeout', 'tool_error', 'model_error', 'validation_error', 'system_error', 'sandbox_error'
        """
        if not error_message:
            return None
        
        error_lower = error_message.lower()
        
        if "timeout" in error_lower:
            return "timeout"
        elif "tool" in error_lower or "function" in error_lower:
            return "tool_error"
        elif "model" in error_lower or "openai" in error_lower or "llm" in error_lower:
            return "model_error"
        elif "validation" in error_lower or "parsing" in error_lower or "json" in error_lower or "schema" in error_lower:
            return "validation_error"
        elif "modal" in error_lower or "sandbox" in error_lower or "container" in error_lower:
            return "sandbox_error"
        else:
            return "system_error"  # Default fallback for any other errors

    async def record_batch_tool_usage(
        self,
        tool_usage_records: List[Dict[str, Any]]
    ) -> List[str]:
        """
        Record multiple tool usage records in batch.
        
        Args:
            tool_usage_records: List of tool usage data dictionaries
            
        Returns:
            List of usage_id strings for the recorded entries
        """
        if not tool_usage_records:
            return []
        
        try:
            usage_ids = [str(uuid.uuid4()) for _ in tool_usage_records]
            batch_data = []
            
            for i, record in enumerate(tool_usage_records):
                batch_data.append({
                    "usage_id": usage_ids[i],
                    "outcome_id": record["outcome_id"],
                    "execution_time": record.get("execution_time", datetime.now(timezone.utc)),
                    "tool_name": record["tool_name"],
                    "call_count": record["call_count"],
                    "avg_duration_ms": record["avg_duration_ms"],
                    "success_rate": record["success_rate"],
                    "tool_sequence": json.dumps(record.get("tool_sequence", {})),
                    "total_cost_usd": record.get("total_cost_usd", 0.0),
                    "error_types": record.get("error_types", []),
                    "modal_function_calls": json.dumps(record.get("modal_function_calls", {})),
                    "sandbox_tool_logs": record.get("sandbox_tool_logs")
                })
            
            async with self.db.engine.begin() as conn:
                await conn.execute(text("""
                    INSERT INTO tool_usage (
                        usage_id, outcome_id, execution_time, tool_name,
                        call_count, avg_duration_ms, success_rate, tool_sequence,
                        total_cost_usd, error_types, modal_function_calls, sandbox_tool_logs
                    ) VALUES (
                        :usage_id, :outcome_id, :execution_time, :tool_name,
                        :call_count, :avg_duration_ms, :success_rate, :tool_sequence,
                        :total_cost_usd, :error_types, :modal_function_calls, :sandbox_tool_logs
                    )
                """), batch_data)
            
            logger.info(f"Recorded {len(tool_usage_records)} tool usage records in batch")
            return usage_ids
            
        except Exception as e:
            logger.error(f"Failed to record tool usage batch: {e}")
            raise DatabaseError(f"Failed to record tool usage batch: {e}") from e
    
    async def record_batch_failures(
        self,
        failure_records: List[Dict[str, Any]]
    ) -> List[str]:
        """
        Record multiple failure pattern records in batch.
        
        Args:
            failure_records: List of failure pattern data dictionaries
            
        Returns:
            List of pattern_id strings for the recorded entries
        """
        if not failure_records:
            return []
        
        try:
            pattern_ids = [str(uuid.uuid4()) for _ in failure_records]
            batch_data = []
            
            for i, record in enumerate(failure_records):
                batch_data.append({
                    "pattern_id": pattern_ids[i],
                    "outcome_id": record["outcome_id"],
                    "execution_time": record.get("execution_time", datetime.now(timezone.utc)),
                    "failure_type": record["failure_type"],
                    "failure_category": record["failure_category"],
                    "error_message": record.get("error_message"),
                    "pattern_cluster_id": record.get("pattern_cluster_id"),
                    "cluster_confidence": record.get("cluster_confidence"),
                    "similar_failures_count": record.get("similar_failures_count", 0),
                    "recovery_attempted": record.get("recovery_attempted", False),
                    "recovery_successful": record.get("recovery_successful", False),
                    "resolution_steps": record.get("resolution_steps"),
                    "llm_analysis": json.dumps(record.get("llm_analysis", {}))
                })
            
            async with self.db.engine.begin() as conn:
                await conn.execute(text("""
                    INSERT INTO failure_patterns (
                        pattern_id, outcome_id, execution_time, failure_type,
                        failure_category, error_message, pattern_cluster_id, cluster_confidence,
                        similar_failures_count, recovery_attempted, recovery_successful,
                        resolution_steps, llm_analysis
                    ) VALUES (
                        :pattern_id, :outcome_id, :execution_time, :failure_type,
                        :failure_category, :error_message, :pattern_cluster_id, :cluster_confidence,
                        :similar_failures_count, :recovery_attempted, :recovery_successful,
                        :resolution_steps, :llm_analysis
                    )
                """), batch_data)
            
            logger.info(f"Recorded {len(failure_records)} failure pattern records in batch")
            return pattern_ids
            
        except Exception as e:
            logger.error(f"Failed to record failure patterns batch: {e}")
            raise DatabaseError(f"Failed to record failure patterns batch: {e}") from e


# Convenience function for creating API instance
async def create_arc_api(connection_string: Optional[str] = None) -> ArcAPI:
    """
    Create and initialize an ArcAPI instance.
    
    Args:
        connection_string: Optional database connection string
        
    Returns:
        Initialized ArcAPI instance
    """
    db_client = ArcDBClient(connection_string)
    await db_client.initialize()
    return ArcAPI(db_client) 