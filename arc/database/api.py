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
from typing import Dict, Any, List, Optional, Tuple
from enum import Enum

from arc.database.client import ArcDBClient, DatabaseError, RetryableError
from sqlalchemy import text

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
        scenarios: List[Dict[str, Any]],
        simulation_name: Optional[str] = None,
        modal_app_id: Optional[str] = None,
        modal_environment: str = "production",
        sandbox_instances: int = 1,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
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
            # Extract scenario IDs from scenario objects
            scenario_ids = [s.get("id", f"scenario_{i}") for i, s in enumerate(scenarios)]
            
            # Ensure scenarios exist before creating simulation
            for scenario in scenarios:
                await self.db.ensure_scenario_exists(
                    scenario.get("id", f"scenario_{scenarios.index(scenario)}"),
                    scenario
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
        scenario_result: Dict[str, Any],
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
            reliability_score = scenario_result.get("reliability_score", {})
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
                "reliability_score": reliability_score.get("overall_score", 0.0),
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
                    "reliability_dimensions": reliability_score.get("dimension_scores", {})
                },
                "modal_call_id": modal_call_id,
                "sandbox_id": sandbox_id,
                "error_code": trajectory.get("error") if trajectory.get("status") == "error" else None,
                "error_category": self._categorize_error(trajectory.get("error")),
                "metrics": {
                    "prompt_tokens": trajectory.get("token_usage", {}).get("prompt_tokens", 0),
                    "completion_tokens": trajectory.get("token_usage", {}).get("completion_tokens", 0),
                    "trajectory_event_count": trajectory.get("trajectory_event_count", 0),
                    "reliability_grade": reliability_score.get("grade", "N/A"),
                    **reliability_score.get("dimension_scores", {})
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
        scenario_results: List[Dict[str, Any]],
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
                reliability_score = result.get("reliability_score", {})
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
                    "reliability_score": reliability_score.get("overall_score", 0.0),
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
                        "reliability_dimensions": reliability_score.get("dimension_scores", {})
                    },
                    "modal_call_id": modal_call_ids[i] if modal_call_ids else None,
                    "error_code": trajectory.get("error") if trajectory.get("status") == "error" else None,
                    "error_category": self._categorize_error(trajectory.get("error")),
                    "metrics": {
                        "prompt_tokens": trajectory.get("token_usage", {}).get("prompt_tokens", 0),
                        "completion_tokens": trajectory.get("token_usage", {}).get("completion_tokens", 0),
                        "trajectory_event_count": trajectory.get("trajectory_event_count", 0),
                        "reliability_grade": reliability_score.get("grade", "N/A"),
                        **reliability_score.get("dimension_scores", {})
                    }
                })
            
            # Batch insert
            outcome_ids = await self.db.record_outcomes_batch(outcomes)
            
            # Update simulation-scenario statuses in batch
            async with self.db.engine.begin() as conn:
                for i, result in enumerate(scenario_results):
                    scenario = result.get("scenario", {})
                    trajectory = result.get("trajectory", {})
                    
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
                        "scenario_id": scenario.get("id", f"scenario_{i}"),
                        "status": "completed" if trajectory.get("status") == "success" else "failed",
                        "modal_call_id": modal_call_ids[i] if modal_call_ids else None
                    })
                
                # Update simulation completed count
                await conn.execute(text("""
                    UPDATE simulations
                    SET completed_scenarios = completed_scenarios + :count
                    WHERE simulation_id = :simulation_id
                """), {
                    "simulation_id": simulation_id,
                    "count": len(scenario_results)
                })
            
            logger.info(f"Recorded {len(outcome_ids)} outcomes in batch for simulation {simulation_id}")
            return outcome_ids
            
        except Exception as e:
            logger.error(f"Failed to record batch outcomes: {e}")
            raise DatabaseError(f"Failed to record batch outcomes: {e}") from e
    
    async def complete_simulation(
        self,
        simulation_id: str,
        suite_result: Dict[str, Any]
    ) -> Dict[str, Any]:
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
    
    async def get_simulation_status(self, simulation_id: str) -> Dict[str, Any]:
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
                
                data = dict(row._mapping)
                
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
    ) -> List[Dict[str, Any]]:
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
                    data = dict(row._mapping)
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
        """Categorize errors based on error message patterns."""
        if not error_message:
            return None
        
        error_lower = error_message.lower()
        
        if "timeout" in error_lower:
            return "timeout"
        elif "rate limit" in error_lower:
            return "rate_limit"
        elif "connection" in error_lower or "network" in error_lower:
            return "network"
        elif "parsing" in error_lower or "json" in error_lower:
            return "parsing"
        elif "tool" in error_lower:
            return "tool_error"
        elif "memory" in error_lower or "resource" in error_lower:
            return "resource"
        else:
            return "other"


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