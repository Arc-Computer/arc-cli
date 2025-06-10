"""Hybrid state management with database and file storage."""

import asyncio
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
from uuid import uuid4

from arc.cli.utils.state import CLIState, RunResult
from arc.cli.utils.console import ArcConsole
from arc.cli.utils.db_connection import db_manager

console = ArcConsole()


class HybridState(CLIState):
    """State manager that writes to both file and database storage."""
    
    def __init__(self, db_connected: bool = False):
        """Initialize hybrid state manager.
        
        Args:
            db_connected: Whether database is connected
        """
        super().__init__()
        self.db_connected = db_connected
        self._user_id = self._get_or_create_user_id()
    
    def _get_or_create_user_id(self) -> str:
        """Get or create a user ID for database tracking."""
        # For day 1, use a simple approach
        user_id = self.config.get("user_id")
        if not user_id:
            user_id = f"user_{uuid4().hex[:12]}"
            self.config["user_id"] = user_id
            self._save_config()
        return user_id
    
    async def save_run_async(self, result: RunResult) -> Path:
        """Save run results to both file and database.
        
        Args:
            result: Run result to save
            
        Returns:
            Path to saved run directory
        """
        # Save to files first (existing behavior)
        run_dir = super().save_run(result)
        
        # Also save to database if available
        if self.db_connected:
            try:
                await self._save_to_database(result)
            except Exception as e:
                console.print(
                    f"Warning: Database save failed: {str(e)}. Data saved to files only.",
                    style="warning"
                )
        
        return run_dir
    
    async def _save_to_database(self, result: RunResult) -> None:
        """Save run result to database.
        
        Args:
            result: Run result to save
        """
        async with db_manager.get_client() as db_client:
            if not db_client:
                return
                
            # Create configuration if not exists
            config_name = Path(result.config_path).stem
            config_id = await db_client.create_configuration(
                name=config_name,
                user_id=self._user_id,
                initial_config={"path": result.config_path}  # Simplified for now
            )
            
            # Create simulation record
            simulation_id = await db_client.create_simulation(
                config_version_id=config_id,  # Using config_id directly for now
                scenario_set=[s.get("scenario_id", f"scenario_{i}") 
                             for i, s in enumerate(result.scenarios)],
                simulation_name=result.run_id,
                simulation_type="evaluation",
                status="completed",
                overall_score=result.reliability_score,
                total_cost_usd=result.total_cost,
                metadata={
                    "scenario_count": result.scenario_count,
                    "success_count": result.success_count,
                    "failure_count": result.failure_count,
                    "execution_time": result.execution_time
                }
            )
            
            # Save outcomes in batch
            if result.results:
                outcomes = []
                for r in result.results:
                    outcome = {
                        "simulation_id": simulation_id,
                        "scenario_id": r.get("scenario_id", "unknown"),
                        "status": "success" if r.get("success") else "error",
                        "reliability_score": r.get("reliability_score", 1.0 if r.get("success") else 0.0),
                        "execution_time_ms": int(r.get("execution_time", 0) * 1000),
                        "tokens_used": r.get("tokens_used", 0),
                        "cost_usd": r.get("cost", 0.0),
                        "trajectory": r.get("trajectory", {}),
                        "modal_call_id": r.get("modal_call_id"),
                        "error_code": r.get("error_code"),
                        "error_category": self._categorize_error(r.get("failure_reason"))
                    }
                    outcomes.append(outcome)
                
                await db_client.record_outcomes_batch(outcomes)
            
            console.print(f"Run {result.run_id} saved to database", style="success")
    
    def _categorize_error(self, failure_reason: Optional[str]) -> Optional[str]:
        """Categorize error based on failure reason.
        
        Args:
            failure_reason: The failure reason text
            
        Returns:
            Error category or None
        """
        if not failure_reason:
            return None
            
        failure_lower = failure_reason.lower()
        if "currency" in failure_lower:
            return "currency_assumption"
        elif "timeout" in failure_lower:
            return "timeout"
        elif "tool" in failure_lower:
            return "tool_error"
        elif "api" in failure_lower:
            return "api_error"
        else:
            return "other"
    
    def save_run(self, result: RunResult) -> Path:
        """Synchronous wrapper for save_run_async.
        
        Args:
            result: Run result to save
            
        Returns:
            Path to saved run directory
        """
        # First save to files synchronously
        run_dir = super().save_run(result)
        
        # Then try to save to database if available
        if self.db_connected:
            try:
                # Run async operation in sync context
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                loop.run_until_complete(self._save_to_database(result))
                loop.close()
            except Exception as e:
                console.print(
                    f"Warning: Database save failed: {str(e)}. Data saved to files only.",
                    style="warning"
                )
        
        return run_dir