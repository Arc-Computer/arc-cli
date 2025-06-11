"""Hybrid state management with database and file storage."""

import asyncio
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from arc.cli.utils.console import ArcConsole
from arc.cli.utils.db_connection import db_manager
from arc.cli.utils.error_helpers import categorize_error
from arc.cli.utils.state import CLIState, RunResult

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
            # Use a proper UUID format for database compatibility
            user_id = str(uuid4())
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
                    style="warning",
                )

        return run_dir

    async def _save_to_database(self, result: RunResult) -> None:
        """Save run result to database.

        Args:
            result: Run result to save
        """
        db_client = db_manager.get_client()
        if not db_client:
            return

        # Create configuration if not exists
        config_name = Path(result.config_path).stem
        version_id = await db_client.create_configuration(
            name=config_name,
            user_id=self._user_id,
            initial_config={"path": result.config_path},  # Simplified for now
        )

        # Create simulation record
        simulation_id = await db_client.create_simulation(
            config_version_id=version_id,
            scenario_set=[
                s.get("scenario_id", f"scenario_{i}")
                for i, s in enumerate(result.scenarios)
            ],
            simulation_name=result.run_id,
            modal_app_id=None,
        )

        # Save outcomes in batch
        if result.results:
            outcomes = []
            for r in result.results:
                outcome = {
                    "simulation_id": simulation_id,
                    "scenario_id": r.get("scenario_id", "unknown"),
                    "status": "success" if r.get("success") else "error",
                    "reliability_score": r.get(
                        "reliability_score", 1.0 if r.get("success") else 0.0
                    ),
                    "execution_time_ms": int(r.get("execution_time", 0) * 1000),
                    "tokens_used": r.get("tokens_used", 0),
                    "cost_usd": r.get("cost", 0.0),
                    "trajectory": r.get("trajectory", {}),
                    "modal_call_id": r.get("modal_call_id"),
                    "error_code": r.get("error_code"),
                    "error_category": categorize_error(r.get("failure_reason")),
                }
                outcomes.append(outcome)

            await db_client.record_outcomes_batch(outcomes)

        await db_client.finalize_simulation(
            simulation_id=simulation_id,
            status="completed",
            overall_score=result.reliability_score,
            total_cost_usd=result.total_cost,
            execution_time_ms=int(result.execution_time * 1000),
            metadata={
                "scenario_count": result.scenario_count,
                "success_count": result.success_count,
                "failure_count": result.failure_count,
            },
            completed_scenarios=result.scenario_count,
            completed_at=datetime.utcnow(),
        )

        console.print(f"Run {result.run_id} saved to database", style="success")

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
                # Check if an event loop is already running
                try:
                    loop = asyncio.get_running_loop()
                    # We're in an async context, use ensure_future for fire-and-forget
                    # This avoids blocking the current operation
                    future = asyncio.ensure_future(self._save_to_database(result))
                    # Add error handler to log any failures
                    future.add_done_callback(self._handle_db_save_error)
                except RuntimeError:
                    # No event loop running - we're in a sync context
                    # Create a new event loop in a thread to avoid blocking
                    import threading
                    
                    def run_async_save():
                        try:
                            # Create a new event loop for this thread
                            new_loop = asyncio.new_event_loop()
                            asyncio.set_event_loop(new_loop)
                            new_loop.run_until_complete(self._save_to_database(result))
                            new_loop.close()
                        except Exception as e:
                            console.print(
                                f"Warning: Background database save failed: {str(e)}. Data saved to files only.",
                                style="warning",
                            )
                    
                    # Run in a background thread to avoid blocking
                    thread = threading.Thread(target=run_async_save, daemon=True)
                    thread.start()
            except Exception as e:
                console.print(
                    f"Warning: Database save failed: {str(e)}. Data saved to files only.",
                    style="warning",
                )

        return run_dir
    
    def _handle_db_save_error(self, task: asyncio.Task) -> None:
        """Handle errors from async database save task."""
        try:
            # This will raise an exception if the task failed
            task.result()
        except Exception as e:
            # Log the error but don't crash - data is already saved to files
            console.print(
                f"Warning: Async database save failed: {str(e)}. Data saved to files only.",
                style="warning",
            )
