"""Public API for Arc simulations on Modal.

This module provides a clean interface for using deployed Modal functions
without requiring authentication.
"""

import os
from collections.abc import AsyncIterator
from typing import Any

try:
    import modal
    MODAL_AVAILABLE = True
except ImportError:
    MODAL_AVAILABLE = False


class ArcModalAPI:
    """Public API for Arc simulations on Modal."""

    @staticmethod
    async def run_scenarios(
        scenarios: list[dict[str, Any]],
        agent_config: dict[str, Any],
        batch_size: int = 10
    ) -> AsyncIterator[dict[str, Any]]:
        """Run scenarios using deployed Modal app.

        Args:
            scenarios: List of scenarios to execute
            agent_config: Agent configuration
            batch_size: Number of scenarios per batch

        Yields:
            Results from scenario execution

        Raises:
            RuntimeError: If Modal is not available or function lookup fails
        """
        if not MODAL_AVAILABLE:
            raise RuntimeError("Modal not installed")

        try:
            # Look up the deployed function
            app_name = "arc-production"
            func_name = "evaluate_single_scenario"

            evaluate_fn = modal.Function.from_name(app_name, func_name)

        except Exception as e:
            raise RuntimeError(f"Failed to find deployed Modal function: {e}") from e

        # Prepare scenarios with config
        scenario_tuples = [
            (scenario, agent_config, i)
            for i, scenario in enumerate(scenarios)
        ]

        # Execute in batches using the deployed function
        for i in range(0, len(scenario_tuples), batch_size):
            batch = scenario_tuples[i:i + batch_size]

            try:
                # Use the remote function's map interface
                async for result in evaluate_fn.map.aio(batch):
                    yield result
            except Exception as e:
                # Return error results for failed batch
                for scenario_tuple in batch:
                    yield {
                        "scenario_id": f"scenario_{scenario_tuple[2]}",
                        "success": False,
                        "execution_time": 0,
                        "failure_reason": f"Modal execution error: {str(e)}",
                        "cost": 0,
                        "trajectory": {},
                        "reliability_score": {"overall_score": 0}
                    }

    @staticmethod
    def is_available() -> bool:
        """Check if deployed Modal app is available."""
        if not MODAL_AVAILABLE:
            return False

        # Check if we're configured to use deployed app
        if not os.environ.get("ARC_USE_DEPLOYED_APP"):
            return False

        try:
            # Try to find the deployed function
            modal.Function.from_name("arc-production", "evaluate_single_scenario")
            return True
        except Exception:
            return False

