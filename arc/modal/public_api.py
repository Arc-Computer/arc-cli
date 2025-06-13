"""Public API for Arc simulations on Modal.

This module provides a clean interface for using deployed Modal functions
without requiring authentication by using HTTP web endpoints.
"""

import os
import asyncio
import aiohttp
import json
from datetime import datetime
from collections.abc import AsyncIterator
from typing import Any

from arc.cli.utils.json_utils import json_serializer


class ArcModalAPI:
    """Public API for Arc simulations on Modal using HTTP endpoints."""

    @staticmethod
    async def run_scenarios(
        scenarios: list[dict[str, Any]],
        agent_config: dict[str, Any],
        batch_size: int = 10
    ) -> AsyncIterator[dict[str, Any]]:
        """Run scenarios using deployed Modal web endpoint.

        Args:
            scenarios: List of scenarios to execute
            agent_config: Agent configuration
            batch_size: Number of scenarios per batch

        Yields:
            Results from scenario execution

        Raises:
            RuntimeError: If the web endpoint is not available
        """
        # Get the web endpoint URL
        base_url = os.environ.get("ARC_MODAL_ENDPOINT_URL")
        if not base_url:
            # Default URL pattern for Modal web endpoints
            workspace = os.environ.get("ARC_MODAL_WORKSPACE", "your-workspace")
            base_url = f"https://{workspace}--arc-production-evaluate-scenario-endpoint.modal.run"

        # Execute scenarios in batches
        async with aiohttp.ClientSession() as session:
            for i in range(0, len(scenarios), batch_size):
                batch = scenarios[i:i + batch_size]

                # Process each scenario in the batch
                for j, scenario in enumerate(batch):
                    scenario_index = i + j
                    
                    request_data = {
                        "scenario": scenario,
                        "agent_config": agent_config,
                        "scenario_index": scenario_index
                    }
                    
                    try:
                        # Serialize the request data with custom serializer
                        json_data = json.dumps(request_data, default=json_serializer)
                        
                        async with session.post(
                            base_url,
                            data=json_data,
                            headers={"Content-Type": "application/json"},
                            timeout=aiohttp.ClientTimeout(total=300)  # 5 minute timeout
                        ) as response:
                            if response.status == 200:
                                result = await response.json()
                                yield result
                            else:
                                error_text = await response.text()
                                # Return error result
                                error_result = {
                                    "scenario": scenario,
                                    "trajectory": {
                                        "start_time": datetime.now().isoformat(),
                                        "status": "error",
                                        "task_prompt": scenario.get("task_prompt", ""),
                                        "final_response": f"HTTP error {response.status}: {error_text}",
                                        "trajectory_event_count": 0,
                                        "execution_time_seconds": 0,
                                        "full_trajectory": [],
                                        "error": f"HTTP {response.status}",
                                        "token_usage": {
                                            "prompt_tokens": 0,
                                            "completion_tokens": 0,
                                            "total_tokens": 0,
                                            "total_cost": 0,
                                        },
                                    },
                                    "reliability_score": {"overall_score": 0},
                                    "scenario_index": scenario_index,
                                }
                                yield error_result
                                 
                    except Exception as e:
                        # Return error result
                        error_result = {
                            "scenario": scenario,
                            "trajectory": {
                                "start_time": datetime.now().isoformat(),
                                "status": "error", 
                                "task_prompt": scenario.get("task_prompt", ""),
                                "final_response": f"Request error: {str(e)}",
                                "trajectory_event_count": 0,
                                "execution_time_seconds": 0,
                                "full_trajectory": [],
                                "error": str(e),
                                "token_usage": {
                                    "prompt_tokens": 0,
                                    "completion_tokens": 0,
                                    "total_tokens": 0,
                                    "total_cost": 0,
                                },
                            },
                            "reliability_score": {"overall_score": 0},
                            "scenario_index": scenario_index,
                        }
                        yield error_result

    @staticmethod
    def is_available() -> bool:
        """Check if deployed Modal web endpoint is available."""
        # For HTTP endpoints, we assume they're available if configured
        # The actual availability will be tested when making requests
        return True

