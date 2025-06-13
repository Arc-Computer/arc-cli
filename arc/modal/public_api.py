"""Public API for Arc simulations on Modal.

This module provides a clean interface for using deployed Modal functions
without requiring authentication by using HTTP web endpoints.
"""

import os
import asyncio
import aiohttp
import json
import logging
from datetime import datetime
from collections.abc import AsyncIterator
from typing import Any

from arc.cli.utils.json_utils import json_serializer

# Set up debug logging for Modal responses
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class ArcModalAPI:
    """Public API for Arc simulations on Modal using HTTP endpoints."""

    @staticmethod
    async def run_scenarios(
        scenarios: list[dict[str, Any]],
        agent_config: dict[str, Any],
        batch_size: int = 20  # Increased for better parallel performance
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
        
        print(f"🔍 DEBUG: Using Modal endpoint: {base_url}")
        logger.info(f"Using Modal endpoint: {base_url}")

        # Execute scenarios in parallel batches
        async with aiohttp.ClientSession() as session:
            for i in range(0, len(scenarios), batch_size):
                batch = scenarios[i:i + batch_size]
                
                print(f"🔍 DEBUG: Starting parallel batch of {len(batch)} scenarios")
                
                # Create concurrent tasks for parallel execution
                async def process_scenario(scenario, scenario_index):
                    request_data = {
                        "scenario": scenario,
                        "agent_config": agent_config,
                        "scenario_index": scenario_index
                    }
                    
                    try:
                        # Serialize the request data with custom serializer
                        json_data = json.dumps(request_data, default=json_serializer)
                        
                        print(f"🔍 DEBUG: Sending request to Modal for scenario {scenario_index}: {scenario.get('name', 'Unknown')}")
                        logger.info(f"Sending request to Modal for scenario {scenario_index}")
                        
                        async with session.post(
                            base_url,
                            data=json_data,
                            headers={"Content-Type": "application/json"},
                            timeout=aiohttp.ClientTimeout(total=300)  # 5 minute timeout
                        ) as response:
                            print(f"🔍 DEBUG: Modal response status: {response.status} for scenario {scenario_index}")
                            logger.info(f"Modal response status: {response.status}")
                            
                            if response.status == 200:
                                result = await response.json()
                                
                                # Debug log the raw response
                                print(f"🔍 DEBUG: Raw Modal response for scenario {scenario_index}:")
                                print(f"   Reliability score: {result.get('reliability_score', 'MISSING')}")
                                print(f"   Trajectory status: {result.get('trajectory', {}).get('status', 'MISSING')}")
                                print(f"   Response keys: {list(result.keys())}")
                                logger.debug(f"Modal response for scenario {scenario_index}: {json.dumps(result, indent=2, default=str)}")
                                
                                return result
                            else:
                                error_text = await response.text()
                                # Return error result
                                return {
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
                                 
                    except Exception as e:
                        # Return error result
                        return {
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
                
                # Create tasks for parallel execution
                tasks = []
                for j, scenario in enumerate(batch):
                    scenario_index = i + j
                    task = asyncio.create_task(process_scenario(scenario, scenario_index))
                    tasks.append(task)
                
                # Wait for all tasks in batch to complete and yield results
                results = await asyncio.gather(*tasks, return_exceptions=True)
                for j, result in enumerate(results):
                    if isinstance(result, Exception):
                        # Handle exceptions from tasks - create synthetic error result
                        scenario_index = i + j
                        scenario = batch[j] if j < len(batch) else {}
                        
                        print(f"🔍 DEBUG: Task failed with exception: {result}")
                        logger.error(f"Task for scenario {scenario_index} failed: {result}")
                        
                        # Yield synthetic error result matching success result shape
                        synthetic_result = {
                            "scenario": scenario,
                            "trajectory": {
                                "start_time": datetime.now().isoformat(),
                                "status": "error",
                                "task_prompt": scenario.get("task_prompt", ""),
                                "final_response": f"Task execution error: {str(result)}",
                                "trajectory_event_count": 0,
                                "execution_time_seconds": 0,
                                "full_trajectory": [],
                                "error": str(result),
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
                        yield synthetic_result
                    else:
                        yield result

    @staticmethod
    def is_available() -> bool:
        """Check if deployed Modal web endpoint is available."""
        # For HTTP endpoints, we assume they're available if configured
        # The actual availability will be tested when making requests
        return True

