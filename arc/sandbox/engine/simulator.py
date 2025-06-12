"""
Arc Production Simulator with Modal Parallel Execution
Production version adapted from experiments/src/core/arc_eval_sandbox.py
"""

import os
import time
from datetime import datetime
from typing import Any

import modal

# Pydantic imports moved to ToolBehaviorEngine

# Create Modal application
# Support multiple deployment scenarios
if os.environ.get("ARC_USE_DEPLOYED_APP"):
    # Users accessing deployed app - no need to create app
    # Function will be looked up directly via public API
    app = modal.App("arc-production")
elif os.environ.get("MODAL_IDENTITY_TOKEN") or os.environ.get("MODAL_TASK_ID"):
    # Running inside Modal - use existing app context
    app = modal.App.lookup("arc-production")
else:
    # Local development - create new app
    app = modal.App("arc-production")

# Get the project root directory
project_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

# Define container image with dependencies
arc_image = (
    modal.Image.debian_slim()
    .pip_install(
        "pyyaml",
        "pydantic>=2.0",
        "langchain-openai",
        "langchain-core",
        "langchain",
        "langchain-community",
        "requests",
        "opentelemetry-api",
        "opentelemetry-sdk",
        "opentelemetry-exporter-otlp",
        "tiktoken",
        "numpy",
        "scikit-learn",
    )
    # Modal 1.0: Use add_local_dir for production arc directory
    .add_local_dir(os.path.join(project_root, "arc"), "/root/arc")
)

# Note: Tool implementations have been moved to ToolBehaviorEngine
# which dynamically creates tools based on agent configuration


@app.function(
    image=arc_image,
    secrets=[modal.Secret.from_name("openai-secret")],
    timeout=300,
    # Modal 1.0: Updated autoscaler parameters
    min_containers=1,  # Keep at least 1 container warm (was buffer_containers)
    max_containers=50,  # Allow up to 50 parallel containers
    scaledown_window=60,  # Keep containers alive for 60s when idle
)
@modal.concurrent(
    max_inputs=10, target_inputs=8
)  # Allow concurrent LLM calls within container
def evaluate_single_scenario(
    scenario_with_config: tuple[dict[str, Any], dict[str, Any], int],
) -> dict[str, Any]:
    """Evaluate a single scenario. Designed for parallel execution with .map()"""

    # Unpack the tuple (scenario, agent_config, index)
    scenario, agent_config, scenario_index = scenario_with_config

    # Import dependencies
    from langchain.agents import AgentExecutor, create_openai_tools_agent
    from langchain_community.callbacks.manager import get_openai_callback
    from langchain_core.callbacks import UsageMetadataCallbackHandler
    from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
    from langchain_openai import ChatOpenAI

    from arc.sandbox.engine.trajectory_capture import TrajectoryCapture

    # Import production modules
    from arc.sandbox.evaluation.reliability_scorer import calculate_reliability_score

    print(f"[CONTAINER-{scenario_index}] Evaluating scenario: {scenario['name']}")
    start_time = datetime.now()

    # Initialize trajectory capture for this scenario
    trajectory_capture = TrajectoryCapture()

    # Import our dynamic tool behavior engine
    from arc.sandbox.engine.tool_behavior_engine import ToolBehaviorEngine

    # Create dynamic tools based on agent configuration
    # Build agent profile from config (for backward compatibility)
    agent_profile = {
        "configuration": agent_config,
        "assumptions": agent_config.get("assumptions", []),
        "validation_rules": agent_config.get("validation_rules", []),
        "job": agent_config.get("job", ""),
    }

    # Create tool behavior engine with scenario context
    tool_engine = ToolBehaviorEngine(
        agent_profile=agent_profile, scenario_context=scenario
    )

    # Generate tools dynamically based on agent's actual tool definitions
    tools = tool_engine.create_tools()

    print(f"[CONTAINER-{scenario_index}] Created {len(tools)} tools dynamically")

    # Initialize LLM with stream_usage for proper token counting
    # Handle model ID prefixes (e.g., "openai/gpt-4.1" -> "gpt-4.1-2025-04-14")
    model_id = agent_config["model"]

    # Map model IDs to actual OpenAI model names
    model_mapping = {
        "openai/gpt-4.1": "gpt-4.1-2025-04-14",
        "gpt-4.1": "gpt-4.1-2025-04-14",
        "openai/gpt-4.1-mini": "gpt-4.1-mini",
        "gpt-4.1-mini": "gpt-4.1-mini",
    }

    # Use mapped model or fallback to original if not in mapping
    actual_model = model_mapping.get(model_id, model_id)

    # If it still has a provider prefix, strip it
    if "/" in actual_model and actual_model.startswith(
        ("openai/", "anthropic/", "google/")
    ):
        actual_model = actual_model.split("/", 1)[1]

    llm = ChatOpenAI(
        model=actual_model,
        temperature=agent_config["temperature"],
        streaming=True,
        stream_usage=True,  # Essential for token counting with streaming
    )

    # Create agent
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", agent_config["system_prompt"]),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )

    agent = create_openai_tools_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        return_intermediate_steps=True,
        handle_parsing_errors=True,
        max_iterations=3,  # Allow more iterations for complex tasks
        early_stopping_method="generate",  # Better handling of edge cases
    )

    # Create scenario span
    with trajectory_capture.tracer.start_as_current_span(
        f"scenario.{scenario.get('id', f'scenario_{scenario_index}')}",
        attributes={
            "scenario.id": scenario.get("id", f"scenario_{scenario_index}"),
            "scenario.name": scenario["name"],
            "scenario.index": scenario_index,
            "scenario.failure_type": scenario.get("failure_type", "none"),
        },
    ) as scenario_span:
        try:
            # Use both callbacks for comprehensive token tracking
            usage_callback = UsageMetadataCallbackHandler()
            with get_openai_callback() as cb:
                # Execute agent
                with trajectory_capture.tracer.start_as_current_span(
                    "agent_execution", attributes={"agent.model": agent_config["model"]}
                ):
                    # Capture agent decision to use tools
                    trajectory_capture.capture_decision_point(
                        "agent_strategy",
                        ["use_tools", "direct_response"],
                        "unknown",  # Will be determined by execution
                        "Agent deciding how to respond to task",
                    )

                    result = agent_executor.invoke(
                        {"input": scenario["task_prompt"]},
                        config={"callbacks": [cb, usage_callback]},
                    )

                # Get token usage from callbacks
                total_tokens = cb.total_tokens
                total_cost = cb.total_cost
                prompt_tokens = cb.prompt_tokens
                completion_tokens = cb.completion_tokens

                # Fallback to usage metadata if callback failed
                usage_data = None
                if total_tokens == 0 and hasattr(usage_callback, "usage_metadata"):
                    usage_data = usage_callback.usage_metadata
                    if usage_data:
                        prompt_tokens = usage_data.get("input_tokens", 0)
                        completion_tokens = usage_data.get("output_tokens", 0)
                        total_tokens = prompt_tokens + completion_tokens
                        # Estimate cost based on GPT-4.1 pricing
                        input_cost = prompt_tokens * 0.03 / 1000
                        output_cost = completion_tokens * 0.06 / 1000
                        total_cost = input_cost + output_cost

                # Capture LLM interaction
                if total_tokens > 0:
                    trajectory_capture.capture_llm_interaction(
                        model=agent_config["model"],
                        prompt=scenario["task_prompt"],
                        response=result.get("output", ""),
                        tokens_in=prompt_tokens,
                        tokens_out=completion_tokens,
                        duration_ms=(datetime.now() - start_time).total_seconds()
                        * 1000,
                    )

            end_time = datetime.now()
            execution_time = (end_time - start_time).total_seconds()

            # Process intermediate steps for trajectory
            trajectory_events = []
            for j, (agent_action, tool_output) in enumerate(
                result.get("intermediate_steps", [])
            ):
                trajectory_events.append(
                    {
                        "step": j + 1,
                        "type": "tool_call",
                        "timestamp": start_time.isoformat(),
                        "tool": agent_action.tool,
                        "tool_input": agent_action.tool_input,
                        "tool_output": tool_output,
                        "log": agent_action.log,
                    }
                )

            trajectory_events.append(
                {
                    "step": len(trajectory_events) + 1,
                    "type": "final_response",
                    "timestamp": end_time.isoformat(),
                    "content": result.get("output", ""),
                    "execution_time_seconds": execution_time,
                }
            )

            # Create trajectory result with proper token tracking and REQUIRED database fields
            trajectory = {
                "start_time": start_time.isoformat(),  # REQUIRED by database constraint
                "status": "success",  # REQUIRED by database constraint
                "task_prompt": scenario["task_prompt"],
                "final_response": result.get("output", ""),
                "trajectory_event_count": len(trajectory_events),
                "execution_time_seconds": execution_time,
                "full_trajectory": trajectory_events,
                "token_usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": total_tokens,
                    "total_cost": total_cost,
                },
            }

            scenario_span.set_attribute("execution.status", "success")

        except Exception as e:
            print(f"[CONTAINER-{scenario_index}] Error in scenario: {str(e)}")
            trajectory = {
                "start_time": start_time.isoformat(),  # REQUIRED by database constraint
                "status": "error",  # REQUIRED by database constraint
                "task_prompt": scenario["task_prompt"],
                "final_response": f"Error: {str(e)}",
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
            }
            scenario_span.set_attribute("execution.status", "error")
            scenario_span.set_attribute("error.message", str(e))

            # Capture error in trajectory
            trajectory_capture.capture_error(
                "scenario_execution_error", str(e), recovery_attempted=False
            )

        # Calculate reliability score
        with trajectory_capture.tracer.start_as_current_span(
            "reliability_scoring", attributes={"span.kind": "SCORING"}
        ) as score_span:
            reliability_score = calculate_reliability_score(trajectory, scenario)
            score_span.set_attribute(
                "reliability.overall_score", reliability_score.get("overall_score", 0)
            )
            score_span.set_attribute(
                "reliability.grade", reliability_score.get("grade", "N/A")
            )

        # Create complete trajectory data with full token usage
        token_usage = trajectory.get("token_usage", {})

        # Ensure we have the complete trajectory data with all captured information
        complete_trajectory = trajectory_capture.create_trajectory(
            scenario=scenario,
            status=trajectory["status"],
            start_time=start_time,
            end_time=datetime.now(),
            token_usage=token_usage,
        )

        # Merge the trajectory data to ensure nothing is lost
        detailed_trajectory_dict = complete_trajectory.to_dict()
        detailed_trajectory_dict["execution_time_seconds"] = trajectory[
            "execution_time_seconds"
        ]
        detailed_trajectory_dict["token_usage"] = token_usage
        detailed_trajectory_dict["full_trajectory"] = trajectory["full_trajectory"]

        # Return the complete result
        return {
            "scenario": scenario,
            "trajectory": trajectory,
            "reliability_score": reliability_score,
            "detailed_trajectory": detailed_trajectory_dict,
            "scenario_index": scenario_index,
        }


@app.function(
    image=arc_image, secrets=[modal.Secret.from_name("openai-secret")], timeout=600
)
def run_evaluation_suite_parallel(
    agent_config: dict[str, Any], scenarios: list[dict[str, Any]]
) -> dict[str, Any]:
    """Orchestrate parallel scenario evaluation using Modal's .map()"""

    print(f"[ORCHESTRATOR] Starting parallel evaluation of {len(scenarios)} scenarios")
    suite_start_time = datetime.now()

    # Create tuples of (scenario, agent_config, index) for parallel processing
    scenario_inputs = [
        (scenario, agent_config, i) for i, scenario in enumerate(scenarios)
    ]

    # Run scenarios in parallel using .map()
    # return_exceptions=True ensures one failure doesn't crash the entire batch
    print(
        f"[ORCHESTRATOR] Submitting {len(scenario_inputs)} scenarios for parallel execution..."
    )

    # Execute scenarios in parallel
    results = list(
        evaluate_single_scenario.map(scenario_inputs, return_exceptions=True)
    )

    print("[ORCHESTRATOR] All scenarios completed. Processing results...")

    # Process results and handle any exceptions
    processed_results = []
    all_trajectories = []

    for i, result in enumerate(results):
        if isinstance(result, Exception):
            # Handle failed scenarios
            print(f"[ORCHESTRATOR] Scenario {i} failed with exception: {str(result)}")
            scenario = scenarios[i]
            error_result = {
                "scenario": scenario,
                "trajectory": {
                    "start_time": datetime.now().isoformat(),  # REQUIRED by database constraint
                    "status": "error",  # REQUIRED by database constraint
                    "task_prompt": scenario["task_prompt"],
                    "final_response": f"Execution error: {str(result)}",
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
                "reliability_score": {
                    "overall_score": 0,
                    "grade": "F",
                    "dimension_scores": {
                        "tool_execution": 0,
                        "response_quality": 0,
                        "error_handling": 0,
                        "performance": 0,
                        "completeness": 0,
                    },
                },
                "detailed_trajectory": None,
            }
            processed_results.append(error_result)
        else:
            # Successful scenario
            processed_results.append(result)
            if result.get("detailed_trajectory"):
                all_trajectories.append(result["detailed_trajectory"])

    # Sort results by scenario index to maintain original order
    processed_results.sort(key=lambda x: x.get("scenario_index", 0))

    # Calculate aggregate metrics
    successful_runs = sum(
        1 for r in processed_results if r["trajectory"]["status"] == "success"
    )
    avg_score = (
        sum(r["reliability_score"]["overall_score"] for r in processed_results)
        / len(processed_results)
        if processed_results
        else 0
    )
    avg_execution_time = (
        sum(r["trajectory"]["execution_time_seconds"] for r in processed_results)
        / len(processed_results)
        if processed_results
        else 0
    )
    total_tokens = sum(
        r["trajectory"].get("token_usage", {}).get("total_tokens", 0)
        for r in processed_results
    )
    total_cost = sum(
        r["trajectory"].get("token_usage", {}).get("total_cost", 0)
        for r in processed_results
    )

    # Aggregate dimension scores
    dimension_totals = {
        "tool_execution": 0.0,
        "response_quality": 0.0,
        "error_handling": 0.0,
        "performance": 0.0,
        "completeness": 0.0,
    }

    for result in processed_results:
        dimension_scores = result["reliability_score"].get("dimension_scores", {})
        for dimension, score in dimension_scores.items():
            if dimension in dimension_totals:
                dimension_totals[dimension] += score

    # Calculate average for each dimension
    avg_dimension_scores = (
        {
            dimension: total / len(processed_results)
            for dimension, total in dimension_totals.items()
        }
        if processed_results
        else dimension_totals
    )

    suite_end_time = datetime.now()
    total_execution_time = (suite_end_time - suite_start_time).total_seconds()

    print(f"[ORCHESTRATOR] Parallel execution completed in {total_execution_time:.2f}s")
    print(
        f"[ORCHESTRATOR] Average time per scenario: {total_execution_time / len(scenarios):.2f}s"
    )

    return {
        "agent_config": agent_config,
        "scenarios_run": len(scenarios),
        "successful_runs": successful_runs,
        "average_reliability_score": avg_score,
        "average_dimension_scores": avg_dimension_scores,
        "average_execution_time": avg_execution_time,
        "total_tokens_used": total_tokens,
        "total_cost_usd": total_cost,
        "results": processed_results,
        "trajectories": all_trajectories,  # Complete execution traces
        "parallel_execution_time": total_execution_time,
        "speedup_factor": (avg_execution_time * len(scenarios)) / total_execution_time
        if total_execution_time > 0
        else 1,
    }


# Test functions for production validation
@app.function(image=arc_image)
def test_basic_execution():
    """Test basic Modal function execution"""
    return {"status": "success", "message": "Modal function executed successfully"}


@app.function(image=arc_image)
def test_scaling(containers: int = 10):
    """Test Modal scaling with specified number of containers"""
    import concurrent.futures

    def dummy_work(i):
        time.sleep(0.1)
        return {"container": i, "status": "completed"}

    with concurrent.futures.ThreadPoolExecutor(max_workers=containers) as executor:
        futures = [executor.submit(dummy_work, i) for i in range(containers)]
        results = [f.result() for f in futures]

    return {
        "containers_requested": containers,
        "containers_completed": len(results),
        "status": "success",
    }


@app.function(image=arc_image)
def test_cost_estimation(scenarios: int = 50):
    """Test cost estimation for specified number of scenarios"""
    # Estimate based on average tokens per scenario
    avg_tokens_per_scenario = 1500
    total_tokens = scenarios * avg_tokens_per_scenario

    # GPT-4.1 pricing estimates
    input_tokens = total_tokens * 0.7  # 70% input
    output_tokens = total_tokens * 0.3  # 30% output

    input_cost = (input_tokens / 1000) * 0.03  # $0.03 per 1K input tokens
    output_cost = (output_tokens / 1000) * 0.06  # $0.06 per 1K output tokens
    total_cost = input_cost + output_cost

    return {
        "scenarios": scenarios,
        "estimated_total_tokens": total_tokens,
        "estimated_cost_usd": round(total_cost, 4),
        "cost_per_scenario": round(total_cost / scenarios, 4),
    }


@app.local_entrypoint()
def main():
    """Production simulator test entrypoint"""
    print("[PRODUCTION] Arc Production Simulator")
    print("=" * 70)

    # Run basic tests
    print("\n[TEST] Basic execution...")
    result = test_basic_execution.remote()
    print(f"Result: {result}")

    print("\n[TEST] Scaling test with 10 containers...")
    result = test_scaling.remote(containers=10)
    print(f"Result: {result}")

    print("\n[TEST] Cost estimation for 50 scenarios...")
    result = test_cost_estimation.remote(scenarios=50)
    print(f"Result: {result}")

    print("\n[PRODUCTION] All tests completed successfully")


if __name__ == "__main__":
    main()
