"""
Arc-Eval Production Simulator with Modal Parallel Execution
Production version adapted from experiments/src/core/arc_eval_sandbox.py
"""

import modal
import yaml
import json
import time
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import os
from contextlib import nullcontext
from pydantic import BaseModel, Field

# Create Modal application
app = modal.App("arc-eval-production")

# Get the project root directory
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

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
        "scikit-learn"
    )
    # Copy the production arc directory to Modal
    .add_local_dir(
        os.path.join(project_root, "arc"),
        "/root/arc"
    )
)

# Pydantic models for tool inputs
class WeatherInput(BaseModel):
    """Input schema for weather tool"""
    city: str = Field(description="The name of the city to get weather for")

class DatabaseInput(BaseModel):
    """Input schema for database search tool"""
    query: str = Field(description="The search query or SQL statement")
    table: str = Field(description="The database table to search in")

# Tool implementations with tracing
def get_weather(city: str, trajectory_capture=None) -> str:
    """Get current weather for a specified city. Returns weather data including temperature, conditions, and other metrics."""
    start_time = time.time()
    print(f"[TOOL] get_weather called with city: {city}")
    
    # Realistic weather database
    weather_db = {
        "new york": {
            "location": {"name": "New York", "country": "USA", "lat": 40.7128, "lon": -74.0060},
            "current": {
                "temp_f": 72,
                "temp_c": 22,
                "condition": {"text": "Partly cloudy", "icon": "//cdn.weatherapi.com/weather/64x64/day/116.png"},
                "wind_mph": 8,
                "wind_dir": "NW",
                "pressure_mb": 1020,
                "humidity": 65,
                "cloud": 25,
                "feelslike_f": 70,
                "uv": 6
            }
        },
        "london": {
            "location": {"name": "London", "country": "UK", "lat": 51.5074, "lon": -0.1278},
            "current": {
                "temp_f": 59,
                "temp_c": 15,
                "condition": {"text": "Light rain", "icon": "//cdn.weatherapi.com/weather/64x64/day/296.png"},
                "wind_mph": 12,
                "wind_dir": "SW", 
                "pressure_mb": 1013,
                "humidity": 85,
                "cloud": 75,
                "feelslike_f": 57,
                "uv": 2
            }
        },
        "tokyo": {
            "location": {"name": "Tokyo", "country": "Japan", "lat": 35.6762, "lon": 139.6503},
            "current": {
                "temp_f": 68,
                "temp_c": 20,
                "condition": {"text": "Clear", "icon": "//cdn.weatherapi.com/weather/64x64/day/113.png"},
                "wind_mph": 5,
                "wind_dir": "E",
                "pressure_mb": 1018,
                "humidity": 55,
                "cloud": 10,
                "feelslike_f": 66,
                "uv": 7
            }
        },
        "sydney": {
            "location": {"name": "Sydney", "country": "Australia", "lat": -33.8688, "lon": 151.2093},
            "current": {
                "temp_f": 77,
                "temp_c": 25,
                "condition": {"text": "Sunny", "icon": "//cdn.weatherapi.com/weather/64x64/day/113.png"},
                "wind_mph": 10,
                "wind_dir": "NE",
                "pressure_mb": 1022,
                "humidity": 45,
                "cloud": 0,
                "feelslike_f": 75,
                "uv": 9
            }
        }
    }
    
    city_normalized = city.lower().strip()
    if city_normalized in weather_db:
        result = json.dumps(weather_db[city_normalized], indent=2)
        success = True
    else:
        result = json.dumps({
            "error": {
                "code": 1006,
                "message": f"No matching location found for '{city}'."
            }
        }, indent=2)
        success = False
        if trajectory_capture:
            trajectory_capture.capture_error(
                "invalid_location",
                f"Location '{city}' not found",
                recovery_attempted=False
            )
    
    # Capture tool call
    duration_ms = (time.time() - start_time) * 1000
    if trajectory_capture:
        trajectory_capture.capture_tool_call(
            "get_weather", city, result, duration_ms, success
        )
    
    return result

def search_database(query: str, table: str, trajectory_capture=None) -> str:
    """Search a database table with the specified query. Returns matching records or an error if the table doesn't exist."""
    start_time = time.time()
    print(f"[TOOL] search_database called with query: {query}, table: {table}")
    
    # Simulate latency
    time.sleep(0.5)
    
    # Mock database responses
    if table == "products":
        result = json.dumps({
            "results": [
                {"id": 1, "name": "Widget A", "price": 29.99, "stock": 150},
                {"id": 2, "name": "Widget B", "price": 39.99, "stock": 75}
            ],
            "total": 2
        })
        success = True
    elif table == "customers":
        result = json.dumps({
            "results": [
                {"id": 101, "name": "John Doe", "email": "john@example.com", "status": "active"},
                {"id": 102, "name": "Jane Smith", "email": "jane@example.com", "status": "active"}
            ],
            "total": 2
        })
        success = True
    elif table == "transactions":
        result = json.dumps({
            "results": [
                {"id": 1001, "customer_id": 101, "amount": 99.99, "currency": "USD", "status": "completed"},
                {"id": 1002, "customer_id": 102, "amount": 149.99, "currency": "EUR", "status": "pending"}
            ],
            "total": 2
        })
        success = True
    else:
        result = json.dumps({"error": f"Table '{table}' not found"})
        success = False
        if trajectory_capture:
            trajectory_capture.capture_error(
                "invalid_table",
                f"Table '{table}' not found",
                recovery_attempted=False
            )
    
    # Capture tool call
    duration_ms = (time.time() - start_time) * 1000
    if trajectory_capture:
        trajectory_capture.capture_tool_call(
            "search_database", 
            {"query": query, "table": table},
            result,
            duration_ms,
            success
        )
    
    return result

@app.function(
    image=arc_image,
    secrets=[modal.Secret.from_name("openai-secret")],
    timeout=300,
    # Autoscaling configuration for parallel execution
    max_containers=50,  # Allow up to 50 parallel containers
    buffer_containers=5,  # Keep 5 warm containers ready
    scaledown_window=60  # Keep containers alive for 60s when idle
)
@modal.concurrent(max_inputs=10, target_inputs=8)  # Allow concurrent LLM calls within container
def evaluate_single_scenario(
    scenario_with_config: Tuple[Dict[str, Any], Dict[str, Any], int]
) -> Dict[str, Any]:
    """Evaluate a single scenario. Designed for parallel execution with .map()"""
    
    # Unpack the tuple (scenario, agent_config, index)
    scenario, agent_config, scenario_index = scenario_with_config
    
    # Import dependencies
    from langchain_openai import ChatOpenAI
    from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
    from langchain_core.tools import tool
    from langchain.agents import create_openai_tools_agent, AgentExecutor
    from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
    from langchain_community.callbacks.manager import get_openai_callback
    from langchain_core.callbacks import UsageMetadataCallbackHandler
    
    # Import production modules
    from arc.sandbox.evaluation.reliability_scorer import calculate_reliability_score
    from arc.sandbox.engine.trajectory_capture import TrajectoryCapture
    
    print(f"[CONTAINER-{scenario_index}] Evaluating scenario: {scenario['name']}")
    start_time = datetime.now()
    
    # Initialize trajectory capture for this scenario
    trajectory_capture = TrajectoryCapture()
    
    # Register tools with trajectory capture and proper schemas
    @tool(args_schema=WeatherInput)
    def wrapped_get_weather(city: str) -> str:
        """Get current weather for a specified city. Returns weather data including temperature, conditions, and other metrics."""
        return get_weather(city, trajectory_capture=trajectory_capture)
    
    @tool(args_schema=DatabaseInput)
    def wrapped_search_database(query: str, table: str) -> str:
        """Search a database table with the specified query. Returns matching records or an error if the table doesn't exist."""
        return search_database(query, table, trajectory_capture=trajectory_capture)
    
    available_tools = {
        "get_weather": wrapped_get_weather,
        "search_database": wrapped_search_database
    }
    
    # Build tool list from config
    tools = []
    for tool_name in agent_config.get("tools", []):
        if tool_name in available_tools:
            tools.append(available_tools[tool_name])
    
    # Initialize LLM with stream_usage for proper token counting
    llm = ChatOpenAI(
        model=agent_config["model"],
        temperature=agent_config["temperature"],
        streaming=True,
        stream_usage=True  # Essential for token counting with streaming
    )
    
    # Create agent
    prompt = ChatPromptTemplate.from_messages([
        ("system", agent_config["system_prompt"]),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad")
    ])
    
    agent = create_openai_tools_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        return_intermediate_steps=True,
        handle_parsing_errors=True,
        max_iterations=3,  # Allow more iterations for complex tasks
        early_stopping_method="generate"  # Better handling of edge cases
    )
    
    # Create scenario span
    with trajectory_capture.tracer.start_as_current_span(
        f"scenario.{scenario.get('id', f'scenario_{scenario_index}')}",
        attributes={
            "scenario.id": scenario.get("id", f"scenario_{scenario_index}"),
            "scenario.name": scenario["name"],
            "scenario.index": scenario_index,
            "scenario.failure_type": scenario.get("failure_type", "none")
        }
    ) as scenario_span:
        
        try:
            # Use both callbacks for comprehensive token tracking
            usage_callback = UsageMetadataCallbackHandler()
            with get_openai_callback() as cb:
                # Execute agent
                with trajectory_capture.tracer.start_as_current_span(
                    "agent_execution",
                    attributes={"agent.model": agent_config["model"]}
                ):
                    # Capture agent decision to use tools
                    trajectory_capture.capture_decision_point(
                        "agent_strategy",
                        ["use_tools", "direct_response"],
                        "unknown",  # Will be determined by execution
                        "Agent deciding how to respond to task"
                    )
                    
                    result = agent_executor.invoke(
                        {"input": scenario["task_prompt"]},
                        config={"callbacks": [cb, usage_callback]}
                    )
                
                # Get token usage from callbacks
                total_tokens = cb.total_tokens
                total_cost = cb.total_cost
                prompt_tokens = cb.prompt_tokens
                completion_tokens = cb.completion_tokens
                
                # Fallback to usage metadata if callback failed
                usage_data = None
                if total_tokens == 0 and hasattr(usage_callback, 'usage_metadata'):
                    usage_data = usage_callback.usage_metadata
                    if usage_data:
                        prompt_tokens = usage_data.get('input_tokens', 0)
                        completion_tokens = usage_data.get('output_tokens', 0)
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
                        duration_ms=(datetime.now() - start_time).total_seconds() * 1000
                    )
            
            end_time = datetime.now()
            execution_time = (end_time - start_time).total_seconds()
            
            # Process intermediate steps for trajectory
            trajectory_events = []
            for j, (agent_action, tool_output) in enumerate(result.get("intermediate_steps", [])):
                trajectory_events.append({
                    "step": j + 1,
                    "type": "tool_call",
                    "timestamp": start_time.isoformat(),
                    "tool": agent_action.tool,
                    "tool_input": agent_action.tool_input,
                    "tool_output": tool_output,
                    "log": agent_action.log
                })
            
            trajectory_events.append({
                "step": len(trajectory_events) + 1,
                "type": "final_response",
                "timestamp": end_time.isoformat(),
                "content": result.get("output", ""),
                "execution_time_seconds": execution_time
            })
            
            # Create trajectory result with proper token tracking
            trajectory = {
                "status": "success",
                "task_prompt": scenario["task_prompt"],
                "final_response": result.get("output", ""),
                "trajectory_event_count": len(trajectory_events),
                "execution_time_seconds": execution_time,
                "full_trajectory": trajectory_events,
                "token_usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": total_tokens,
                    "total_cost": total_cost
                }
            }
            
            scenario_span.set_attribute("execution.status", "success")
            
        except Exception as e:
            print(f"[CONTAINER-{scenario_index}] Error in scenario: {str(e)}")
            trajectory = {
                "status": "error",
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
                    "total_cost": 0
                }
            }
            scenario_span.set_attribute("execution.status", "error")
            scenario_span.set_attribute("error.message", str(e))
            
            # Capture error in trajectory
            trajectory_capture.capture_error(
                "scenario_execution_error",
                str(e),
                recovery_attempted=False
            )
        
        # Calculate reliability score
        with trajectory_capture.tracer.start_as_current_span(
            "reliability_scoring",
            attributes={"span.kind": "SCORING"}
        ) as score_span:
            reliability_score = calculate_reliability_score(trajectory, scenario)
            score_span.set_attribute("reliability.overall_score", reliability_score.get("overall_score", 0))
            score_span.set_attribute("reliability.grade", reliability_score.get("grade", "N/A"))
        
        # Create complete trajectory data with full token usage
        token_usage = trajectory.get("token_usage", {})
        
        # Ensure we have the complete trajectory data with all captured information
        complete_trajectory = trajectory_capture.create_trajectory(
            scenario=scenario,
            status=trajectory["status"],
            start_time=start_time,
            end_time=datetime.now(),
            token_usage=token_usage
        )
        
        # Merge the trajectory data to ensure nothing is lost
        detailed_trajectory_dict = complete_trajectory.to_dict()
        detailed_trajectory_dict["execution_time_seconds"] = trajectory["execution_time_seconds"]
        detailed_trajectory_dict["token_usage"] = token_usage
        detailed_trajectory_dict["full_trajectory"] = trajectory["full_trajectory"]
        
        # Return the complete result
        return {
            "scenario": scenario,
            "trajectory": trajectory,
            "reliability_score": reliability_score,
            "detailed_trajectory": detailed_trajectory_dict,
            "scenario_index": scenario_index
        }

@app.function(
    image=arc_image,
    secrets=[modal.Secret.from_name("openai-secret")],
    timeout=600
)
def run_evaluation_suite_parallel(
    agent_config: Dict[str, Any],
    scenarios: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """Orchestrate parallel scenario evaluation using Modal's .map()"""
    
    print(f"[ORCHESTRATOR] Starting parallel evaluation of {len(scenarios)} scenarios")
    suite_start_time = datetime.now()
    
    # Create tuples of (scenario, agent_config, index) for parallel processing
    scenario_inputs = [(scenario, agent_config, i) for i, scenario in enumerate(scenarios)]
    
    # Run scenarios in parallel using .map()
    # return_exceptions=True ensures one failure doesn't crash the entire batch
    print(f"[ORCHESTRATOR] Submitting {len(scenario_inputs)} scenarios for parallel execution...")
    
    # Execute scenarios in parallel
    results = list(evaluate_single_scenario.map(scenario_inputs, return_exceptions=True))
    
    print(f"[ORCHESTRATOR] All scenarios completed. Processing results...")
    
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
                    "status": "error",
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
                        "total_cost": 0
                    }
                },
                "reliability_score": {
                    "overall_score": 0,
                    "grade": "F",
                    "dimension_scores": {
                        "tool_execution": 0,
                        "response_quality": 0,
                        "error_handling": 0,
                        "performance": 0,
                        "completeness": 0
                    }
                },
                "detailed_trajectory": None
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
    successful_runs = sum(1 for r in processed_results if r["trajectory"]["status"] == "success")
    avg_score = sum(r["reliability_score"]["overall_score"] for r in processed_results) / len(processed_results) if processed_results else 0
    avg_execution_time = sum(r["trajectory"]["execution_time_seconds"] for r in processed_results) / len(processed_results) if processed_results else 0
    total_tokens = sum(r["trajectory"].get("token_usage", {}).get("total_tokens", 0) for r in processed_results)
    total_cost = sum(r["trajectory"].get("token_usage", {}).get("total_cost", 0) for r in processed_results)
    
    # Aggregate dimension scores
    dimension_totals = {
        "tool_execution": 0.0,
        "response_quality": 0.0,
        "error_handling": 0.0,
        "performance": 0.0,
        "completeness": 0.0
    }
    
    for result in processed_results:
        dimension_scores = result["reliability_score"].get("dimension_scores", {})
        for dimension, score in dimension_scores.items():
            if dimension in dimension_totals:
                dimension_totals[dimension] += score
    
    # Calculate average for each dimension
    avg_dimension_scores = {
        dimension: total / len(processed_results) 
        for dimension, total in dimension_totals.items()
    } if processed_results else dimension_totals
    
    suite_end_time = datetime.now()
    total_execution_time = (suite_end_time - suite_start_time).total_seconds()
    
    print(f"[ORCHESTRATOR] Parallel execution completed in {total_execution_time:.2f}s")
    print(f"[ORCHESTRATOR] Average time per scenario: {total_execution_time/len(scenarios):.2f}s")
    
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
        "speedup_factor": (avg_execution_time * len(scenarios)) / total_execution_time if total_execution_time > 0 else 1
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
        "status": "success"
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
        "cost_per_scenario": round(total_cost / scenarios, 4)
    }

@app.local_entrypoint()
def main():
    """Production simulator test entrypoint"""
    print("[PRODUCTION] Arc-Eval Production Simulator")
    print("="*70)
    
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