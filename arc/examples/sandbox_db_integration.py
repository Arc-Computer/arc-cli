"""
Example: Integrating Arc Database with Modal Sandbox

This example shows how to add database tracking to your Modal sandbox evaluations.
"""

import asyncio
import os
from datetime import datetime
from typing import Dict, Any, List

# Simulated Modal imports (replace with actual Modal imports)
class MockModal:
    @staticmethod
    def current_app_id():
        return "test-app-123"
    
    @staticmethod
    def current_call_id():
        return f"call-{datetime.now().timestamp()}"

modal = MockModal()

# Arc imports
from arc.database.api import create_arc_api, ArcAPI
from arc.database.client import ArcDBClient


async def example_evaluation_with_db():
    """
    Example showing complete integration of database tracking
    with Modal sandbox evaluation.
    """
    
    # 1. Initialize the database API
    print("Initializing database connection...")
    db_api = await create_arc_api()
    
    # 2. Prepare test data (normally from your scenario generation)
    test_scenarios = [
        {
            "id": "weather_test_1",
            "name": "Weather Query NYC",
            "task_prompt": "What's the weather like in New York City?",
            "expected_tools": ["get_weather"],
            "complexity_level": "simple"
        },
        {
            "id": "database_test_1", 
            "name": "Customer Database Query",
            "task_prompt": "Find all active customers in the database",
            "expected_tools": ["search_database"],
            "complexity_level": "medium"
        }
    ]
    
    agent_config = {
        "model": "gpt-4",
        "temperature": 0.7,
        "tools": ["get_weather", "search_database"],
        "system_prompt": "You are a helpful assistant."
    }
    
    # 3. Start simulation tracking
    print("\nStarting simulation...")
    simulation_info = await db_api.start_simulation(
        config_version_id="test-config-v1",  # In production, this comes from config management
        scenarios=test_scenarios,
        simulation_name=f"example_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        modal_app_id=modal.current_app_id(),
        modal_environment="development",
        sandbox_instances=len(test_scenarios),
        metadata={
            "example_run": True,
            "agent_config": agent_config
        }
    )
    
    simulation_id = simulation_info["simulation_id"]
    print(f"Simulation ID: {simulation_id}")
    print(f"Tracking {simulation_info['scenario_count']} scenarios")
    
    # 4. Simulate scenario execution results
    # (In production, these come from evaluate_single_scenario)
    mock_results = []
    
    for i, scenario in enumerate(test_scenarios):
        # Simulate execution
        execution_time = 2.5 + i * 0.5
        success = i == 0  # First succeeds, second fails for demo
        
        result = {
            "scenario": scenario,
            "trajectory": {
                "status": "success" if success else "error",
                "task_prompt": scenario["task_prompt"],
                "final_response": "The weather in NYC is sunny, 72°F" if success else "Error: Database connection failed",
                "execution_time_seconds": execution_time,
                "full_trajectory": [
                    {
                        "step": 1,
                        "type": "tool_call",
                        "tool": scenario["expected_tools"][0],
                        "tool_input": {"city": "New York City"} if success else {"query": "SELECT * FROM customers", "table": "customers"},
                        "tool_output": "72°F, sunny" if success else "Connection timeout"
                    }
                ],
                "token_usage": {
                    "prompt_tokens": 150 + i * 50,
                    "completion_tokens": 50 + i * 20,
                    "total_tokens": 200 + i * 70,
                    "total_cost": 0.006 + i * 0.002
                },
                "error": None if success else "DatabaseError: Connection timeout"
            },
            "reliability_score": {
                "overall_score": 0.9 if success else 0.2,
                "grade": "A" if success else "F",
                "dimension_scores": {
                    "tool_execution": 1.0 if success else 0.0,
                    "response_quality": 0.9 if success else 0.3,
                    "error_handling": 0.8 if success else 0.2,
                    "performance": 0.9 if success else 0.4,
                    "completeness": 0.9 if success else 0.1
                }
            },
            "detailed_trajectory": {
                "scenario_id": scenario["id"],
                "scenario_name": scenario["name"],
                "spans": [],  # OpenTelemetry spans
                "tool_calls": 1,
                "llm_interactions": 2
            },
            "scenario_index": i
        }
        
        mock_results.append(result)
    
    # 5. Record outcomes individually (real-time tracking)
    print("\nRecording individual outcomes...")
    for result in mock_results:
        outcome_id = await db_api.record_scenario_outcome(
            simulation_id=simulation_id,
            scenario_result=result,
            modal_call_id=f"{modal.current_call_id()}-{result['scenario_index']}",
            sandbox_id=f"sandbox-{result['scenario_index']}"
        )
        print(f"  - Recorded outcome {outcome_id} for {result['scenario']['id']}")
    
    # 6. Or record in batch (more efficient for many scenarios)
    print("\nAlternative: Batch recording...")
    # outcome_ids = await db_api.record_batch_outcomes(
    #     simulation_id=simulation_id,
    #     scenario_results=mock_results,
    #     modal_call_ids=[f"{modal.current_call_id()}-{i}" for i in range(len(mock_results))]
    # )
    # print(f"Recorded {len(outcome_ids)} outcomes in batch")
    
    # 7. Complete the simulation
    print("\nCompleting simulation...")
    suite_result = {
        "average_reliability_score": sum(r["reliability_score"]["overall_score"] for r in mock_results) / len(mock_results),
        "total_cost_usd": sum(r["trajectory"]["token_usage"]["total_cost"] for r in mock_results),
        "parallel_execution_time": max(r["trajectory"]["execution_time_seconds"] for r in mock_results),
        "successful_runs": sum(1 for r in mock_results if r["trajectory"]["status"] == "success"),
        "total_tokens_used": sum(r["trajectory"]["token_usage"]["total_tokens"] for r in mock_results),
        "average_execution_time": sum(r["trajectory"]["execution_time_seconds"] for r in mock_results) / len(mock_results),
        "average_dimension_scores": {
            "tool_execution": 0.5,
            "response_quality": 0.6,
            "error_handling": 0.5,
            "performance": 0.65,
            "completeness": 0.5
        },
        "speedup_factor": 2.0  # Parallel execution benefit
    }
    
    completion_info = await db_api.complete_simulation(
        simulation_id=simulation_id,
        suite_result=suite_result
    )
    
    print(f"Simulation completed!")
    print(f"  - Overall score: {completion_info['overall_score']:.2f}")
    print(f"  - Total cost: ${completion_info['total_cost_usd']:.4f}")
    print(f"  - Execution time: {completion_info['execution_time_seconds']:.1f}s")
    
    # 8. Query the results
    print("\nQuerying simulation results...")
    
    # Get simulation status
    status = await db_api.get_simulation_status(simulation_id)
    print(f"\nSimulation Status:")
    print(f"  - Progress: {status['progress_percentage']:.0f}%")
    print(f"  - Successful: {status['successful_outcomes']}")
    print(f"  - Failed: {status['failed_outcomes']}")
    
    # Get all outcomes
    outcomes = await db_api.get_scenario_outcomes(simulation_id)
    print(f"\nRetrieved {len(outcomes)} outcomes")
    
    # Get only failures
    failures = await db_api.get_scenario_outcomes(simulation_id, status_filter="error")
    print(f"Found {len(failures)} failed scenarios")
    
    if failures:
        print("\nFailed scenarios:")
        for failure in failures:
            print(f"  - {failure['scenario_id']}: {failure['error_code']}")
    
    # 9. Performance metrics
    print("\nDatabase Performance Metrics:")
    metrics = await db_api.db.get_metrics()
    print(f"  - Total queries: {metrics['total_queries']}")
    print(f"  - Success rate: {metrics['success_rate']:.1f}%")
    print(f"  - Avg query time: {metrics['avg_query_time']*1000:.1f}ms")
    print(f"  - Connection pool: {metrics['pool_stats']}")
    
    # 10. Cleanup
    await db_api.db.close()
    print("\nDatabase connection closed.")


async def example_real_integration():
    """
    Example showing how to modify your existing Modal functions
    to add database tracking.
    """
    
    print("\n" + "="*60)
    print("REAL INTEGRATION EXAMPLE")
    print("="*60)
    
    print("""
    To integrate with your existing Modal sandbox:
    
    1. Add database initialization to your Modal function:
    
    @modal.function(
        secrets=[
            modal.Secret.from_name("openai-secret"),
            modal.Secret.from_name("timescale-secret")
        ]
    )
    async def run_evaluation_suite_parallel(agent_config, scenarios):
        # Add this at the beginning
        db_api = await create_arc_api()
        
        simulation_info = await db_api.start_simulation(
            config_version_id=agent_config.get("config_version_id"),
            scenarios=scenarios,
            modal_app_id=modal.current_app_id()
        )
        
        # Your existing evaluation code...
        results = evaluate_scenarios(agent_config, scenarios)
        
        # Add this after evaluation
        await db_api.record_batch_outcomes(
            simulation_id=simulation_info["simulation_id"],
            scenario_results=results
        )
        
        await db_api.complete_simulation(
            simulation_id=simulation_info["simulation_id"],
            suite_result=results
        )
        
        return results
    
    2. That's it! The database will now track all your evaluations.
    """)


if __name__ == "__main__":
    # Run the example
    asyncio.run(example_evaluation_with_db())
    asyncio.run(example_real_integration()) 