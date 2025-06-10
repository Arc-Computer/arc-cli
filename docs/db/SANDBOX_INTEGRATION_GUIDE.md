# Arc Database Integration Guide for Modal Sandbox

This guide shows how to integrate the Arc database API with your Modal sandbox implementation.

## Quick Start

### 1. Installation

```python
from arc.database.api import create_arc_api, ArcAPI
from arc.database.client import ArcDBClient
```

### 2. Initialize the API

```python
# In your Modal function or at startup
db_api = await create_arc_api()  # Uses TIMESCALE_SERVICE_URL env var

# Or with explicit connection string
db_client = ArcDBClient("your-connection-string")
await db_client.initialize()
db_api = ArcAPI(db_client)
```

## Integration Pattern for Modal Sandbox

### Complete Example: Integrating with `run_evaluation_suite_parallel`

```python
import modal
from arc.database.api import create_arc_api
from datetime import datetime

@modal.function(
    image=arc_image,
    secrets=[
        modal.Secret.from_name("openai-secret"),
        modal.Secret.from_name("timescale-secret")  # Contains TIMESCALE_SERVICE_URL
    ],
    timeout=600
)
async def run_evaluation_suite_with_db(
    agent_config: Dict[str, Any],
    scenarios: List[Dict[str, Any]],
    config_version_id: str  # Required for database tracking
) -> Dict[str, Any]:
    """Run evaluation suite with database integration."""
    
    # 1. Initialize database API
    db_api = await create_arc_api()
    
    # 2. Start simulation tracking
    simulation_info = await db_api.start_simulation(
        config_version_id=config_version_id,
        scenarios=scenarios,
        simulation_name=f"eval_{agent_config['model']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        modal_app_id=modal.current_app_id(),
        modal_environment=os.getenv("MODAL_ENVIRONMENT", "production"),
        sandbox_instances=len(scenarios),  # Assuming parallel execution
        metadata={
            "agent_model": agent_config["model"],
            "agent_temperature": agent_config["temperature"],
            "tools_enabled": agent_config.get("tools", [])
        }
    )
    
    simulation_id = simulation_info["simulation_id"]
    print(f"[DB] Started simulation {simulation_id}")
    
    try:
        # 3. Run the actual evaluation (your existing code)
        suite_result = run_evaluation_suite_parallel(agent_config, scenarios)
        
        # 4. Record all outcomes in batch
        modal_call_ids = [f"{modal.current_call_id()}_{i}" for i in range(len(scenarios))]
        outcome_ids = await db_api.record_batch_outcomes(
            simulation_id=simulation_id,
            scenario_results=suite_result["results"],
            modal_call_ids=modal_call_ids
        )
        
        print(f"[DB] Recorded {len(outcome_ids)} outcomes")
        
        # 5. Complete the simulation
        completion_info = await db_api.complete_simulation(
            simulation_id=simulation_id,
            suite_result=suite_result
        )
        
        print(f"[DB] Simulation completed with score {completion_info['overall_score']:.2f}")
        
        # 6. Add simulation info to results
        suite_result["simulation_id"] = simulation_id
        suite_result["database_tracking"] = {
            "simulation_id": simulation_id,
            "outcome_ids": outcome_ids,
            "completion_status": completion_info
        }
        
        return suite_result
        
    except Exception as e:
        # Mark simulation as failed
        await db_api.db.engine.execute(
            db_api.db.engine.text(
                "UPDATE simulations SET status = 'failed', completed_at = NOW() WHERE simulation_id = :id"
            ),
            {"id": simulation_id}
        )
        raise
```

### Individual Scenario Recording

If you want to record scenarios as they complete (for real-time monitoring):

```python
@modal.function(
    image=arc_image,
    secrets=[modal.Secret.from_name("openai-secret"), modal.Secret.from_name("timescale-secret")],
    timeout=300
)
async def evaluate_single_scenario_with_db(
    scenario_with_config_and_sim: Tuple[Dict[str, Any], Dict[str, Any], int, str]
) -> Dict[str, Any]:
    """Evaluate a single scenario with real-time database recording."""
    
    # Unpack the tuple (scenario, agent_config, index, simulation_id)
    scenario, agent_config, scenario_index, simulation_id = scenario_with_config_and_sim
    
    # Initialize DB API
    db_api = await create_arc_api()
    
    # Run evaluation (your existing code)
    result = evaluate_single_scenario((scenario, agent_config, scenario_index))
    
    # Record outcome immediately
    outcome_id = await db_api.record_scenario_outcome(
        simulation_id=simulation_id,
        scenario_result=result,
        modal_call_id=modal.current_call_id(),
        sandbox_id=f"sandbox_{scenario_index}"
    )
    
    # Add tracking info to result
    result["database_tracking"] = {
        "outcome_id": outcome_id,
        "simulation_id": simulation_id,
        "recorded_at": datetime.utcnow().isoformat()
    }
    
    return result
```

## Data Structure Alignment

The API is designed to work directly with your existing data structures:

### Input: Scenario Result from `evaluate_single_scenario`

```python
{
    "scenario": {
        "id": "scenario_123",
        "name": "Weather Query Test",
        "task_prompt": "What's the weather in NYC?",
        # ... other scenario fields
    },
    "trajectory": {
        "status": "success",  # or "error"
        "task_prompt": "What's the weather in NYC?",
        "final_response": "The weather in NYC is...",
        "execution_time_seconds": 2.5,
        "full_trajectory": [...],  # Tool calls and responses
        "token_usage": {
            "prompt_tokens": 150,
            "completion_tokens": 50,
            "total_tokens": 200,
            "total_cost": 0.006
        }
    },
    "reliability_score": {
        "overall_score": 0.85,
        "grade": "B",
        "dimension_scores": {
            "tool_execution": 0.9,
            "response_quality": 0.8,
            "error_handling": 0.85,
            "performance": 0.9,
            "completeness": 0.8
        }
    },
    "detailed_trajectory": {
        # TrajectoryData object converted to dict
    }
}
```

### Output: What Gets Stored

The API automatically extracts and stores:

1. **Simulation Record**:
   - Configuration version ID
   - Scenario list
   - Modal app ID and environment
   - Start/end times
   - Overall metrics

2. **Outcome Records** (one per scenario):
   - Execution status and timing
   - Reliability scores (overall and per dimension)
   - Token usage and costs
   - Complete trajectory data
   - Error information (if failed)
   - Modal call ID for debugging

3. **Simulation-Scenario Mappings**:
   - Execution order
   - Individual scenario status
   - Modal call IDs

## Query Examples

### Check Simulation Progress

```python
# Get real-time progress of a running simulation
status = await db_api.get_simulation_status(simulation_id)
print(f"Progress: {status['progress_percentage']:.1f}%")
print(f"Successful: {status['successful_outcomes']}")
print(f"Failed: {status['failed_outcomes']}")
```

### Retrieve Scenario Outcomes

```python
# Get all outcomes for a simulation
outcomes = await db_api.get_scenario_outcomes(simulation_id)

# Get only failed scenarios
failures = await db_api.get_scenario_outcomes(
    simulation_id, 
    status_filter="error"
)

# Get specific scenario outcome
outcome = await db_api.get_scenario_outcomes(
    simulation_id,
    scenario_id="scenario_123"
)
```

### Performance Analytics

```python
# Get performance metrics over time (using the client directly)
perf = await db_api.db.get_simulation_performance(
    simulation_id,
    time_range=timedelta(hours=24)
)

# Get recent failures across all simulations
failures = await db_api.db.get_recent_failures(limit=100)
```

## Best Practices

### 1. Error Handling

```python
try:
    outcome_id = await db_api.record_scenario_outcome(...)
except DatabaseError as e:
    logger.error(f"Failed to record outcome: {e}")
    # Continue execution - don't let DB errors crash evaluations
```

### 2. Batch Operations

For high-throughput scenarios, use batch recording:

```python
# Collect results first
results = list(evaluate_single_scenario.map(scenario_inputs))

# Then batch record
outcome_ids = await db_api.record_batch_outcomes(
    simulation_id=simulation_id,
    scenario_results=results
)
```

### 3. Connection Management

The API handles connection pooling automatically, but for long-running Modal functions:

```python
# Close connections when done
await db_api.db.close()
```

### 4. Monitoring

Enable monitoring to track database performance:

```python
# Get performance metrics
metrics = await db_api.db.get_metrics()
print(f"Success rate: {metrics['success_rate']:.1f}%")
print(f"Avg query time: {metrics['avg_query_time']:.1f}ms")
```

## Environment Variables

Required environment variables:

```bash
# TimescaleDB connection string
TIMESCALE_SERVICE_URL=postgresql://user:pass@host:port/dbname?sslmode=require

# Optional: Modal environment
MODAL_ENVIRONMENT=production  # or staging, development
```

## Troubleshooting

### Connection Issues

```python
# Test database connection
db_client = ArcDBClient()
health = await db_client.initialize()
print(health)  # Should show extensions, hypertables, pool stats
```

### Schema Deployment

```python
# Deploy/update schema if needed
result = await db_client.deploy_schema()
print(result)
```

### Debug Logging

```python
# Enable debug logging
import logging
logging.getLogger("arc.database").setLevel(logging.DEBUG)
```

## Support

For issues or questions:
1. Check the database logs
2. Verify environment variables
3. Test connection with health check
4. Review the error categories in recorded outcomes 