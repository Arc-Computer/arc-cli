# Trajectory Data Requirements

## Overview

The `outcomes` table in TimescaleDB has a `trajectory` JSONB field that stores execution trace data. This field has specific constraints that must be met for successful database insertion.

## Required Fields

The database enforces the following constraint:

```sql
CONSTRAINT valid_trajectory CHECK (
    trajectory ? 'start_time' AND 
    trajectory ? 'status'
)
```

### Mandatory Fields

| Field | Type | Description | Example |
|-------|------|-------------|---------|
| `start_time` | String (ISO 8601) | When the scenario execution began | `"2024-06-10T14:30:00Z"` |
| `status` | String | Final execution status | `"success"`, `"error"`, `"timeout"` |

## Valid Status Values

The `status` field must match one of these values (enforced by table-level constraint):

- ✅ `"success"` - Scenario completed successfully
- ❌ `"error"` - Scenario failed with an error
- ⏱️ `"timeout"` - Scenario exceeded time limit
- 🚫 `"cancelled"` - Scenario was cancelled

## Example Trajectory Structure

### Minimal Valid Trajectory

```json
{
  "start_time": "2024-06-10T14:30:00Z",
  "status": "success"
}
```

### Complete Trajectory Example

```json
{
  "start_time": "2024-06-10T14:30:00Z",
  "status": "success",
  "task_prompt": "Calculate the total cost of items in the shopping cart",
  "final_response": "The total cost is $45.67",
  "execution_time_seconds": 2.5,
  "full_trajectory": [
    {
      "step": 1,
      "action": "read_cart",
      "timestamp": "2024-06-10T14:30:00.100Z",
      "result": "Found 3 items"
    },
    {
      "step": 2,
      "action": "calculate_total",
      "timestamp": "2024-06-10T14:30:02.400Z",
      "result": "$45.67"
    }
  ],
  "token_usage": {
    "prompt_tokens": 150,
    "completion_tokens": 75,
    "total_tokens": 225,
    "total_cost": 0.0045
  },
  "tools_used": ["cart_reader", "calculator"],
  "error_message": null
}
```

## Common Validation Errors

### Missing Required Fields

```json
// ❌ INVALID - Missing start_time
{
  "status": "success",
  "final_response": "Task completed"
}

// ❌ INVALID - Missing status  
{
  "start_time": "2024-06-10T14:30:00Z",
  "final_response": "Task completed"
}
```

### Invalid Status Values

```json
// ❌ INVALID - Invalid status value
{
  "start_time": "2024-06-10T14:30:00Z", 
  "status": "completed"  // Should be "success"
}
```

## Database Integration Notes

### Modal Sandbox Integration

When using the ArcAPI for Modal sandbox integration, trajectories are automatically constructed with required fields:

```python
from arc.database.api import ArcAPI

api = ArcAPI(db_client)

# This automatically ensures trajectory has required fields
outcome_id = await api.record_scenario_outcome(
    simulation_id="sim_123",
    scenario_result={
        "scenario": {"id": "test_001"},
        "trajectory": {
            "start_time": "2024-06-10T14:30:00Z",  # ✅ Required
            "status": "success",                    # ✅ Required
            "task_prompt": "Test scenario",
            "final_response": "Success"
        }
    }
)
```

### Manual Data Insertion

When inserting outcomes manually, ensure trajectory compliance:

```python
outcome_data = {
    "simulation_id": simulation_id,
    "scenario_id": "test_001", 
    "status": "success",
    "reliability_score": 0.95,
    "trajectory": {
        "start_time": datetime.now(timezone.utc).isoformat(),  # ✅ Required
        "status": "success",                                   # ✅ Required
        "task_prompt": "Manual test",
        "final_response": "Completed successfully"
    }
    # ... other fields
}

outcome_id = await db_client.record_outcome(outcome_data)
```

## Troubleshooting

### Error: "CHECK constraint violation"

This error occurs when the trajectory is missing required fields:

```
ERROR: new row for relation "outcomes" violates check constraint "valid_trajectory"
DETAIL: Failing row contains trajectory missing 'start_time' or 'status'
```

**Solution**: Ensure your trajectory JSON includes both `start_time` and `status` fields.

### Error: "Invalid status value"

This error occurs when the status field contains an invalid value:

```
ERROR: new row for relation "outcomes" violates check constraint 
DETAIL: Value must be one of: success, error, timeout, cancelled
```

**Solution**: Use only the allowed status values listed above.

## Best Practices

1. **Always include timestamps** in ISO 8601 format with timezone
2. **Use descriptive error messages** when status is "error" 
3. **Include execution metrics** for performance analysis
4. **Store full trajectory** for debugging and analysis
5. **Validate trajectory structure** before database insertion

## Schema Evolution

If you need to modify trajectory requirements:

1. Update the constraint in `tables.sql`
2. Update this documentation
3. Update validation in `arc/database/client.py`
4. Test with existing data to ensure compatibility