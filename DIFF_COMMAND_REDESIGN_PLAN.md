# Arc Diff Command Redesign Plan

## Problem Statement

The current `arc diff` command is fundamentally flawed:
1. It re-executes scenarios instead of comparing stored results
2. It has a critical bug calling non-existent `_execute_with_modal()` function  
3. It creates unfair comparisons by using different execution environments for each config
4. It doesn't follow the established pattern of other Arc analysis commands

## Proposed Solution

Transform `arc diff` into a pure analysis command that compares stored run results, similar to `arc analyze` and `arc recommend`.

## Implementation Plan

### Phase 1: Remove Execution Logic
- Remove all scenario generation code
- Remove Modal execution attempts
- Remove simulation execution code
- Keep only the statistical analysis logic

### Phase 2: Add Run Comparison Logic

#### CLI Interface Options:
```bash
# Option 1: Compare two specific runs by ID
arc diff run_20241105_143022 run_20241105_145510

# Option 2: Compare latest runs of two configs
arc diff finance_agent_v1.yaml finance_agent_v2.yaml

# Option 3: Interactive selection
arc diff --interactive
```

#### Core Logic:
1. **Load Run Results**
   - Accept run IDs directly, or
   - Find most recent runs for given config paths
   - Load full results including scenarios and outcomes

2. **Validate Comparability**
   - Check if runs used similar scenarios
   - Option to filter to only common scenarios
   - Warn if scenario sets are very different

3. **Statistical Analysis**
   - Keep existing statistical comparison logic
   - Add scenario-level comparison (which scenarios failed in A but passed in B)
   - Identify patterns in failures

4. **Display Results**
   - Keep existing display format
   - Add scenario-level insights
   - Show which specific capabilities improved/degraded

### Phase 3: Enhanced Features

1. **Historical Tracking**
   - Store diff results for future reference
   - Track improvement trends over time
   - Identify regression patterns

2. **Scenario Matching**
   - Compare only scenarios that appear in both runs
   - Weight comparison by scenario difficulty
   - Group by scenario type/domain

3. **Detailed Analysis**
   - Show specific scenarios where behavior differs
   - Identify assumption violations that changed
   - Highlight capability improvements

## Benefits

1. **Simplicity**: No complex execution logic, just data analysis
2. **Reliability**: No Modal authentication issues or execution failures
3. **Efficiency**: Instant comparison of existing results
4. **Fairness**: Compares runs that were executed identically
5. **Flexibility**: Can compare any historical runs

## Migration Path

1. Create new implementation in parallel
2. Mark old execution-based diff as deprecated
3. Provide clear migration instructions
4. Remove old implementation in next major version

## Example Implementation Flow

```python
# Simplified diff command flow
async def diff(config1_or_run1, config2_or_run2):
    # 1. Resolve to run IDs
    run1 = resolve_to_run_id(config1_or_run1)
    run2 = resolve_to_run_id(config2_or_run2)
    
    # 2. Load stored results
    results1 = state.get_run(run1)
    results2 = state.get_run(run2)
    
    # 3. Find common scenarios
    common_scenarios = find_common_scenarios(results1, results2)
    
    # 4. Compare results on common scenarios
    comparison = compare_results(results1, results2, common_scenarios)
    
    # 5. Perform statistical analysis
    stats = analyze_statistical_significance(comparison)
    
    # 6. Display results
    display_comparison(comparison, stats)
```

## Success Criteria

1. ✅ No execution code in diff command
2. ✅ Works with any stored runs
3. ✅ Fair comparison of identical scenarios
4. ✅ Clear statistical significance testing
5. ✅ Helpful insights about capability differences
6. ✅ No Modal authentication issues
7. ✅ Fast execution (< 1 second for analysis)