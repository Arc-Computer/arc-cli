# Arc V1 Scenario Generation Strategy: Capability-Centric Approach

## Executive Summary

Arc's V1 scenario generation must shift from tool-based testing to capability-based evaluation. Instead of generating scenarios that test if "weather API returns null", we generate scenarios that test if an agent can "help users plan outdoor activities" under various conditions. This fundamental shift enables us to discover which agent assumptions are false and provide actionable improvements.

## Core Product Insight

**The right question isn't "does this tool work?" but "can this agent do the job it was hired for?"**

Our scenario generation strategy must:
1. Automatically discover agent capabilities from configuration
2. Decompose capabilities into testable assumptions
3. Generate scenarios that stress-test these assumptions
4. Track failure origins through capability breakdowns
5. Provide actionable fixes at the capability level

## V1 Architecture: Capability-Centric Pipeline

### Phase 1: Agent Capability Discovery

```python
# Input: agent_config.yaml
# Output: capability_profile.json

{
  "declared_capabilities": {
    "primary_job": "Travel planning assistant",
    "core_capabilities": [
      {
        "id": "weather_interpretation",
        "description": "Interpret weather data for activity planning",
        "required_tools": ["get_weather"],
        "assumptions": [
          "Users provide location context",
          "Weather preferences are inferable",
          "Activities have weather requirements"
        ]
      }
    ]
  },
  "inferred_capabilities": {
    "environment_observation": ["web_content", "api_responses"],
    "memory": "conversation_context",
    "tool_execution": ["weather", "search"],
    "decision_making": "recommendation_generation"
  }
}
```

### Phase 2: Assumption-Based Scenario Generation

Instead of static failure patterns, generate scenarios that test specific assumptions:

```yaml
scenario:
  capability_under_test: "weather_interpretation"
  assumption_violated: "Users provide location context"
  test_progression:
    - prompt: "Is it good weather for a picnic?"
      missing: ["location", "time"]
    - prompt: "Is it good weather for a picnic tomorrow?"
      missing: ["location"]
    - prompt: "Is it good weather for a picnic in [ambiguous: 'Paris']?"
      ambiguity: "Paris, France or Paris, Texas?"
  expected_behavior:
    optimal: "Agent requests clarification"
    acceptable: "Agent makes reasonable assumption with disclaimer"
    failure: "Agent provides answer without location"
```

### Phase 3: Multi-Level Failure Injection

Failures should be injected at multiple abstraction levels:

```python
failure_injection_levels = {
    "infrastructure": {
        "tool_timeout": "Weather API takes 30s to respond",
        "rate_limit": "Database allows only 1 query/minute"
    },
    "data": {
        "malformed_response": "Weather data missing temperature",
        "conflicting_data": "Two sources report different weather"
    },
    "capability": {
        "memory_overflow": "Context exceeds token limit",
        "observation_noise": "Webpage contains contradictory info"
    },
    "semantic": {
        "ambiguous_intent": "User says 'nice weather' (subjective)",
        "implicit_context": "User assumes agent knows their location"
    }
}
```

### Phase 4: Intelligent Scenario Clustering

Group scenarios by capability coverage, not tool usage:

```python
scenario_clusters = {
    "single_capability_isolation": [
        # Tests that isolate one capability
        "weather_only_scenarios",
        "memory_only_scenarios"
    ],
    "capability_interference": [
        # Tests where capabilities conflict
        "memory_vs_tool_trust",
        "observation_vs_prior_knowledge"
    ],
    "capability_degradation": [
        # Progressive failure scenarios
        "memory_fills_over_conversation",
        "tool_reliability_decreases"
    ],
    "capability_boundaries": [
        # Tests at the edge of capabilities
        "max_memory_capacity",
        "tool_combination_limits"
    ]
}
```

## Implementation Strategy

### 1. Capability Profile Generator

```python
class CapabilityProfiler:
    """Analyzes agent config to extract capabilities"""
    
    def analyze(self, agent_config: Dict) -> CapabilityProfile:
        # Extract declared capabilities from system prompt
        declared = self._parse_system_prompt(agent_config['system_prompt'])
        
        # Infer capabilities from tools
        tool_capabilities = self._map_tools_to_capabilities(agent_config['tools'])
        
        # Use LLM to identify implicit capabilities
        implicit = self._llm_capability_analysis(agent_config)
        
        # Decompose into testable assumptions
        assumptions = self._extract_assumptions(declared, tool_capabilities)
        
        return CapabilityProfile(declared, tool_capabilities, implicit, assumptions)
```

### 2. Assumption-Driven Scenario Generator

```python
class AssumptionScenarioGenerator:
    """Generates scenarios that test specific assumptions"""
    
    def generate_for_assumption(self, 
                               capability: Capability,
                               assumption: Assumption) -> List[Scenario]:
        # Generate base case where assumption holds
        base_case = self._create_valid_scenario(capability, assumption)
        
        # Generate variations that violate assumption
        violations = self._create_assumption_violations(assumption)
        
        # Generate edge cases at assumption boundaries  
        edge_cases = self._create_boundary_scenarios(assumption)
        
        # Add progressive complexity
        complex_cases = self._create_compound_violations(assumption)
        
        return [base_case] + violations + edge_cases + complex_cases
```

### 3. Failure Attribution Engine

```python
class FailureAttributor:
    """Traces failures back to capability breakdowns"""
    
    def attribute_failure(self, trace: ExecutionTrace) -> FailureAttribution:
        # Identify where expected behavior diverged
        divergence_point = self._find_divergence(trace)
        
        # Map to capability breakdown
        failed_capability = self._map_to_capability(divergence_point)
        
        # Identify violated assumption
        violated_assumption = self._identify_assumption(divergence_point, failed_capability)
        
        # Generate minimal reproduction
        minimal_repro = self._extract_minimal_repro(trace, divergence_point)
        
        return FailureAttribution(
            capability=failed_capability,
            assumption=violated_assumption,
            repro=minimal_repro,
            fix_suggestion=self._suggest_fix(failed_capability, violated_assumption)
        )
```

## Quality Metrics for V1

### 1. Capability Coverage Score
- % of identified capabilities with test scenarios
- Depth of assumption testing per capability
- Interaction coverage between capabilities

### 2. Failure Discovery Rate
- Unique failure modes discovered / scenarios run
- Novel failures (not in predefined patterns)
- Actionable failures (lead to concrete fixes)

### 3. Attribution Quality
- % of failures correctly traced to root cause
- Time to identify failure origin
- Minimal repro accuracy

### 4. Improvement Velocity
- Time from failure discovery to fix
- % of suggested fixes that work
- Reduction in failure rate after fixes

## V1 Production Roadmap

### Week 1-2: Capability Infrastructure
1. Build CapabilityProfiler for agent analysis
2. Define capability taxonomy (based on Microsoft research)
3. Create assumption extraction pipeline

### Week 3-4: Scenario Generation 2.0
1. Implement AssumptionScenarioGenerator
2. Migrate best patterns from experimentation
3. Build quality scoring for capability coverage

### Week 5-6: Attribution & Analytics
1. Implement FailureAttributor
2. Create capability-based clustering
3. Build actionable recommendation engine

### Week 7-8: Integration & Testing
1. Integrate with Modal execution pipeline
2. Connect to LLM judge for evaluation
3. End-to-end testing with real agents

## Multi-Agent Extensibility

The capability model naturally extends to multi-agent:

```yaml
multi_agent_scenario:
  agents:
    - id: "coordinator"
      capabilities: ["task_decomposition", "agent_orchestration"]
    - id: "researcher"  
      capabilities: ["information_gathering", "synthesis"]
  interaction_assumptions:
    - "Agents share consistent world model"
    - "Communication is lossless"
    - "Agents trust each other's outputs"
  failure_modes:
    - "Conflicting world models"
    - "Communication ambiguity"
    - "Trust violation"
```

## Success Criteria for V1

1. **10x Bug Discovery**: Find bugs that tool-based testing misses
2. **Root Cause Attribution**: 90% of failures traced to capability/assumption
3. **Actionable Fixes**: 70% of recommendations improve agent reliability
4. **Scenario Quality**: <5% false positive rate in failure detection

## Conclusion

By shifting from tool-centric to capability-centric scenario generation, Arc V1 will deliver on its promise of continuous improvement for AI systems. This approach:

- Discovers what agents are actually hired to do
- Tests the assumptions that underpin agent reliability
- Provides actionable fixes at the right abstraction level
- Scales naturally from single to multi-agent systems

The key insight: **Great evals don't just find bugs, they reveal which assumptions about agent capabilities are false.**