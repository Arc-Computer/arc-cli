# Arc-Eval Generation Integration Guide

## Overview

The generation system now has multiple components that work together to create high-quality test scenarios:

```bash
┌─────────────────────────────────────────────────────────────┐
│                    Enhanced Generator                       │
│  ┌─────────────────┐        ┌──────────────────────────┐    │
│  │ Synthetic       │        │ Pattern-Based Generator  │    │
│  │ Generator       │        │ ┌──────────────────────┐ │    │
│  │ (generator.py)  │        │ │ Pattern Library      │ │    │
│  └─────────────────┘        │ │ (failure patterns)   │ │    │
│                             │ └──────────────────────┘ │    │
│                             │ ┌──────────────────────┐ │    │
│                             │ │ Pattern Selector     │ │    │
│                             │ │ (Stage A)            │ │    │
│                             │ └──────────────────────┘ │    │
│                             │ ┌──────────────────────┐ │    │
│                             │ │ Scenario Instantiator│ │    │
│                             │ │ (Stage B)            │ │    │
│                             │ └──────────────────────┘ │    │
│                             └──────────────────────────┘    │
│  ┌─────────────────────────────────────────────────────┐    │
│  │           Quality Control Pipeline                  │    │
│  │  ┌──────────────────┐    ┌──────────────────────┐   │    │
│  │  │ Quality Scorer   │    │ Deduplicator         │   │    │
│  │  │ (min score: 3.0) │    │ (SHA256 hashing)     │   │    │
│  │  └──────────────────┘    └──────────────────────┘   │    │
│  └─────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
```

## File Structure

```
generation/
├── generator.py                 # Original LLM-based generator
├── enhanced_generator.py        # New hybrid generator (main entry point)
├── pattern_based_generator.py   # Two-stage pattern-based generation
├── scenario_quality_scorer.py   # Quality scoring and deduplication
├── failure_patterns/           # 31 failure patterns across 9 categories
├── tool_behavior_profiles.json # Tool response profiles
└── patch_trajectory_capture.py # Failure capture instrumentation
```

## Usage Options

### 1. Drop-in Replacement (Recommended)

Use `enhanced_generator.py` as a drop-in replacement for the original generator:

```python
from enhanced_generator import generate_high_quality_scenarios

# Generate 150 high-quality scenarios
scenarios = await generate_high_quality_scenarios(
    agent_config_path="../config/agent_config.yaml",
    count=150,
    use_patterns=True,      # Enable pattern-based generation
    pattern_ratio=0.7,      # 70% pattern-based, 30% pure LLM
    quality_threshold=3.0,  # Minimum quality score
    output_path="generated_scenarios.json"
)
```

### 2. Direct Pattern-Based Generation

For pure pattern-based generation:

```python
from pattern_based_generator import PatternBasedGenerator

generator = PatternBasedGenerator("../config/agent_config.yaml")
scenarios = await generator.generate(
    total_scenarios=100,
    patterns_per_batch=3,
    scenarios_per_pattern=5
)
```

### 3. Original Generator (Backward Compatible)

The original generator still works as before:

```python
from generator import ScenarioGenerator

generator = ScenarioGenerator("../config/agent_config.yaml")
scenarios = await generator.generate_scenarios_batch(count=100)
```

### 4. Quality Scoring Only

Apply quality scoring to existing scenarios:

```python
from scenario_quality_scorer import ScenarioQualityScorer

scorer = ScenarioQualityScorer(min_threshold=3.0)
passed, failed = scorer.score_batch(scenarios)
```

## Integration with Existing Code

### run_scenarios_in_sandbox.py

The enhanced generator maintains the same output format, so it works seamlessly:

```python
# No changes needed - just use enhanced generator output
python run_scenarios_in_sandbox.py enhanced_scenarios.json
```

### multi_model_tester.py

Enhanced scenarios include pattern metadata for better analysis:

```python
# Scenarios now include:
# - pattern_id: Which failure pattern was used
# - quality_score: Scenario quality metric
# - expected_error: Specific error description
```

## Key Benefits

1. **Higher Failure Discovery Rate**: Pattern-based scenarios target known failure modes
2. **Better Quality**: Quality scoring ensures only good scenarios are used
3. **No Duplicates**: SHA256 deduplication prevents redundant testing
4. **Cost Efficient**: Two-stage generation reduces API costs
5. **Backward Compatible**: Works with all existing tools

## Metrics Tracking

The enhanced generator provides comprehensive metrics:

```json
{
  "total_generated": 150,
  "generation_method": "hybrid",
  "patterns_used": 12,
  "average_quality_score": 4.2,
  "duplicates_removed": 5,
  "scenarios_rejected": 23,
  "generation_cost": "$0.0135",
  "cost_per_scenario": "$0.0001"
}
```

## Best Practices

1. **Pattern Ratio**: Use 70-80% pattern-based for optimal results
2. **Quality Threshold**: Keep at 3.0 or higher
3. **Batch Size**: Use smaller batches (3-5 patterns) for diversity
4. **Pattern Selection**: Let the system auto-select based on agent tools

## Extending the System

### Adding New Failure Patterns

1. Create JSON file in appropriate category folder
2. Follow the schema with severity field
3. Include specific trigger conditions

### Custom Quality Metrics

Extend `ScenarioQualityScorer` to add domain-specific quality checks:

```python
class CustomScorer(ScenarioQualityScorer):
    def _score_domain_specific(self, scenario):
        # Add custom scoring logic
        return score
```

## Validation

To validate the improvements:

```bash
# Generate scenarios with enhanced generator
python enhanced_generator.py --count 150 --output test_scenarios.json

# Run validation metrics
python validate_improvements.py test_scenarios.json
```

Expected improvements:
- Failure rate: 2.3% → 10-15%
- Cost per scenario: $0.0002 → $0.0001
- Duplicate rate: <5%
- Domain accuracy: 70%+