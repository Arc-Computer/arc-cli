# Failure Pattern Library

This directory contains a comprehensive library of AI agent failure patterns based on insights from the Who&When dataset and Microsoft's taxonomy of failure modes in agentic AI systems.

## Overview

Total patterns: 29 across 9 categories

## Categories

### Data (5 patterns)
- `empty_results.json` - Valid but empty API responses
- `malformed_response.json` - Malformed JSON/XML responses
- `schema_mismatch.json` - Response doesn't match expected schema
- `parsing_type_confusion.json` - Type confusion in data parsing
- `encoding_mismatch.json` - Character encoding issues

### Infrastructure (4 patterns)
- `rate_limiting.json` - API rate limit exceeded
- `service_outage.json` - Backend service unavailable
- `timeout_api_delay.json` - Request timeout due to slow response
- `network_partition.json` - Partial network connectivity
- `resource_exhaustion.json` - System resource limits reached

### Security (4 patterns)
- `memory_poisoning.json` - Malicious data corrupts agent memory
- `tool_compromise.json` - Compromised tool returns malicious data
- `xpia_injection.json` - Cross-domain prompt injection
- `privilege_escalation.json` - Unintended privilege escalation

### Logic (4 patterns)
- `ambiguous_request.json` - Unclear user requirements
- `circular_dependency.json` - Tools have circular dependencies
- `state_corruption.json` - Agent state becomes inconsistent
- `race_condition.json` - Concurrent operations conflict

### Navigation (3 patterns)
- `tool_misuse.json` - Agent uses tool incorrectly
- `tool_hallucination.json` - Agent assumes non-existent capabilities
- `tool_version_mismatch.json` - API version incompatibility

### Multi-Agent (2 patterns)
- `agent_flow_manipulation.json` - One agent manipulates another
- `coordination_deadlock.json` - Agents deadlock waiting for each other

### Calculation (2 patterns)
- `precision_loss.json` - Floating point precision errors
- `overflow_error.json` - Integer overflow in calculations

### Temporal (2 patterns)
- `timezone_confusion.json` - Timezone handling errors
- `time_sync_drift.json` - System clock synchronization issues

### Auth (2 patterns)
- `token_expiry_race.json` - Token expires during operation
- `permission_boundary_violation.json` - Cross-tenant access attempts

## Pattern Schema

Each pattern follows this schema:

```json
{
  "id": "unique_identifier",
  "title": "Human readable title",
  "category": "category_name",
  "severity": "low|medium|high|critical",
  "trigger_conditions": ["condition1", "condition2"],
  "expected_error": "Error message or behavior",
  "recovery_patterns": ["recovery1", "recovery2"],
  "frequency": "rare|uncommon|common|very_common",
  "description": "Detailed description",
  "example_instantiation": "Concrete example"
}
```

## High-Variance Failure Families

Based on analysis, these failure types show high variance and are particularly valuable for testing:

1. **Temporal failures** - Timezone and time synchronization issues
2. **Auth failures** - Token management and permission boundaries
3. **Data failures** - Type confusion and encoding problems
4. **Calculation failures** - Precision and overflow errors

## Usage

These patterns are designed to be used by the scenario generator to create realistic test cases that expose real agent vulnerabilities. Each pattern provides enough detail to instantiate concrete scenarios while remaining agent-agnostic.

## Insights from Who&When Dataset

- Data parsing errors are the most common failure mode
- Early step failures (steps 0-2) indicate fundamental understanding issues
- Multi-agent coordination failures are less common but have high severity
- Time-based failures are very common but often overlooked in testing