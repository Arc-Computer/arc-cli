"""Advanced failure analysis using OpenAI o3 background mode."""

import asyncio
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import json

from arc.core.utils.openai_client import get_background_client, BackgroundTaskStatus

logger = logging.getLogger(__name__)


@dataclass
class FailureTrace:
    """Represents a single failure with full context."""
    scenario_id: str
    failure_reason: str
    agent_config: Dict[str, Any]
    tools_used: List[str]
    execution_time: float
    cost: float
    context: Dict[str, Any]


@dataclass
class AdvancedAnalysisResult:
    """Result of o3 advanced failure analysis."""
    task_id: str
    root_causes: List[Dict[str, Any]]
    systemic_patterns: List[Dict[str, Any]]
    minimal_reproductions: List[Dict[str, Any]]
    infrastructure_recommendations: List[Dict[str, Any]]
    confidence_score: float
    processing_time: float
    analysis_cost: float


class AdvancedFailureAnalyzer:
    """Advanced failure pattern analysis using o3 background mode."""
    
    def __init__(self, model: str = "o3", cost_limit: float = 0.50):
        """Initialize the advanced analyzer.
        
        Args:
            model: o3 model to use (o3 or o3-mini)
            cost_limit: Maximum cost per analysis in USD
        """
        self.model = model
        self.cost_limit = cost_limit
        self.client = get_background_client()
    
    async def deep_failure_analysis(
        self, 
        failures: List[FailureTrace], 
        agent_config: Dict[str, Any],
        clustering_confidence: float = 0.0
    ) -> Optional[AdvancedAnalysisResult]:
        """Perform deep failure analysis using o3 background mode.
        
        Args:
            failures: List of failure traces to analyze
            agent_config: Agent configuration for context
            clustering_confidence: Confidence score from previous clustering
            
        Returns:
            Advanced analysis result or None if not applicable
        """
        # Only run deep analysis for complex failures
        if not self._should_run_deep_analysis(failures, agent_config, clustering_confidence):
            logger.info("Skipping deep analysis - criteria not met")
            return None
        
        try:
            # Create analysis prompt
            prompt = self._create_analysis_prompt(failures, agent_config)
            
            # Estimate cost
            estimated_cost = self.client._estimate_cost(prompt, self.model)
            if estimated_cost > self.cost_limit:
                logger.warning(f"Deep analysis cost ${estimated_cost:.3f} exceeds limit ${self.cost_limit}")
                return None
            
            logger.info(f"Starting deep analysis with o3 (estimated cost: ${estimated_cost:.3f})")
            
            # Start background task
            task_id = await self.client.create_background_analysis(
                prompt=prompt,
                model=self.model
            )
            
            # Poll for completion
            task = await self.client.poll_task(task_id, poll_interval=10)
            
            if task.status == BackgroundTaskStatus.COMPLETED:
                return self._parse_analysis_result(task)
            else:
                logger.error(f"Deep analysis failed: {task.error}")
                return None
                
        except Exception as e:
            logger.error(f"Deep analysis error: {str(e)}")
            return None
    
    def _should_run_deep_analysis(
        self, 
        failures: List[FailureTrace], 
        agent_config: Dict[str, Any],
        clustering_confidence: float
    ) -> bool:
        """Determine if deep analysis should be triggered.
        
        Args:
            failures: List of failures
            agent_config: Agent configuration
            clustering_confidence: Confidence from clustering
            
        Returns:
            True if deep analysis is warranted
        """
        # Check failure complexity
        if len(failures) < 3:
            return False
        
        # Check tool usage complexity
        tools_used = set()
        for failure in failures:
            tools_used.update(failure.tools_used)
        
        if len(tools_used) < 3:
            return False
        
        # Check clustering confidence
        if clustering_confidence > 0.70:
            return False
        
        # Check for multi-step failures
        multi_step_failures = sum(1 for f in failures if len(f.tools_used) > 2)
        if multi_step_failures < 2:
            return False
        
        logger.info(f"Deep analysis criteria met: {len(failures)} failures, "
                   f"{len(tools_used)} tools, {clustering_confidence:.2f} confidence")
        return True
    
    def _create_analysis_prompt(self, failures: List[FailureTrace], agent_config: Dict[str, Any]) -> str:
        """Create comprehensive analysis prompt for o3.
        
        Args:
            failures: List of failure traces
            agent_config: Agent configuration
            
        Returns:
            Formatted prompt for o3 analysis
        """
        # Serialize failures for analysis
        failure_data = []
        for f in failures:
            failure_data.append({
                "scenario_id": f.scenario_id,
                "failure_reason": f.failure_reason,
                "tools_used": f.tools_used,
                "execution_time": f.execution_time,
                "cost": f.cost,
                "context": f.context
            })
        
        prompt = f"""# Advanced Multi-Tool Agent Failure Analysis

You are an expert in analyzing complex multi-tool agent failures. Perform a comprehensive analysis of the following failure patterns, focusing on systemic issues and infrastructure improvements.

## Agent Configuration
```json
{json.dumps(agent_config, indent=2)}
```

## Failure Data ({len(failures)} failures)
```json
{json.dumps(failure_data, indent=2)}
```

## Analysis Requirements

Please provide a thorough analysis with the following sections:

### 1. Root Cause Analysis
Identify the fundamental causes of these failures, looking beyond surface-level error messages. Consider:
- Tool boundary issues and interaction patterns
- State management problems across tool invocations
- Assumption violations in multi-step workflows
- Resource contention and timing issues

### 2. Systemic Patterns
Identify patterns that suggest systemic issues rather than isolated problems:
- Cross-tool dependencies that create failure cascades
- Configuration assumptions that break under certain conditions
- Infrastructure limitations affecting multiple scenarios

### 3. Minimal Reproductions
For each major pattern identified, provide:
- The simplest scenario that would reproduce the issue
- Specific tool sequences and parameters
- Environmental conditions required
- Expected vs actual behavior

### 4. Infrastructure Recommendations
Following the "Infrastructure Over Intelligence" philosophy, suggest concrete improvements:
- Tool configuration changes
- Error handling mechanisms
- Monitoring and observability enhancements
- Timeout and retry strategies
- Resource allocation adjustments

### 5. Confidence Assessment
Rate your confidence in each finding (0.0-1.0) and explain the reasoning.

## Output Format
Provide your analysis as a structured JSON response with the following schema:

```json
{{
  "root_causes": [
    {{
      "title": "Cause title",
      "description": "Detailed explanation",
      "affected_scenarios": ["scenario_1", "scenario_2"],
      "confidence": 0.85
    }}
  ],
  "systemic_patterns": [
    {{
      "pattern_name": "Pattern name",
      "description": "Pattern description",
      "frequency": "high|medium|low",
      "impact": "critical|major|minor",
      "evidence": ["evidence_1", "evidence_2"]
    }}
  ],
  "minimal_reproductions": [
    {{
      "pattern_id": "pattern_1",
      "scenario": {{
        "description": "Scenario description",
        "tools": ["tool1", "tool2"],
        "steps": ["step1", "step2"],
        "expected_failure": "Expected failure description"
      }},
      "simplified_from": ["original_scenario_1", "original_scenario_2"]
    }}
  ],
  "infrastructure_recommendations": [
    {{
      "category": "error_handling|monitoring|configuration|performance",
      "title": "Recommendation title",
      "description": "Detailed recommendation",
      "implementation_effort": "low|medium|high",
      "impact": "high|medium|low",
      "priority": "critical|high|medium|low"
    }}
  ],
  "overall_confidence": 0.75,
  "analysis_summary": "Brief summary of key findings"
}}
```

Focus on actionable insights that will prevent similar failures across different scenarios and configurations."""

        return prompt
    
    def _parse_analysis_result(self, task) -> AdvancedAnalysisResult:
        """Parse o3 analysis result into structured format.
        
        Args:
            task: Completed background task
            
        Returns:
            Structured analysis result
        """
        try:
            content = task.result["content"]
            
            # Extract JSON from content (o3 might include additional text)
            json_start = content.find('{')
            json_end = content.rfind('}') + 1
            
            if json_start == -1 or json_end == 0:
                raise ValueError("No JSON found in analysis result")
            
            json_content = content[json_start:json_end]
            analysis_data = json.loads(json_content)
            
            # Calculate actual cost if usage data available
            actual_cost = task.cost_estimate
            if task.result.get("usage"):
                usage = task.result["usage"]
                if self.model == "o3":
                    actual_cost = (
                        (usage.get("prompt_tokens", 0) / 1_000_000) * 2.00 +
                        (usage.get("completion_tokens", 0) / 1_000_000) * 8.00
                    )
                # Add o3-mini pricing when available
            
            return AdvancedAnalysisResult(
                task_id=task.task_id,
                root_causes=analysis_data.get("root_causes", []),
                systemic_patterns=analysis_data.get("systemic_patterns", []),
                minimal_reproductions=analysis_data.get("minimal_reproductions", []),
                infrastructure_recommendations=analysis_data.get("infrastructure_recommendations", []),
                confidence_score=analysis_data.get("overall_confidence", 0.0),
                processing_time=task.completed_at - task.created_at,
                analysis_cost=actual_cost
            )
            
        except Exception as e:
            logger.error(f"Failed to parse analysis result: {str(e)}")
            # Return partial result with raw content
            return AdvancedAnalysisResult(
                task_id=task.task_id,
                root_causes=[{
                    "title": "Analysis Parsing Error",
                    "description": f"Could not parse o3 analysis: {str(e)}",
                    "affected_scenarios": [],
                    "confidence": 0.1
                }],
                systemic_patterns=[],
                minimal_reproductions=[],
                infrastructure_recommendations=[],
                confidence_score=0.1,
                processing_time=task.completed_at - task.created_at,
                analysis_cost=task.cost_estimate
            )


async def generate_advanced_scenarios(
    agent_config: Dict[str, Any], 
    count: int = 10,
    model: str = "o3"
) -> List[Dict[str, Any]]:
    """Generate sophisticated edge cases using o3 reasoning.
    
    Args:
        agent_config: Agent configuration to generate scenarios for
        count: Number of scenarios to generate
        model: Model to use for generation
        
    Returns:
        List of generated scenarios
    """
    client = get_background_client()
    
    prompt = f"""# Advanced Agent Scenario Generation

Generate {count} sophisticated test scenarios that expose hidden assumptions and edge cases for the following agent configuration.

## Agent Configuration
```json
{json.dumps(agent_config, indent=2)}
```

## Generation Requirements

Create scenarios that test:
1. **Assumption violations** - situations where implicit assumptions break
2. **Multi-tool boundary conditions** - complex interactions between different tools
3. **Resource constraint scenarios** - performance under pressure
4. **Temporal edge cases** - timing-dependent failures
5. **State corruption scenarios** - situations that could corrupt agent state

Each scenario should be designed to expose potential failure modes that simple testing might miss.

## Output Format

Provide scenarios as a JSON array:

```json
[
  {{
    "id": "scenario_1",
    "title": "Scenario title",
    "description": "Detailed scenario description",
    "category": "assumption_violation|multi_tool|resource_constraint|temporal|state_corruption",
    "complexity": "low|medium|high",
    "tools_involved": ["tool1", "tool2"],
    "expected_challenges": ["challenge1", "challenge2"],
    "success_criteria": "How to determine if agent handles this correctly",
    "failure_indicators": ["indicator1", "indicator2"]
  }}
]
```

Focus on scenarios that would be difficult for current testing approaches to discover."""

    try:
        task_id = await client.create_background_analysis(
            prompt=prompt,
            model=model
        )
        
        task = await client.poll_task(task_id)
        
        if task.status == BackgroundTaskStatus.COMPLETED:
            content = task.result["content"]
            
            # Extract JSON
            json_start = content.find('[')
            json_end = content.rfind(']') + 1
            
            if json_start != -1 and json_end > 0:
                json_content = content[json_start:json_end]
                scenarios = json.loads(json_content)
                logger.info(f"Generated {len(scenarios)} advanced scenarios")
                return scenarios
            
        logger.error(f"Scenario generation failed: {task.error}")
        return []
        
    except Exception as e:
        logger.error(f"Advanced scenario generation error: {str(e)}")
        return []