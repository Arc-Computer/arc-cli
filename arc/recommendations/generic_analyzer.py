"""
Generic Agent Analyzer

This analyzer works with ANY agent YAML configuration by using actual simulation data
to guide LLM recommendations instead of hardcoded patterns.
"""

import logging
import json
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import asyncio

logger = logging.getLogger(__name__)


@dataclass
class SimulationContext:
    """Complete context from a simulation run."""
    original_yaml: Dict[str, Any]
    scenario_data: List[Dict[str, Any]]
    execution_trajectories: List[Dict[str, Any]]
    failure_data: List[Dict[str, Any]]
    success_data: List[Dict[str, Any]]
    performance_metrics: Dict[str, Any]


@dataclass
class GenericRecommendation:
    """LLM-generated recommendation based on actual data."""
    issue_description: str
    root_cause_analysis: str
    recommended_yaml_changes: str
    expected_impact: str
    confidence_score: float  # 0.0 to 1.0
    evidence: Dict[str, Any]


class GenericAgentAnalyzer:
    """
    Generic analyzer that works with any agent configuration.
    Uses actual simulation data to guide LLM recommendations.
    """
    
    def __init__(self, llm_client=None):
        self.llm_client = llm_client
        
    async def analyze_agent_failures(self, simulation_context: SimulationContext) -> List[GenericRecommendation]:
        """
        Analyze agent failures using actual simulation data and LLM.
        Works with any agent configuration.
        """
        if not self.llm_client:
            logger.warning("No LLM client available - falling back to basic analysis")
            return self._basic_analysis_fallback(simulation_context)
        
        try:
            # Prepare comprehensive analysis data
            analysis_data = self._prepare_analysis_data(simulation_context)
            
            # Generate LLM recommendations
            recommendations = await self._generate_llm_recommendations(analysis_data)
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error in generic analysis: {e}")
            return self._basic_analysis_fallback(simulation_context)
    
    def _prepare_analysis_data(self, context: SimulationContext) -> Dict[str, Any]:
        """Prepare comprehensive analysis data for LLM."""
        
        # Extract pattern insights from actual data
        failure_patterns = self._extract_failure_patterns(context.failure_data, context.execution_trajectories)
        success_patterns = self._extract_success_patterns(context.success_data, context.execution_trajectories)
        
        # Analyze what the agent was supposed to do vs what it actually did
        expectation_gaps = self._analyze_expectation_gaps(context.scenario_data, context.execution_trajectories)
        
        return {
            "original_configuration": context.original_yaml,
            "performance_summary": context.performance_metrics,
            "failure_patterns": failure_patterns,
            "success_patterns": success_patterns,
            "expectation_gaps": expectation_gaps,
            "scenario_characteristics": self._characterize_scenarios(context.scenario_data),
            "execution_insights": self._extract_execution_insights(context.execution_trajectories)
        }
    
    def _extract_failure_patterns(self, failures: List[Dict], trajectories: List[Dict]) -> Dict[str, Any]:
        """Extract patterns from actual failure data."""
        patterns = {
            "common_errors": {},
            "failure_points": [],
            "tool_issues": [],
            "reasoning_failures": [],
            "execution_issues": []
        }
        
        for i, failure in enumerate(failures):
            trajectory = trajectories[i] if i < len(trajectories) else {}
            
            # Count error types
            error_msg = failure.get("error_message", "").lower()
            patterns["common_errors"][error_msg] = patterns["common_errors"].get(error_msg, 0) + 1
            
            # Analyze where failures occurred
            events = trajectory.get("full_trajectory", [])
            if events:
                failure_point = len(events)  # How far the agent got
                patterns["failure_points"].append(failure_point)
                
                # Check for tool-related issues
                tool_events = [e for e in events if e.get("type") == "tool_call"]
                if tool_events:
                    last_tool = tool_events[-1]
                    if "error" in str(last_tool.get("tool_output", "")).lower():
                        patterns["tool_issues"].append({
                            "tool": last_tool.get("tool"),
                            "error": last_tool.get("tool_output"),
                            "scenario": failure.get("scenario_name")
                        })
                
                # Check for reasoning issues
                reasoning_events = [e for e in events if e.get("type") == "reasoning"]
                if reasoning_events and len(reasoning_events) < 3:  # Very little reasoning
                    patterns["reasoning_failures"].append(failure.get("scenario_name"))
            else:
                # No events = immediate failure
                patterns["execution_issues"].append(failure.get("scenario_name"))
        
        return patterns
    
    def _extract_success_patterns(self, successes: List[Dict], trajectories: List[Dict]) -> Dict[str, Any]:
        """Extract patterns from successful executions."""
        patterns = {
            "successful_approaches": [],
            "effective_tools": {},
            "reasoning_depth": [],
            "execution_lengths": []
        }
        
        for i, success in enumerate(successes):
            trajectory = trajectories[i] if i < len(trajectories) else {}
            events = trajectory.get("full_trajectory", [])
            
            if events:
                # Track execution length
                patterns["execution_lengths"].append(len(events))
                
                # Track reasoning depth
                reasoning_count = len([e for e in events if e.get("type") == "reasoning"])
                patterns["reasoning_depth"].append(reasoning_count)
                
                # Track effective tools
                tool_events = [e for e in events if e.get("type") == "tool_call"]
                for tool_event in tool_events:
                    tool_name = tool_event.get("tool")
                    if tool_name:
                        patterns["effective_tools"][tool_name] = patterns["effective_tools"].get(tool_name, 0) + 1
                
                # Extract approach pattern
                approach = {
                    "scenario": success.get("scenario_name"),
                    "reasoning_steps": reasoning_count,
                    "tools_used": [e.get("tool") for e in tool_events],
                    "execution_length": len(events)
                }
                patterns["successful_approaches"].append(approach)
        
        return patterns
    
    def _analyze_expectation_gaps(self, scenarios: List[Dict], trajectories: List[Dict]) -> List[Dict]:
        """Analyze gaps between what was expected and what actually happened."""
        gaps = []
        
        for i, scenario in enumerate(scenarios):
            trajectory = trajectories[i] if i < len(trajectories) else {}
            
            expected_tools = scenario.get("expected_tools", [])
            actual_tools = self._extract_tools_from_trajectory(trajectory)
            
            gap = {
                "scenario": scenario.get("name", f"scenario_{i}"),
                "expected_tools": expected_tools,
                "actual_tools": actual_tools,
                "missing_tools": [t for t in expected_tools if t not in actual_tools],
                "unexpected_tools": [t for t in actual_tools if t not in expected_tools],
                "expected_complexity": scenario.get("complexity_level", "unknown"),
                "actual_execution_length": len(trajectory.get("full_trajectory", [])),
                "task_prompt": scenario.get("task_prompt", "")
            }
            
            gaps.append(gap)
        
        return gaps
    
    def _extract_tools_from_trajectory(self, trajectory: Dict) -> List[str]:
        """Extract tools actually used from trajectory."""
        events = trajectory.get("full_trajectory", [])
        tools = []
        
        for event in events:
            if event.get("type") == "tool_call":
                tool_name = event.get("tool")
                if tool_name and tool_name not in tools:
                    tools.append(tool_name)
        
        return tools
    
    def _characterize_scenarios(self, scenarios: List[Dict]) -> Dict[str, Any]:
        """Characterize the types of scenarios being tested."""
        characteristics = {
            "domains": {},
            "complexity_levels": {},
            "task_types": {},
            "tool_requirements": {}
        }
        
        for scenario in scenarios:
            domain = scenario.get("inferred_domain", "unknown")
            characteristics["domains"][domain] = characteristics["domains"].get(domain, 0) + 1
            
            complexity = scenario.get("complexity_level", "unknown")
            characteristics["complexity_levels"][complexity] = characteristics["complexity_levels"].get(complexity, 0) + 1
            
            # Infer task type from prompt
            task_prompt = scenario.get("task_prompt", "").lower()
            task_type = self._infer_task_type(task_prompt)
            characteristics["task_types"][task_type] = characteristics["task_types"].get(task_type, 0) + 1
            
            # Track tool requirements
            expected_tools = scenario.get("expected_tools", [])
            for tool in expected_tools:
                characteristics["tool_requirements"][tool] = characteristics["tool_requirements"].get(tool, 0) + 1
        
        return characteristics
    
    def _infer_task_type(self, task_prompt: str) -> str:
        """Infer task type from prompt text."""
        prompt_lower = task_prompt.lower()
        
        if any(word in prompt_lower for word in ["search", "find", "lookup", "query"]):
            return "search_task"
        elif any(word in prompt_lower for word in ["calculate", "compute", "math", "number"]):
            return "calculation_task"
        elif any(word in prompt_lower for word in ["analyze", "parse", "process", "extract"]):
            return "analysis_task"
        elif any(word in prompt_lower for word in ["write", "create", "generate", "compose"]):
            return "generation_task"
        else:
            return "general_task"
    
    def _extract_execution_insights(self, trajectories: List[Dict]) -> Dict[str, Any]:
        """Extract insights from execution patterns."""
        insights = {
            "avg_execution_length": 0,
            "reasoning_patterns": {},
            "tool_usage_patterns": {},
            "common_failure_points": []
        }
        
        if not trajectories:
            return insights
        
        # Calculate average execution length
        lengths = []
        reasoning_counts = []
        
        for trajectory in trajectories:
            events = trajectory.get("full_trajectory", [])
            lengths.append(len(events))
            
            reasoning_events = [e for e in events if e.get("type") == "reasoning"]
            reasoning_counts.append(len(reasoning_events))
        
        if lengths:
            insights["avg_execution_length"] = sum(lengths) / len(lengths)
        
        if reasoning_counts:
            insights["avg_reasoning_steps"] = sum(reasoning_counts) / len(reasoning_counts)
        
        return insights
    
    async def _generate_llm_recommendations(self, analysis_data: Dict[str, Any]) -> List[GenericRecommendation]:
        """Generate recommendations using LLM based on actual data."""
        
        prompt = self._create_comprehensive_analysis_prompt(analysis_data)
        
        try:
            logger.info(f"Making LLM call with prompt length: {len(prompt)} characters")
            response = await self.llm_client.generate(prompt)
            logger.info(f"LLM response received: {len(response)} characters")
            recommendations = self._parse_llm_recommendations(response)
            logger.info(f"Parsed {len(recommendations)} recommendations")
            return recommendations
            
        except Exception as e:
            logger.error(f"Error generating LLM recommendations: {e}", exc_info=True)
            return []
    
    def _create_comprehensive_analysis_prompt(self, analysis_data: Dict[str, Any]) -> str:
        """Create a comprehensive prompt for LLM analysis."""
        
        original_config = analysis_data["original_configuration"]
        failure_patterns = analysis_data["failure_patterns"]
        success_patterns = analysis_data["success_patterns"]
        expectation_gaps = analysis_data["expectation_gaps"]
        
        # Convert datetime objects to strings for JSON serialization
        def convert_datetime_to_string(obj):
            if isinstance(obj, dict):
                return {k: convert_datetime_to_string(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_datetime_to_string(item) for item in obj]
            elif hasattr(obj, 'isoformat'):  # datetime objects
                return obj.isoformat()
            else:
                return obj
        
        performance_summary = convert_datetime_to_string(analysis_data.get('performance_summary', {}))
        
        prompt = f"""
You are analyzing an AI agent's performance based on ACTUAL SIMULATION DATA. Your task is to provide specific, actionable YAML configuration improvements based on the real data patterns observed.

CURRENT AGENT CONFIGURATION:
```yaml
{self._yaml_to_string(original_config)}
```

ACTUAL SIMULATION RESULTS:

FAILURE PATTERNS OBSERVED:
- Common Errors: {json.dumps(failure_patterns.get('common_errors', {}), indent=2)}
- Tool Issues: {json.dumps(failure_patterns.get('tool_issues', []), indent=2)}
- Reasoning Failures: {failure_patterns.get('reasoning_failures', [])}
- Execution Issues: {failure_patterns.get('execution_issues', [])}

SUCCESS PATTERNS OBSERVED:
- Effective Tools: {json.dumps(success_patterns.get('effective_tools', {}), indent=2)}
- Average Reasoning Depth: {sum(success_patterns.get('reasoning_depth', [0])) / max(len(success_patterns.get('reasoning_depth', [1])), 1)}
- Successful Approaches: {len(success_patterns.get('successful_approaches', []))}

EXPECTATION VS REALITY GAPS:
{self._format_expectation_gaps(expectation_gaps)}

PERFORMANCE METRICS:
{json.dumps(performance_summary, indent=2)}

TASK: Generate specific YAML configuration improvements that directly address the observed issues. Each recommendation should:

1. Be based on ACTUAL PATTERNS from the simulation data
2. Include SPECIFIC YAML changes (not generic advice)
3. Address ROOT CAUSES identified in the data
4. Be tailored to THIS specific agent configuration
5. Include confidence score based on evidence strength

RESPONSE FORMAT (JSON):
```json
[
  {{
    "issue_description": "Specific issue observed in the data",
    "root_cause_analysis": "Why this issue is happening based on actual evidence",
    "recommended_yaml_changes": "Exact YAML to add/modify with proper formatting",
    "expected_impact": "Specific improvements expected and why",
    "confidence_score": 0.0-1.0,
    "evidence": {{
      "failure_count": number,
      "success_rate_impact": "percentage",
      "specific_scenarios_affected": ["list"]
    }}
  }}
]
```

Focus on CONCRETE, DATA-DRIVEN recommendations, not generic best practices.
"""
        
        return prompt
    
    def _yaml_to_string(self, yaml_dict: Dict) -> str:
        """Convert YAML dict to string format."""
        try:
            import yaml
            return yaml.dump(yaml_dict, default_flow_style=False, indent=2)
        except ImportError:
            # Fallback to JSON if PyYAML not available
            return json.dumps(yaml_dict, indent=2)
    
    def _format_expectation_gaps(self, gaps: List[Dict]) -> str:
        """Format expectation gaps for the prompt."""
        formatted = []
        for gap in gaps[:5]:  # Limit to top 5 to avoid prompt bloat
            formatted.append(f"""
Scenario: {gap['scenario']}
- Expected tools: {gap['expected_tools']}
- Actually used: {gap['actual_tools']}
- Missing tools: {gap['missing_tools']}
- Task: {gap['task_prompt'][:100]}...
""")
        return "\n".join(formatted)
    
    def _parse_llm_recommendations(self, response: str) -> List[GenericRecommendation]:
        """Parse LLM response into structured recommendations."""
        try:
            # Extract JSON from response
            import re
            json_match = re.search(r'\[.*\]', response, re.DOTALL)
            if json_match:
                recommendations_data = json.loads(json_match.group())
                
                recommendations = []
                for rec_data in recommendations_data:
                    recommendation = GenericRecommendation(
                        issue_description=rec_data.get("issue_description", ""),
                        root_cause_analysis=rec_data.get("root_cause_analysis", ""),
                        recommended_yaml_changes=rec_data.get("recommended_yaml_changes", ""),
                        expected_impact=rec_data.get("expected_impact", ""),
                        confidence_score=rec_data.get("confidence_score", 0.5),
                        evidence=rec_data.get("evidence", {})
                    )
                    recommendations.append(recommendation)
                
                return recommendations
            else:
                logger.warning("Could not parse JSON from LLM response")
                return []
                
        except Exception as e:
            logger.error(f"Error parsing LLM recommendations: {e}")
            return []
    
    def _basic_analysis_fallback(self, context: SimulationContext) -> List[GenericRecommendation]:
        """Basic analysis when LLM is not available."""
        recommendations = []
        
        # Simple pattern-based analysis
        failure_count = len(context.failure_data)
        success_count = len(context.success_data)
        
        if failure_count > 0:
            # Generic recommendation based on failure rate
            recommendation = GenericRecommendation(
                issue_description=f"Agent has {failure_count} failures out of {failure_count + success_count} runs",
                root_cause_analysis="Analysis requires LLM client for detailed insights",
                recommended_yaml_changes="# LLM client needed for specific recommendations",
                expected_impact="Detailed analysis requires LLM integration",
                confidence_score=0.3,
                evidence={"failure_count": failure_count, "total_runs": failure_count + success_count}
            )
            recommendations.append(recommendation)
        
        return recommendations 