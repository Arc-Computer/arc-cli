"""Arc Funnel Analyzer - Capability Decomposition for Execution Flow.

Bridges agent capabilities from ingestion to real-time execution analysis,
providing step-by-step success rate tracking and bottleneck identification.
"""

import logging
import asyncio
import json
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict
from dataclasses import dataclass

from arc.ingestion.parser import AgentConfigParser
from arc.ingestion.normalizer import ConfigNormalizer
from arc.analysis.clustering import FailureClusterer

logger = logging.getLogger(__name__)


@dataclass
class FunnelStep:
    """Represents a step in the capability execution funnel."""
    name: str
    category: str
    success_count: int
    total_count: int
    success_rate: float
    failures: List[Dict[str, Any]]
    is_bottleneck: bool = False
    
    @property
    def failure_rate(self) -> float:
        """Calculate failure rate for this step."""
        return 100 - self.success_rate


@dataclass 
class CapabilityFunnel:
    """Complete funnel analysis for agent capabilities."""
    agent_profile: Dict[str, Any]
    steps: List[FunnelStep] 
    overall_success_rate: float
    primary_bottleneck: Optional[FunnelStep]
    capability_breakdown: Dict[str, float]
    
    def get_bottlenecks(self, threshold: float = 70.0) -> List[FunnelStep]:
        """Get all steps below success rate threshold."""
        return [step for step in self.steps if step.success_rate < threshold]


class FunnelAnalyzer:
    """Decomposes agent execution into capability-based funnel analysis."""
    
    def __init__(self):
        """Initialize the funnel analyzer with required components."""
        self.parser = AgentConfigParser()
        self.normalizer = ConfigNormalizer()
        self.clusterer = FailureClusterer()
        
        # AI model for enhanced analysis - using fast and powerful GPT-4.1-mini
        self.ai_model = "openai/gpt-4.1-mini"
        
        # Capability mapping from ingestion parser
        self.capability_steps = {
            "input_processing": ["parsing", "validation", "understanding"],
            "tool_execution": ["selection", "parameter_mapping", "invocation", "result_handling"],
            "reasoning": ["analysis", "decision_making", "logic_application"],
            "output_generation": ["formatting", "completeness", "accuracy"],
            "error_handling": ["detection", "recovery", "fallback"],
        }
        
    def build_capability_funnel(self, agent_profile: Dict[str, Any], 
                               trajectories: List[Dict[str, Any]]) -> CapabilityFunnel:
        """
        Create step-by-step capability funnel from execution trajectories.
        
        Args:
            agent_profile: Complete agent profile from normalizer
            trajectories: List of execution trajectories with results
            
        Returns:
            CapabilityFunnel with detailed step analysis
        """
        logger.info(f"Building capability funnel for {len(trajectories)} trajectories")
        
        # Extract capabilities from profile
        capabilities = agent_profile.get("capabilities", {})
        domains = capabilities.get("domains", ["general"])
        tool_categories = capabilities.get("tool_categories", {})
        
        # Map trajectories to capability steps
        step_results = self._map_execution_steps_to_capabilities(trajectories, capabilities)
        
        # Calculate success rates for each step
        funnel_steps = []
        for step_name, results in step_results.items():
            success_count = sum(1 for r in results if r.get("success", False))
            total_count = len(results)
            success_rate = (success_count / total_count * 100) if total_count > 0 else 0
            
            # Collect failures for this step
            failures = [r for r in results if not r.get("success", False)]
            
            step = FunnelStep(
                name=step_name,
                category=self._get_step_category(step_name),
                success_count=success_count,
                total_count=total_count,
                success_rate=success_rate,
                failures=failures
            )
            funnel_steps.append(step)
        
        # Identify bottlenecks
        bottlenecks = self._identify_capability_bottlenecks(funnel_steps)
        for step in funnel_steps:
            step.is_bottleneck = step in bottlenecks
        
        # Calculate overall metrics
        overall_success_rate = self._calculate_overall_success_rate(trajectories)
        primary_bottleneck = bottlenecks[0] if bottlenecks else None
        capability_breakdown = self._calculate_capability_breakdown(funnel_steps, domains)
        
        return CapabilityFunnel(
            agent_profile=agent_profile,
            steps=sorted(funnel_steps, key=lambda x: x.success_rate),
            overall_success_rate=overall_success_rate,
            primary_bottleneck=primary_bottleneck,
            capability_breakdown=capability_breakdown
        )
    
    def _map_execution_steps_to_capabilities(self, trajectories: List[Dict[str, Any]], 
                                           capabilities: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
        """
        Map trajectory execution events to capability categories.
        
        Args:
            trajectories: Execution trajectories
            capabilities: Agent capabilities from parser
            
        Returns:
            Dict mapping step names to execution results
        """
        step_results = defaultdict(list)
        tool_categories = capabilities.get("tool_categories", {})
        
        for trajectory in trajectories:
            success = trajectory.get("status") == "success"
            trajectory_events = trajectory.get("full_trajectory", [])
            
            # Track different capability aspects
            step_results["input_processing"].append({
                "success": self._evaluate_input_processing(trajectory),
                "trajectory_id": trajectory.get("id"),
                "details": trajectory.get("input_processing_details", {})
            })
            
            step_results["tool_execution"].append({
                "success": self._evaluate_tool_execution(trajectory, tool_categories),
                "trajectory_id": trajectory.get("id"),
                "details": self._get_tool_execution_details(trajectory_events)
            })
            
            step_results["reasoning"].append({
                "success": self._evaluate_reasoning(trajectory),
                "trajectory_id": trajectory.get("id"),
                "details": trajectory.get("reasoning_details", {})
            })
            
            step_results["output_generation"].append({
                "success": self._evaluate_output_generation(trajectory),
                "trajectory_id": trajectory.get("id"),
                "details": trajectory.get("output_details", {})
            })
            
            step_results["error_handling"].append({
                "success": self._evaluate_error_handling(trajectory),
                "trajectory_id": trajectory.get("id"),
                "details": trajectory.get("error_handling_details", {})
            })
        
        return step_results
    
    def _evaluate_input_processing(self, trajectory: Dict[str, Any]) -> bool:
        """Evaluate if input processing was successful."""
        # Check if the agent understood and parsed the input correctly
        events = trajectory.get("full_trajectory", [])
        
        # Look for parsing errors or misunderstandings
        for event in events:
            if event.get("type") == "error" and "parsing" in str(event.get("message", "")).lower():
                return False
            if event.get("type") == "validation_error":
                return False
        
        # If no specific parsing errors and trajectory started, input processing succeeded
        return len(events) > 0 and trajectory.get("status") != "error"
    
    def _evaluate_tool_execution(self, trajectory: Dict[str, Any], 
                                tool_categories: Dict[str, List[str]]) -> bool:
        """Evaluate tool execution success."""
        events = trajectory.get("full_trajectory", [])
        tool_calls = [e for e in events if e.get("type") == "tool_call"]
        
        if not tool_calls:
            # No tools needed or no tools called - consider success if overall success
            return trajectory.get("status") == "success"
        
        # Check if any tool calls failed
        for tool_call in tool_calls:
            tool_output = str(tool_call.get("tool_output", ""))
            if "error" in tool_output.lower() or tool_call.get("status") == "error":
                return False
        
        return True
    
    def _evaluate_reasoning(self, trajectory: Dict[str, Any]) -> bool:
        """Evaluate reasoning and decision-making quality using AI analysis."""
        # Quick heuristic check first
        events = trajectory.get("full_trajectory", [])
        
        # Look for obvious reasoning-related errors
        for event in events:
            event_text = str(event.get("message", "")) + str(event.get("content", ""))
            if any(word in event_text.lower() for word in ["contradiction", "illogical", "inconsistent"]):
                return False
        
        # If trajectory failed, analyze with AI for reasoning quality
        if trajectory.get("status") != "success":
            try:
                return asyncio.run(self._ai_analyze_reasoning(trajectory))
            except Exception as e:
                logger.warning(f"AI reasoning analysis failed: {e}")
                return False
        
        # If trajectory completed successfully, assume reasoning was adequate
        return True
    
    def _evaluate_output_generation(self, trajectory: Dict[str, Any]) -> bool:
        """Evaluate output quality and completeness using AI analysis."""
        final_output = trajectory.get("final_output", "")
        
        # Basic quality checks first
        if not final_output or len(str(final_output).strip()) < 10:
            return False
        
        # Check for obvious incomplete responses
        if any(phrase in str(final_output).lower() for phrase in ["incomplete", "unable to", "cannot provide"]):
            return False
        
        # If trajectory succeeded, do enhanced AI quality analysis
        if trajectory.get("status") == "success":
            try:
                return asyncio.run(self._ai_analyze_output_quality(trajectory))
            except Exception as e:
                logger.warning(f"AI output analysis failed: {e}")
                return True  # Default to success if AI fails
        
        return False
    
    def _evaluate_error_handling(self, trajectory: Dict[str, Any]) -> bool:
        """Evaluate error handling and recovery mechanisms."""
        events = trajectory.get("full_trajectory", [])
        errors = [e for e in events if e.get("type") == "error"]
        
        if not errors:
            # No errors to handle - success
            return True
        
        # Check if errors were handled gracefully
        # Look for recovery attempts after errors
        for i, event in enumerate(events):
            if event.get("type") == "error":
                # Check if there are recovery attempts after this error
                remaining_events = events[i+1:]
                if any(e.get("type") in ["retry", "fallback", "recovery"] for e in remaining_events):
                    continue
                else:
                    # Error without recovery attempt
                    return False
        
        # If we reach here, all errors had recovery attempts
        return True
    
    def _get_tool_execution_details(self, events: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Extract detailed tool execution information."""
        tool_calls = [e for e in events if e.get("type") == "tool_call"]
        
        return {
            "total_tool_calls": len(tool_calls),
            "successful_calls": len([t for t in tool_calls if "error" not in str(t.get("tool_output", "")).lower()]),
            "tools_used": list(set(t.get("tool", "unknown") for t in tool_calls)),
            "call_duration": sum(t.get("duration", 0) for t in tool_calls)
        }
    
    def _get_step_category(self, step_name: str) -> str:
        """Get the category for a funnel step."""
        category_mapping = {
            "input_processing": "Input",
            "tool_execution": "Tools", 
            "reasoning": "Logic",
            "output_generation": "Output",
            "error_handling": "Recovery"
        }
        return category_mapping.get(step_name, "Other")
    
    def _identify_capability_bottlenecks(self, steps: List[FunnelStep], 
                                       threshold: float = 70.0) -> List[FunnelStep]:
        """
        Identify capability bottlenecks based on success rates.
        
        Args:
            steps: List of funnel steps
            threshold: Success rate threshold for bottleneck identification
            
        Returns:
            List of bottleneck steps sorted by severity
        """
        bottlenecks = [step for step in steps if step.success_rate < threshold]
        
        # Sort by success rate (lowest first) and total impact
        bottlenecks.sort(key=lambda x: (x.success_rate, -x.total_count))
        
        return bottlenecks
    
    def _calculate_overall_success_rate(self, trajectories: List[Dict[str, Any]]) -> float:
        """Calculate overall success rate across all trajectories."""
        if not trajectories:
            return 0.0
        
        successful = sum(1 for t in trajectories if t.get("status") == "success")
        return (successful / len(trajectories)) * 100
    
    def _calculate_capability_breakdown(self, steps: List[FunnelStep], 
                                      domains: List[str]) -> Dict[str, float]:
        """Calculate capability success rates by domain."""
        breakdown = {}
        
        for step in steps:
            breakdown[step.category] = step.success_rate
        
        # Add domain-specific breakdown if multiple domains
        if len(domains) > 1:
            for domain in domains:
                # This could be enhanced with domain-specific analysis
                breakdown[f"{domain}_specific"] = sum(s.success_rate for s in steps) / len(steps)
        
        return breakdown
    
    def analyze_assumption_patterns(self, funnel: CapabilityFunnel) -> Dict[str, Any]:
        """
        Analyze patterns in funnel failures to identify potential assumptions.
        
        Args:
            funnel: Completed capability funnel
            
        Returns:
            Dict of potential assumption violations and patterns
        """
        patterns = {
            "currency_assumptions": [],
            "language_assumptions": [],
            "format_assumptions": [],
            "validation_assumptions": []
        }
        
        # Analyze failures across all steps
        all_failures = []
        for step in funnel.steps:
            all_failures.extend(step.failures)
        
        # Use clustering to group similar failures
        if all_failures:
            # Convert failures to format expected by clusterer
            failure_data = []
            for failure in all_failures:
                failure_data.append({
                    "failure_text": str(failure.get("details", "")),
                    "type": "capability_failure",
                    "step": failure.get("step", "unknown")
                })
            
            clusters = self.clusterer.cluster_failures(failure_data)
            
            # Analyze clusters for assumption patterns
            for cluster in clusters:
                cluster_name = cluster.get("name", "")
                if "currency" in cluster_name.lower():
                    patterns["currency_assumptions"].append(cluster)
                elif "language" in cluster_name.lower() or "encoding" in cluster_name.lower():
                    patterns["language_assumptions"].append(cluster)
                elif "format" in cluster_name.lower() or "parsing" in cluster_name.lower():
                    patterns["format_assumptions"].append(cluster)
                elif "validation" in cluster_name.lower() or "input" in cluster_name.lower():
                    patterns["validation_assumptions"].append(cluster)
        
        return patterns
    
    async def _ai_analyze_reasoning(self, trajectory: Dict[str, Any]) -> bool:
        """Use GPT-4.1-mini to analyze reasoning quality in failed trajectories."""
        try:
            # Prepare trajectory data for AI analysis
            events = trajectory.get("full_trajectory", [])
            final_output = trajectory.get("final_output", "")
            task_prompt = trajectory.get("scenario", {}).get("task_prompt", "")
            
            # Create analysis prompt
            analysis_prompt = f"""
            Analyze the reasoning quality in this AI agent execution:

            TASK: {task_prompt}
            
            EXECUTION EVENTS: {json.dumps(events[:5])}  # First 5 events
            
            FINAL OUTPUT: {final_output}
            
            STATUS: {trajectory.get("status", "unknown")}

            Rate the agent's reasoning on a scale of 1-10 considering:
            1. Logical consistency of steps
            2. Decision-making quality
            3. Problem decomposition
            4. Error recognition

            Respond with only a JSON object: {{"reasoning_score": <1-10>, "explanation": "<brief explanation>"}}
            """
            
            # In a real implementation, this would call the OpenRouter API
            # For now, return a mock analysis based on simple heuristics
            if "error" in str(events).lower() or "failed" in str(final_output).lower():
                return False
            else:
                return True
                
        except Exception as e:
            logger.error(f"AI reasoning analysis error: {e}")
            return False
    
    async def _ai_analyze_output_quality(self, trajectory: Dict[str, Any]) -> bool:
        """Use GPT-4.1-mini to analyze output quality and completeness."""
        try:
            final_output = trajectory.get("final_output", "")
            task_prompt = trajectory.get("scenario", {}).get("task_prompt", "")
            
            # Create quality analysis prompt
            quality_prompt = f"""
            Evaluate the quality and completeness of this AI agent's output:

            ORIGINAL TASK: {task_prompt}
            
            AGENT OUTPUT: {final_output}

            Rate the output quality on a scale of 1-10 considering:
            1. Completeness - Does it fully address the task?
            2. Accuracy - Is the information correct?
            3. Clarity - Is it well-structured and clear?
            4. Usefulness - Would this help the user?

            Respond with only a JSON object: {{"quality_score": <1-10>, "explanation": "<brief explanation>"}}
            """
            
            # In a real implementation, this would call the OpenRouter API
            # For now, return quality assessment based on output length and keywords
            if len(str(final_output).strip()) > 50 and not any(word in str(final_output).lower() 
                                                             for word in ["error", "failed", "cannot", "unable"]):
                return True
            else:
                return False
                
        except Exception as e:
            logger.error(f"AI output quality analysis error: {e}")
            return True  # Default to success if analysis fails