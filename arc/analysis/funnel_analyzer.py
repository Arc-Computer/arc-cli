"""Arc Funnel Analyzer - Capability Decomposition for Execution Flow.

Bridges agent capabilities from ingestion to real-time execution analysis,
providing step-by-step success rate tracking and bottleneck identification.
"""

import logging
import asyncio
import json
import os
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache
import hashlib
import aiohttp

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
        
        # Cache for AI analysis results
        self._ai_cache = {}
        
        # Capability mapping from ingestion parser
        self.capability_steps = {
            "input_processing": ["parsing", "validation", "understanding"],
            "tool_execution": ["selection", "parameter_mapping", "invocation", "result_handling"],
            "reasoning": ["analysis", "decision_making", "logic_application"],
            "output_generation": ["formatting", "completeness", "accuracy"],
            "error_handling": ["detection", "recovery", "fallback"],
        }
    
    def _get_trajectory_hash(self, trajectory: Dict[str, Any], analysis_type: str) -> str:
        """Generate a hash key for caching trajectory analysis."""
        # Create a deterministic hash from key trajectory elements
        key_elements = {
            "type": analysis_type,
            "status": trajectory.get("status"),
            "task": trajectory.get("scenario", {}).get("task_prompt", "")[:100],
            "output": str(trajectory.get("final_output", ""))[:100],
            "event_count": len(trajectory.get("full_trajectory", []))
        }
        key_str = json.dumps(key_elements, sort_keys=True)
        return hashlib.md5(key_str.encode()).hexdigest()
        
    async def build_capability_funnel(self, agent_profile: Dict[str, Any], 
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
        step_results = await self._map_execution_steps_to_capabilities(trajectories, capabilities)
        
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
    
    async def _map_execution_steps_to_capabilities(self, trajectories: List[Dict[str, Any]], 
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
                "success": await self._evaluate_reasoning(trajectory),
                "trajectory_id": trajectory.get("id"),
                "details": trajectory.get("reasoning_details", {})
            })
            
            step_results["output_generation"].append({
                "success": await self._evaluate_output_generation(trajectory),
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
    
    async def _evaluate_reasoning(self, trajectory: Dict[str, Any]) -> bool:
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
                return await self._ai_analyze_reasoning(trajectory)
            except Exception as e:
                logger.warning(f"AI reasoning analysis failed: {e}")
                return False
        
        # If trajectory completed successfully, assume reasoning was adequate
        return True
    
    async def _evaluate_output_generation(self, trajectory: Dict[str, Any]) -> bool:
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
                return await self._ai_analyze_output_quality(trajectory)
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
    
    def _summarize_trajectory(self, events: List[Dict[str, Any]], max_events: int = 5) -> List[Dict[str, Any]]:
        """Summarize trajectory to reduce token usage while preserving key information."""
        if len(events) <= max_events:
            return events
        
        # Prioritize error events and tool calls
        error_events = [e for e in events if e.get("type") == "error"]
        tool_events = [e for e in events if e.get("type") == "tool_call"]
        message_events = [e for e in events if e.get("type") == "message"]
        
        # Build summary with most important events
        summary = []
        
        # Always include first and last events for context
        if events:
            summary.append(events[0])
        
        # Add all errors (most important for analysis)
        summary.extend(error_events[:2])
        
        # Add sample tool calls
        summary.extend(tool_events[:2])
        
        # Fill remaining slots with messages
        remaining_slots = max_events - len(summary) - 1  # -1 for last event
        if remaining_slots > 0:
            summary.extend(message_events[:remaining_slots])
        
        # Always include last event
        if len(events) > 1 and events[-1] not in summary:
            summary.append(events[-1])
        
        # Sort by original order
        summary.sort(key=lambda e: events.index(e) if e in events else 0)
        
        return summary[:max_events]
    
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
        # Check cache first
        cache_key = self._get_trajectory_hash(trajectory, "reasoning")
        if cache_key in self._ai_cache:
            logger.info(f"Using cached reasoning analysis for {cache_key}")
            return self._ai_cache[cache_key]
        
        try:
            # Prepare trajectory data for AI analysis
            events = trajectory.get("full_trajectory", [])
            summarized_events = self._summarize_trajectory(events, max_events=5)
            final_output = trajectory.get("final_output", "")
            task_prompt = trajectory.get("scenario", {}).get("task_prompt", "")
            
            # Create analysis prompt with few-shot examples
            analysis_prompt = f"""
            Analyze the reasoning quality in this AI agent execution.
            
            Examples:
            Task: "Calculate the total cost of 5 items at $10 each"
            Events: [error: "TypeError: unsupported operand"], [message: "Retrying with string conversion"]
            Output: "Unable to complete calculation"
            Response: {{"reasoning_score": 3, "explanation": "Failed to handle type conversion, no proper error recovery"}}
            
            Task: "Find weather for Paris"
            Events: [tool_call: "weather_api"], [message: "Parsed location"], [tool_output: "22°C, sunny"]
            Output: "The weather in Paris is 22°C and sunny"
            Response: {{"reasoning_score": 9, "explanation": "Clear logical flow, correct tool usage, complete output"}}

            Now analyze:
            TASK: {task_prompt[:200]}
            EXECUTION: {json.dumps(summarized_events)}
            OUTPUT: {final_output[:200]}
            STATUS: {trajectory.get("status", "unknown")}

            Respond with only a JSON object with reasoning_score (1-10) and explanation fields.
            """
            
            # Make real API call to OpenRouter
            api_key = os.environ.get("OPENROUTER_API_KEY")
            if not api_key:
                logger.warning("OPENROUTER_API_KEY not set, using heuristic analysis")
                return "error" not in str(events).lower() and "failed" not in str(final_output).lower()
            
            async with aiohttp.ClientSession() as session:
                headers = {
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                    "HTTP-Referer": "https://github.com/Arc-Computer/arc-cli",
                    "X-Title": "Arc CLI Analysis"
                }
                
                payload = {
                    "model": self.ai_model,
                    "messages": [
                        {"role": "system", "content": "You are an AI quality assurance expert analyzing agent execution quality."},
                        {"role": "user", "content": analysis_prompt}
                    ],
                    "temperature": 0.3,
                    "max_tokens": 200
                }
                
                async with session.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers=headers,
                    json=payload
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        ai_response = result["choices"][0]["message"]["content"]
                        
                        # Parse JSON response
                        try:
                            analysis = json.loads(ai_response)
                            score = analysis.get("reasoning_score", 5)
                            logger.info(f"AI reasoning analysis: score={score}, explanation={analysis.get('explanation', 'N/A')}")
                            result = score >= 7  # Consider good if score is 7 or above
                            # Cache the result
                            self._ai_cache[cache_key] = result
                            return result
                        except json.JSONDecodeError:
                            logger.warning(f"Failed to parse AI response as JSON: {ai_response}")
                            return False
                    else:
                        logger.error(f"OpenRouter API error: {response.status}")
                        # Fallback to heuristic
                        return "error" not in str(events).lower() and "failed" not in str(final_output).lower()
                
        except Exception as e:
            logger.error(f"AI reasoning analysis error: {e}")
            return False
    
    async def _ai_analyze_output_quality(self, trajectory: Dict[str, Any]) -> bool:
        """Use GPT-4.1-mini to analyze output quality and completeness."""
        # Check cache first
        cache_key = self._get_trajectory_hash(trajectory, "output_quality")
        if cache_key in self._ai_cache:
            logger.info(f"Using cached output quality analysis for {cache_key}")
            return self._ai_cache[cache_key]
        
        try:
            final_output = trajectory.get("final_output", "")
            task_prompt = trajectory.get("scenario", {}).get("task_prompt", "")
            
            # Create quality analysis prompt with few-shot examples
            quality_prompt = f"""
            Evaluate AI agent output quality.
            
            Example 1:
            Task: "Convert 100 USD to EUR"
            Output: "100 USD equals approximately 92 EUR at current exchange rates"
            Response: {{"quality_score": 9, "explanation": "Complete, accurate, and useful conversion"}}
            
            Example 2:
            Task: "Book a flight from NYC to LAX"
            Output: "I cannot book flights"
            Response: {{"quality_score": 2, "explanation": "Incomplete response, no alternative suggestions"}}

            Now evaluate:
            TASK: {task_prompt[:200]}
            OUTPUT: {final_output[:300]}

            Respond with only a JSON object with quality_score (1-10) and explanation fields.
            """
            
            # Make real API call to OpenRouter
            api_key = os.environ.get("OPENROUTER_API_KEY")
            if not api_key:
                logger.warning("OPENROUTER_API_KEY not set, using heuristic analysis")
                return len(str(final_output).strip()) > 50 and not any(
                    word in str(final_output).lower() for word in ["error", "failed", "cannot", "unable"]
                )
            
            async with aiohttp.ClientSession() as session:
                headers = {
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                    "HTTP-Referer": "https://github.com/Arc-Computer/arc-cli",
                    "X-Title": "Arc CLI Analysis"
                }
                
                payload = {
                    "model": self.ai_model,
                    "messages": [
                        {"role": "system", "content": "You are an AI quality assurance expert evaluating agent output quality."},
                        {"role": "user", "content": quality_prompt}
                    ],
                    "temperature": 0.3,
                    "max_tokens": 200
                }
                
                async with session.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers=headers,
                    json=payload
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        ai_response = result["choices"][0]["message"]["content"]
                        
                        # Parse JSON response
                        try:
                            analysis = json.loads(ai_response)
                            score = analysis.get("quality_score", 5)
                            logger.info(f"AI output quality analysis: score={score}, explanation={analysis.get('explanation', 'N/A')}")
                            result = score >= 7  # Consider good if score is 7 or above
                            # Cache the result
                            self._ai_cache[cache_key] = result
                            return result
                        except json.JSONDecodeError:
                            logger.warning(f"Failed to parse AI response as JSON: {ai_response}")
                            return True  # Default to success if parse fails
                    else:
                        logger.error(f"OpenRouter API error: {response.status}")
                        # Fallback to heuristic
                        return len(str(final_output).strip()) > 50 and not any(
                            word in str(final_output).lower() for word in ["error", "failed", "cannot", "unable"]
                        )
                
        except Exception as e:
            logger.error(f"AI output quality analysis error: {e}")
            return True  # Default to success if analysis fails