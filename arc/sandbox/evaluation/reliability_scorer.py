"""
Reliability Scoring Module for Arc-Eval Production
Production version adapted from experiments/src/evaluation/reliability_scorer.py
"""

from typing import Dict, List, Any, Optional
import json
from dataclasses import dataclass, field
from enum import Enum

class ReliabilityDimension(Enum):
    """Dimensions of reliability we measure"""
    TOOL_EXECUTION = "tool_execution"
    RESPONSE_QUALITY = "response_quality"
    ERROR_HANDLING = "error_handling"
    PERFORMANCE = "performance"
    COMPLETENESS = "completeness"

@dataclass
class ReliabilityScore:
    """Composite reliability score with breakdown by dimension"""
    overall_score: float
    dimension_scores: Dict[str, float]
    issues_found: List[Dict[str, Any]] = field(default_factory=list)
    recommendations: List[Dict[str, Any]] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "overall_score": self.overall_score,
            "dimension_scores": self.dimension_scores,
            "issues_found": self.issues_found,
            "recommendations": self.recommendations,
            "grade": self._get_grade()
        }
    
    def _get_grade(self) -> str:
        """Convert score to letter grade"""
        if self.overall_score >= 90:
            return "A"
        elif self.overall_score >= 80:
            return "B"
        elif self.overall_score >= 70:
            return "C"
        elif self.overall_score >= 60:
            return "D"
        else:
            return "F"

class ReliabilityScorer:
    """Calculates reliability scores from agent execution trajectories"""
    
    def __init__(self):
        self.weights = {
            ReliabilityDimension.TOOL_EXECUTION: 0.3,
            ReliabilityDimension.RESPONSE_QUALITY: 0.25,
            ReliabilityDimension.ERROR_HANDLING: 0.2,
            ReliabilityDimension.PERFORMANCE: 0.15,
            ReliabilityDimension.COMPLETENESS: 0.1
        }
    
    def score_trajectory(
        self, 
        trajectory: Dict[str, Any],
        scenario: Optional[Dict[str, Any]] = None
    ) -> ReliabilityScore:
        """Calculate reliability score from execution trajectory"""
        
        dimension_scores = {}
        issues = []
        recommendations = []
        
        # Score each dimension
        tool_score, tool_issues = self._score_tool_execution(trajectory, scenario)
        dimension_scores[ReliabilityDimension.TOOL_EXECUTION.value] = tool_score
        issues.extend(tool_issues)
        
        response_score, response_issues = self._score_response_quality(trajectory, scenario)
        dimension_scores[ReliabilityDimension.RESPONSE_QUALITY.value] = response_score
        issues.extend(response_issues)
        
        error_score, error_issues = self._score_error_handling(trajectory)
        dimension_scores[ReliabilityDimension.ERROR_HANDLING.value] = error_score
        issues.extend(error_issues)
        
        perf_score, perf_issues = self._score_performance(trajectory)
        dimension_scores[ReliabilityDimension.PERFORMANCE.value] = perf_score
        issues.extend(perf_issues)
        
        complete_score, complete_issues = self._score_completeness(trajectory, scenario)
        dimension_scores[ReliabilityDimension.COMPLETENESS.value] = complete_score
        issues.extend(complete_issues)
        
        # Calculate weighted overall score
        overall_score = sum(
            dimension_scores[dim.value] * weight 
            for dim, weight in self.weights.items()
        )
        
        # Generate recommendations based on issues
        recommendations = self._generate_recommendations(issues, dimension_scores)
        
        return ReliabilityScore(
            overall_score=overall_score,
            dimension_scores=dimension_scores,
            issues_found=issues,
            recommendations=recommendations
        )
    
    def _score_tool_execution(
        self, 
        trajectory: Dict[str, Any],
        scenario: Optional[Dict[str, Any]] = None
    ) -> tuple[float, List[Dict[str, Any]]]:
        """Score tool execution reliability"""
        issues = []
        score = 100.0
        
        # Check if tools were called when expected
        expected_tools = scenario.get("expected_tools", []) if scenario else []
        trajectory_events = trajectory.get("full_trajectory", [])
        tool_calls = [e for e in trajectory_events if e.get("type") == "tool_call"]
        
        if expected_tools and not tool_calls:
            issues.append({
                "severity": "high",
                "dimension": "tool_execution",
                "issue": "No tools were called when tools were expected",
                "impact": -30
            })
            score -= 30
        
        # Check for tool execution errors
        for event in tool_calls:
            if "error" in str(event.get("tool_output", "")).lower():
                issues.append({
                    "severity": "medium",
                    "dimension": "tool_execution",
                    "issue": f"Tool '{event.get('tool')}' returned an error",
                    "impact": -20
                })
                score -= 20
        
        # Check for proper tool usage (right tool for the job)
        if scenario and scenario.get("success_criteria"):
            criteria = scenario["success_criteria"]
            if "tool_call_count" in criteria:
                expected_count = criteria["tool_call_count"]
                actual_count = len(tool_calls)
                if actual_count != expected_count:
                    issues.append({
                        "severity": "low",
                        "dimension": "tool_execution",
                        "issue": f"Expected {expected_count} tool calls but got {actual_count}",
                        "impact": -10
                    })
                    score -= 10
        
        return max(0, score), issues
    
    def _score_response_quality(
        self,
        trajectory: Dict[str, Any],
        scenario: Optional[Dict[str, Any]] = None
    ) -> tuple[float, List[Dict[str, Any]]]:
        """Score response quality"""
        issues = []
        score = 100.0
        
        final_response = trajectory.get("final_response", "")
        
        # Check for empty response
        if not final_response or final_response.strip() == "":
            issues.append({
                "severity": "high",
                "dimension": "response_quality",
                "issue": "Empty or missing final response",
                "impact": -40
            })
            score -= 40
        
        # Check for error indicators in response
        error_indicators = ["error:", "exception:", "failed:", "unable to"]
        for indicator in error_indicators:
            if indicator in final_response.lower():
                issues.append({
                    "severity": "medium",
                    "dimension": "response_quality",
                    "issue": f"Response contains error indicator: '{indicator}'",
                    "impact": -15
                })
                score -= 15
                break
        
        # Check success criteria if provided
        if scenario and scenario.get("success_criteria"):
            criteria = scenario["success_criteria"]
            if "contains_key_info" in criteria:
                key_info = criteria["contains_key_info"]
                for info in key_info:
                    if info.lower() not in final_response.lower():
                        issues.append({
                            "severity": "medium",
                            "dimension": "response_quality",
                            "issue": f"Response missing required information: '{info}'",
                            "impact": -20
                        })
                        score -= 20
        
        return max(0, score), issues
    
    def _score_error_handling(
        self,
        trajectory: Dict[str, Any]
    ) -> tuple[float, List[Dict[str, Any]]]:
        """Score error handling capability"""
        issues = []
        score = 100.0
        
        # Check if there were any errors
        if trajectory.get("status") == "error":
            error_message = trajectory.get("error", "Unknown error")
            issues.append({
                "severity": "high",
                "dimension": "error_handling",
                "issue": f"Execution failed with error: {error_message}",
                "impact": -50
            })
            score -= 50
        
        # Check for unhandled exceptions
        trajectory_events = trajectory.get("full_trajectory", [])
        for event in trajectory_events:
            if event.get("type") == "error" and not event.get("handled", False):
                issues.append({
                    "severity": "medium",
                    "dimension": "error_handling",
                    "issue": "Unhandled error during execution",
                    "impact": -25
                })
                score -= 25
        
        return max(0, score), issues
    
    def _score_performance(
        self,
        trajectory: Dict[str, Any]
    ) -> tuple[float, List[Dict[str, Any]]]:
        """Score performance metrics"""
        issues = []
        score = 100.0
        
        execution_time = trajectory.get("execution_time_seconds", 0)
        
        # Check execution time thresholds
        if execution_time > 30:
            issues.append({
                "severity": "high",
                "dimension": "performance",
                "issue": f"Execution took {execution_time:.1f}s (>30s threshold)",
                "impact": -30
            })
            score -= 30
        elif execution_time > 10:
            issues.append({
                "severity": "low",
                "dimension": "performance",
                "issue": f"Execution took {execution_time:.1f}s (>10s threshold)",
                "impact": -10
            })
            score -= 10
        
        # Check token usage
        token_usage = trajectory.get("token_usage", {})
        total_tokens = token_usage.get("total_tokens", 0)
        
        if total_tokens > 4000:
            issues.append({
                "severity": "medium",
                "dimension": "performance",
                "issue": f"High token usage: {total_tokens} tokens",
                "impact": -20
            })
            score -= 20
        
        return max(0, score), issues
    
    def _score_completeness(
        self,
        trajectory: Dict[str, Any],
        scenario: Optional[Dict[str, Any]] = None
    ) -> tuple[float, List[Dict[str, Any]]]:
        """Score task completeness"""
        issues = []
        score = 100.0
        
        # Check if the task was completed
        if trajectory.get("status") != "success":
            issues.append({
                "severity": "high",
                "dimension": "completeness",
                "issue": "Task was not completed successfully",
                "impact": -50
            })
            score -= 50
        
        # Check if all expected steps were completed
        if scenario and scenario.get("expected_steps"):
            expected_steps = scenario["expected_steps"]
            trajectory_events = trajectory.get("full_trajectory", [])
            
            for step in expected_steps:
                step_found = any(
                    step.lower() in str(event).lower() 
                    for event in trajectory_events
                )
                if not step_found:
                    issues.append({
                        "severity": "medium",
                        "dimension": "completeness",
                        "issue": f"Missing expected step: '{step}'",
                        "impact": -15
                    })
                    score -= 15
        
        return max(0, score), issues
    
    def _generate_recommendations(
        self,
        issues: List[Dict[str, Any]],
        dimension_scores: Dict[str, float]
    ) -> List[Dict[str, Any]]:
        """Generate recommendations based on issues found"""
        recommendations = []
        
        # Group issues by dimension
        issues_by_dimension = {}
        for issue in issues:
            dim = issue["dimension"]
            if dim not in issues_by_dimension:
                issues_by_dimension[dim] = []
            issues_by_dimension[dim].append(issue)
        
        # Generate recommendations for each problematic dimension
        for dim, dim_issues in issues_by_dimension.items():
            if dimension_scores.get(dim, 100) < 70:  # Focus on low-scoring dimensions
                if dim == "tool_execution":
                    recommendations.append({
                        "dimension": dim,
                        "priority": "high",
                        "recommendation": "Improve tool selection and error handling logic",
                        "specific_actions": [
                            "Add retry logic for failed tool calls",
                            "Implement tool validation before execution",
                            "Add fallback strategies when tools fail"
                        ]
                    })
                elif dim == "response_quality":
                    recommendations.append({
                        "dimension": dim,
                        "priority": "high",
                        "recommendation": "Enhance response generation and validation",
                        "specific_actions": [
                            "Add response quality checks before returning",
                            "Implement content validation against requirements",
                            "Add structured output formatting"
                        ]
                    })
                elif dim == "error_handling":
                    recommendations.append({
                        "dimension": dim,
                        "priority": "critical",
                        "recommendation": "Implement comprehensive error handling",
                        "specific_actions": [
                            "Add try-catch blocks around critical operations",
                            "Implement graceful degradation strategies",
                            "Add error recovery mechanisms"
                        ]
                    })
                elif dim == "performance":
                    recommendations.append({
                        "dimension": dim,
                        "priority": "medium",
                        "recommendation": "Optimize performance and resource usage",
                        "specific_actions": [
                            "Reduce token usage through prompt optimization",
                            "Implement caching for repeated operations",
                            "Add timeouts for long-running operations"
                        ]
                    })
                elif dim == "completeness":
                    recommendations.append({
                        "dimension": dim,
                        "priority": "high",
                        "recommendation": "Ensure all required steps are completed",
                        "specific_actions": [
                            "Add task completion validation",
                            "Implement step tracking and verification",
                            "Add progress monitoring"
                        ]
                    })
        
        return recommendations


def calculate_reliability_score(
    trajectory: Dict[str, Any],
    scenario: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Convenience function to calculate reliability score"""
    scorer = ReliabilityScorer()
    score = scorer.score_trajectory(trajectory, scenario)
    return score.to_dict()