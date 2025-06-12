"""Root cause attribution system for mapping failures to capability assumptions."""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum


class FailureType(Enum):
    """Types of capability failures."""
    ASSUMPTION_VIOLATION = "assumption_violation"
    TOOL_MISCONFIGURATION = "tool_misconfiguration"
    LOGIC_ERROR = "logic_error"
    PERFORMANCE_DEGRADATION = "performance_degradation"
    EXTERNAL_DEPENDENCY = "external_dependency"


@dataclass
class FailureAttribution:
    """Attribution of a failure to specific capability breakdowns."""
    capability: str
    assumption: str
    failure_type: FailureType
    minimal_repro: str
    fix_suggestion: str
    confidence: float
    business_impact: str
    affected_scenarios: List[str]


class FailureAttributor:
    """Traces failures back to capability breakdowns and assumption violations."""
    
    def __init__(self):
        # Known capability-assumption mappings
        self.capability_assumptions = {
            "currency_processing": {
                "default_currency": "All monetary values are in USD unless specified",
                "conversion_rates": "Real-time conversion rates are always available",
                "currency_formats": "Standard currency symbols are universally recognized"
            },
            "data_validation": {
                "input_format": "Input data follows expected schema",
                "required_fields": "All required fields are always present",
                "data_types": "Data types match expected formats"
            },
            "external_apis": {
                "availability": "External APIs are always available",
                "rate_limits": "API rate limits are never exceeded", 
                "authentication": "API credentials are always valid"
            },
            "error_handling": {
                "graceful_degradation": "System can handle partial failures",
                "recovery_mechanisms": "Automatic recovery is possible for all errors",
                "user_communication": "Error messages are always user-friendly"
            },
            "performance": {
                "response_time": "Operations complete within expected timeframes",
                "resource_usage": "Memory and CPU usage stays within limits",
                "scalability": "System scales linearly with load"
            }
        }
        
        # Failure pattern recognition
        self.failure_patterns = {
            "currency": {
                "keywords": ["currency", "€", "$", "£", "¥", "exchange", "conversion"],
                "capability": "currency_processing",
                "assumption": "default_currency"
            },
            "timeout": {
                "keywords": ["timeout", "slow", "performance", "exceeded", "time limit"],
                "capability": "performance", 
                "assumption": "response_time"
            },
            "api_error": {
                "keywords": ["api", "connection", "network", "service unavailable", "http error"],
                "capability": "external_apis",
                "assumption": "availability"
            },
            "validation": {
                "keywords": ["invalid", "format", "schema", "missing field", "type error"],
                "capability": "data_validation",
                "assumption": "input_format"
            },
            "tool_error": {
                "keywords": ["tool", "function", "parameter", "execution error"],
                "capability": "tool_execution",
                "assumption": "configuration"
            }
        }
    
    def attribute_failure(self, execution_trace: Dict[str, Any]) -> FailureAttribution:
        """Trace failure back to capability breakdown and generate attribution."""
        
        # Step 1: Identify divergence point
        divergence_point = self._find_divergence(execution_trace)
        
        # Step 2: Map to capability breakdown
        failed_capability, violated_assumption = self._map_to_capability(divergence_point)
        
        # Step 3: Classify failure type
        failure_type = self._classify_failure_type(execution_trace, failed_capability)
        
        # Step 4: Generate minimal reproduction
        minimal_repro = self._extract_minimal_repro(execution_trace, divergence_point)
        
        # Step 5: Suggest capability fix
        fix_suggestion = self._suggest_capability_fix(failed_capability, violated_assumption)
        
        # Step 6: Assess business impact
        business_impact = self._assess_business_impact(execution_trace, failed_capability)
        
        # Step 7: Calculate confidence
        confidence = self._calculate_attribution_confidence(execution_trace, divergence_point)
        
        return FailureAttribution(
            capability=failed_capability,
            assumption=violated_assumption,
            failure_type=failure_type,
            minimal_repro=minimal_repro,
            fix_suggestion=fix_suggestion,
            confidence=confidence,
            business_impact=business_impact,
            affected_scenarios=[execution_trace.get("scenario_id", "unknown")]
        )
    
    def _find_divergence(self, execution_trace: Dict[str, Any]) -> Dict[str, Any]:
        """Identify where expected behavior diverged from actual behavior."""
        
        divergence_info = {
            "step": None,
            "expected": None,
            "actual": None,
            "context": {}
        }
        
        # Check trajectory events for errors
        trajectory = execution_trace.get("full_trajectory", [])
        
        for i, event in enumerate(trajectory):
            if event.get("type") == "tool_call":
                # Check for tool execution errors
                tool_output = event.get("tool_output", "")
                if "error" in str(tool_output).lower():
                    divergence_info.update({
                        "step": i,
                        "expected": "Successful tool execution",
                        "actual": f"Tool error: {tool_output}",
                        "context": {
                            "tool": event.get("tool"),
                            "input": event.get("tool_input"),
                            "error_output": tool_output
                        }
                    })
                    break
            
            elif event.get("type") == "error":
                # Direct error event
                divergence_info.update({
                    "step": i,
                    "expected": "Successful execution",
                    "actual": f"Error: {event.get('error', 'Unknown error')}",
                    "context": event
                })
                break
        
        # If no specific divergence found, check final status
        if divergence_info["step"] is None:
            status = execution_trace.get("status")
            if status == "error":
                divergence_info.update({
                    "step": len(trajectory),
                    "expected": "Successful completion",
                    "actual": f"Failed with error: {execution_trace.get('error', 'Unknown')}",
                    "context": {
                        "final_response": execution_trace.get("final_response"),
                        "error": execution_trace.get("error")
                    }
                })
        
        return divergence_info
    
    def _map_to_capability(self, divergence_point: Dict[str, Any]) -> tuple[str, str]:
        """Map divergence point to specific capability and assumption."""
        
        context_text = str(divergence_point.get("context", {})).lower()
        actual_behavior = str(divergence_point.get("actual", "")).lower()
        
        # Search for pattern matches
        for pattern_name, pattern_info in self.failure_patterns.items():
            keywords = pattern_info["keywords"]
            
            # Check if any keywords match the failure context
            if any(keyword in context_text or keyword in actual_behavior for keyword in keywords):
                capability = pattern_info["capability"]
                assumption_key = pattern_info["assumption"]
                
                # Get the specific assumption text
                assumption = self.capability_assumptions.get(capability, {}).get(
                    assumption_key, 
                    f"Unknown assumption in {capability}"
                )
                
                return capability, assumption
        
        # Default fallback
        return "general_execution", "System operates as expected under all conditions"
    
    def _classify_failure_type(self, execution_trace: Dict[str, Any], capability: str) -> FailureType:
        """Classify the type of failure based on trace and capability."""
        
        error_msg = str(execution_trace.get("error", "")).lower()
        final_response = str(execution_trace.get("final_response", "")).lower()
        
        if capability == "currency_processing":
            return FailureType.ASSUMPTION_VIOLATION
        elif capability == "external_apis":
            return FailureType.EXTERNAL_DEPENDENCY
        elif capability == "performance":
            return FailureType.PERFORMANCE_DEGRADATION
        elif capability == "tool_execution":
            return FailureType.TOOL_MISCONFIGURATION
        elif "logic" in error_msg or "reasoning" in error_msg:
            return FailureType.LOGIC_ERROR
        else:
            return FailureType.ASSUMPTION_VIOLATION
    
    def _extract_minimal_repro(self, execution_trace: Dict[str, Any], divergence_point: Dict[str, Any]) -> str:
        """Generate minimal reproduction case from complex failure."""
        
        task_prompt = execution_trace.get("task_prompt", "Unknown task")
        expected = divergence_point.get("expected", "Unknown expected behavior")
        actual = divergence_point.get("actual", "Unknown actual behavior")
        
        # Format as a clear repro case
        repro_lines = []
        repro_lines.append(f"Prompt: \"{task_prompt}\"")
        repro_lines.append(f"Expected: {expected}")
        repro_lines.append(f"Actual: {actual}")
        
        # Add context if available
        context = divergence_point.get("context", {})
        if isinstance(context, dict) and context.get("tool"):
            repro_lines.append(f"Tool Used: {context['tool']}")
            if context.get("tool_input"):
                repro_lines.append(f"Tool Input: {context['tool_input']}")
        
        return "\n".join(repro_lines)
    
    def _suggest_capability_fix(self, capability: str, assumption: str) -> str:
        """Suggest specific fix for the capability breakdown."""
        
        capability_fixes = {
            "currency_processing": "Add currency detection and validation. Implement multi-currency support with explicit conversion prompts.",
            "external_apis": "Add retry logic with exponential backoff. Implement circuit breaker pattern for API failures.",
            "performance": "Optimize execution path. Add timeout handling and async processing where possible.",
            "data_validation": "Add comprehensive input validation. Implement schema checking and error handling for malformed data.",
            "tool_execution": "Review tool configurations. Add parameter validation and error recovery mechanisms.",
            "error_handling": "Implement graceful degradation. Add user-friendly error messages and recovery suggestions."
        }
        
        return capability_fixes.get(
            capability, 
            f"Review and strengthen the {capability} capability. Address assumption: '{assumption}'"
        )
    
    def _assess_business_impact(self, execution_trace: Dict[str, Any], capability: str) -> str:
        """Assess the business impact of the capability failure."""
        
        capability_impacts = {
            "currency_processing": "Financial calculation errors affecting customer billing and compliance",
            "external_apis": "Service degradation impacting user experience and system reliability", 
            "performance": "Poor user experience leading to potential customer churn",
            "data_validation": "Data integrity issues potentially causing downstream errors",
            "tool_execution": "Core functionality failures reducing system effectiveness",
            "error_handling": "Poor user experience during error conditions"
        }
        
        base_impact = capability_impacts.get(capability, "System reliability and user experience degradation")
        
        # Enhance with execution specifics
        execution_time = execution_trace.get("execution_time_seconds", 0)
        if execution_time > 10:
            base_impact += ". Extended processing time increases user frustration."
        
        return base_impact
    
    def _calculate_attribution_confidence(self, execution_trace: Dict[str, Any], divergence_point: Dict[str, Any]) -> float:
        """Calculate confidence in the failure attribution."""
        
        confidence = 0.5  # Base confidence
        
        # Increase confidence if we have clear error information
        if divergence_point.get("step") is not None:
            confidence += 0.2
        
        # Increase confidence if we have specific tool/context information
        if divergence_point.get("context"):
            confidence += 0.15
        
        # Increase confidence if pattern matching was successful
        context_text = str(divergence_point.get("context", {})).lower()
        if any(pattern in context_text for pattern_keywords in self.failure_patterns.values() 
               for pattern in pattern_keywords["keywords"]):
            confidence += 0.15
        
        # Cap at 1.0
        return min(confidence, 1.0)


def analyze_failure_patterns(execution_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze multiple execution results to identify common failure patterns."""
    
    attributor = FailureAttributor()
    attributions = []
    
    # Process each failed execution
    for result in execution_results:
        if not result.get("success", True):  # Only process failures
            attribution = attributor.attribute_failure(result.get("trajectory", {}))
            attributions.append(attribution)
    
    # Aggregate patterns
    capability_counts = {}
    assumption_counts = {}
    failure_type_counts = {}
    
    for attr in attributions:
        capability_counts[attr.capability] = capability_counts.get(attr.capability, 0) + 1
        assumption_counts[attr.assumption] = assumption_counts.get(attr.assumption, 0) + 1
        failure_type_counts[attr.failure_type.value] = failure_type_counts.get(attr.failure_type.value, 0) + 1
    
    # Find most common issues
    most_common_capability = max(capability_counts.items(), key=lambda x: x[1]) if capability_counts else None
    most_common_assumption = max(assumption_counts.items(), key=lambda x: x[1]) if assumption_counts else None
    
    return {
        "total_failures": len(attributions),
        "capability_breakdown": capability_counts,
        "assumption_violations": assumption_counts,
        "failure_types": failure_type_counts,
        "primary_issue": {
            "capability": most_common_capability[0] if most_common_capability else None,
            "assumption": most_common_assumption[0] if most_common_assumption else None,
            "impact": f"{most_common_capability[1]} out of {len(attributions)} failures" if most_common_capability else None
        },
        "attributions": attributions
    }