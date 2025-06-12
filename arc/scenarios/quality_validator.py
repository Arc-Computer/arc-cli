"""
TRAIL-based quality validator for enhanced scenario validation.

Validates scenario quality against real-world failure patterns from TRAIL dataset,
ensuring generated scenarios match characteristics of actual agent failures.
"""

import re
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

from ..core.models.scenario import Scenario
from .failure_patterns.library import FailurePattern


class ValidationCriteria(Enum):
    """Validation criteria based on TRAIL analysis."""
    REALISTIC_ERROR_PATTERN = "realistic_error_pattern"
    APPROPRIATE_COMPLEXITY = "appropriate_complexity"
    RECOVERABLE_FAILURE = "recoverable_failure"
    DOMAIN_RELEVANCE = "domain_relevance"
    ASSUMPTION_SPECIFICITY = "assumption_specificity"


@dataclass
class ValidationResult:
    """Result of scenario quality validation."""
    passed: bool
    overall_score: float
    criteria_scores: Dict[ValidationCriteria, float] = field(default_factory=dict)
    feedback: List[str] = field(default_factory=list)
    improvement_suggestions: List[str] = field(default_factory=list)
    trail_alignment_score: float = 0.0


@dataclass
class TrailQualityMetrics:
    """Quality metrics based on TRAIL dataset characteristics."""
    error_pattern_realism: float = 0.0
    failure_mode_accuracy: float = 0.0
    domain_specificity: float = 0.0
    recovery_feasibility: float = 0.0
    assumption_violation_clarity: float = 0.0
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return {
            "error_pattern_realism": self.error_pattern_realism,
            "failure_mode_accuracy": self.failure_mode_accuracy,
            "domain_specificity": self.domain_specificity,
            "recovery_feasibility": self.recovery_feasibility,
            "assumption_violation_clarity": self.assumption_violation_clarity
        }


class TrailQualityValidator:
    """Enhanced quality validator using TRAIL dataset insights."""
    
    def __init__(self, quality_threshold: float = 3.0):
        """Initialize the TRAIL quality validator.
        
        Args:
            quality_threshold: Minimum quality score (1-5 scale)
        """
        self.quality_threshold = quality_threshold
        self.trail_patterns = self._load_trail_quality_patterns()
        self.domain_vocabularies = self._load_domain_vocabularies()
        self.error_pattern_signatures = self._load_error_signatures()
    
    def validate_scenario(self, scenario: Scenario) -> ValidationResult:
        """Validate scenario quality against TRAIL-based criteria.
        
        Args:
            scenario: Scenario to validate
            
        Returns:
            Validation result with detailed feedback
        """
        criteria_scores = {}
        feedback = []
        improvement_suggestions = []
        
        # Validate each criterion
        criteria_scores[ValidationCriteria.REALISTIC_ERROR_PATTERN] = self._validate_error_pattern(scenario, feedback)
        criteria_scores[ValidationCriteria.APPROPRIATE_COMPLEXITY] = self._validate_complexity(scenario, feedback)
        criteria_scores[ValidationCriteria.RECOVERABLE_FAILURE] = self._validate_recoverability(scenario, feedback)
        criteria_scores[ValidationCriteria.DOMAIN_RELEVANCE] = self._validate_domain_relevance(scenario, feedback)
        criteria_scores[ValidationCriteria.ASSUMPTION_SPECIFICITY] = self._validate_assumption_specificity(scenario, feedback)
        
        # Calculate overall score (weighted average)
        weights = {
            ValidationCriteria.REALISTIC_ERROR_PATTERN: 0.3,
            ValidationCriteria.APPROPRIATE_COMPLEXITY: 0.2,
            ValidationCriteria.RECOVERABLE_FAILURE: 0.2,
            ValidationCriteria.DOMAIN_RELEVANCE: 0.15,
            ValidationCriteria.ASSUMPTION_SPECIFICITY: 0.15
        }
        
        overall_score = sum(
            criteria_scores[criterion] * weight
            for criterion, weight in weights.items()
        )
        
        # Calculate TRAIL alignment score
        trail_alignment = self._calculate_trail_alignment(scenario)
        
        # Generate improvement suggestions
        improvement_suggestions = self._generate_improvement_suggestions(scenario, criteria_scores)
        
        # Determine if passed
        passed = overall_score >= self.quality_threshold and trail_alignment >= 0.6
        
        return ValidationResult(
            passed=passed,
            overall_score=overall_score,
            criteria_scores=criteria_scores,
            feedback=feedback,
            improvement_suggestions=improvement_suggestions,
            trail_alignment_score=trail_alignment
        )
    
    def calculate_trail_metrics(self, scenario: Scenario) -> TrailQualityMetrics:
        """Calculate detailed TRAIL-based quality metrics.
        
        Args:
            scenario: Scenario to analyze
            
        Returns:
            Detailed quality metrics
        """
        metrics = TrailQualityMetrics()
        
        # Error pattern realism (based on TRAIL error patterns)
        metrics.error_pattern_realism = self._assess_error_realism(scenario)
        
        # Failure mode accuracy (does failure mode match error type?)
        metrics.failure_mode_accuracy = self._assess_failure_mode_accuracy(scenario)
        
        # Domain specificity (how well does it fit target domain?)
        metrics.domain_specificity = self._assess_domain_specificity(scenario)
        
        # Recovery feasibility (can agent potentially recover?)
        metrics.recovery_feasibility = self._assess_recovery_feasibility(scenario)
        
        # Assumption violation clarity (is assumption violation clear?)
        metrics.assumption_violation_clarity = self._assess_assumption_clarity(scenario)
        
        return metrics
    
    def _validate_error_pattern(self, scenario: Scenario, feedback: List[str]) -> float:
        """Validate if scenario follows realistic error patterns from TRAIL."""
        score = 5.0
        description = scenario.description.lower()
        
        # Check for known error pattern signatures
        realistic_patterns = 0
        pattern_keywords = {
            "api_failure": ["timeout", "connection", "service unavailable", "rate limit"],
            "data_error": ["parse", "format", "encoding", "schema", "malformed"],
            "logic_error": ["assumption", "calculation", "logic", "condition"],
            "auth_error": ["permission", "authentication", "unauthorized", "token"],
            "coordination_error": ["deadlock", "race", "coordination", "conflict"]
        }
        
        for pattern_type, keywords in pattern_keywords.items():
            if any(keyword in description for keyword in keywords):
                realistic_patterns += 1
        
        if realistic_patterns == 0:
            score -= 2.0
            feedback.append("Scenario lacks recognizable error patterns from TRAIL dataset")
        elif realistic_patterns > 2:
            score -= 0.5
            feedback.append("Scenario combines too many error patterns")
        
        # Check for unrealistic patterns
        unrealistic_patterns = ["magic", "impossible", "teleport", "mind reading"]
        if any(pattern in description for pattern in unrealistic_patterns):
            score -= 3.0
            feedback.append("Scenario contains unrealistic failure patterns")
        
        # Boost for TRAIL-specific characteristics
        trail_characteristics = ["step", "trace", "execution", "agent workflow"]
        if any(char in description for char in trail_characteristics):
            score += 0.5
        
        return max(1.0, min(5.0, score))
    
    def _validate_complexity(self, scenario: Scenario, feedback: List[str]) -> float:
        """Validate if scenario has appropriate complexity for testing."""
        score = 5.0
        description = scenario.description
        
        # Check description length (should be detailed enough)
        if len(description) < 50:
            score -= 2.0
            feedback.append("Scenario description too brief for meaningful testing")
        elif len(description) > 500:
            score -= 1.0
            feedback.append("Scenario description overly complex")
        
        # Check for multiple steps/components
        step_indicators = ["first", "then", "when", "during", "after", "while"]
        step_count = sum(1 for indicator in step_indicators if indicator in description.lower())
        
        if step_count == 0:
            score -= 1.5
            feedback.append("Scenario lacks multi-step complexity")
        elif step_count > 5:
            score -= 1.0
            feedback.append("Scenario too complex with too many steps")
        
        # Check for specific failure conditions
        if not scenario.expected_failure_mode or len(scenario.expected_failure_mode) < 10:
            score -= 1.0
            feedback.append("Scenario lacks specific expected failure mode")
        
        return max(1.0, min(5.0, score))
    
    def _validate_recoverability(self, scenario: Scenario, feedback: List[str]) -> float:
        """Validate if failure is potentially recoverable (realistic for testing)."""
        score = 5.0
        
        # Check for irrecoverable failures
        irrecoverable_patterns = [
            "system destroyed", "data lost forever", "irreversible", 
            "catastrophic failure", "complete system failure"
        ]
        
        description_lower = scenario.description.lower()
        if any(pattern in description_lower for pattern in irrecoverable_patterns):
            score -= 3.0
            feedback.append("Scenario describes irrecoverable failure (unrealistic for testing)")
        
        # Check for recovery hints
        recovery_hints = [
            "retry", "fallback", "alternative", "backup", "recover", 
            "handle", "catch", "timeout", "reconnect"
        ]
        
        if any(hint in description_lower for hint in recovery_hints):
            score += 0.5
        else:
            score -= 0.5
            feedback.append("Scenario could benefit from recovery considerations")
        
        # Check metadata for recovery patterns
        metadata = scenario.metadata or {}
        source_pattern = metadata.get("source_pattern")
        if source_pattern and "recovery_patterns" in str(metadata):
            score += 0.5
        
        return max(1.0, min(5.0, score))
    
    def _validate_domain_relevance(self, scenario: Scenario, feedback: List[str]) -> float:
        """Validate domain specificity and relevance."""
        score = 5.0
        description = scenario.description.lower()
        
        # Check for domain-specific vocabulary
        domain_matches = 0
        for domain, vocabulary in self.domain_vocabularies.items():
            matches = sum(1 for term in vocabulary if term in description)
            if matches > 0:
                domain_matches += 1
        
        if domain_matches == 0:
            score -= 2.0
            feedback.append("Scenario lacks domain-specific terminology")
        elif domain_matches > 2:
            score -= 0.5
            feedback.append("Scenario mixes multiple domains (reduce focus)")
        
        # Check for generic placeholder text
        generic_terms = ["example", "sample", "placeholder", "todo", "fixme"]
        if any(term in description for term in generic_terms):
            score -= 2.0
            feedback.append("Scenario contains placeholder text")
        
        # Boost for specific tools/APIs
        if any(tag in scenario.tags for tag in ["api", "tool", "integration"]):
            score += 0.5
        
        return max(1.0, min(5.0, score))
    
    def _validate_assumption_specificity(self, scenario: Scenario, feedback: List[str]) -> float:
        """Validate how clearly the scenario violates specific assumptions."""
        score = 5.0
        
        # Check if scenario is tagged as assumption violation
        if "assumption_violation" not in scenario.tags:
            score -= 1.0
            feedback.append("Scenario not clearly marked as assumption violation")
        
        # Check metadata for assumption details
        metadata = scenario.metadata or {}
        assumption_type = metadata.get("assumption_type")
        assumption_value = metadata.get("assumption_value")
        
        if not assumption_type:
            score -= 1.5
            feedback.append("Scenario lacks specific assumption type identification")
        
        if not assumption_value:
            score -= 1.0
            feedback.append("Scenario lacks specific assumption value being tested")
        
        # Check description for assumption violation language
        violation_terms = [
            "assumes", "expects", "configured for", "default", "expected", 
            "supposed to", "designed for", "violates", "contradicts"
        ]
        
        description_lower = scenario.description.lower()
        if not any(term in description_lower for term in violation_terms):
            score -= 1.0
            feedback.append("Scenario doesn't clearly describe assumption violation")
        
        return max(1.0, min(5.0, score))
    
    def _calculate_trail_alignment(self, scenario: Scenario) -> float:
        """Calculate how well scenario aligns with TRAIL dataset characteristics."""
        alignment_score = 0.0
        
        # Check for TRAIL-specific tags
        trail_tags = ["trail_adapted", "trail_inspired"]
        if any(tag in scenario.tags for tag in trail_tags):
            alignment_score += 0.3
        
        # Check for error type classification
        error_types = ["reasoning", "execution", "planning"]
        scenario_tags = [tag.lower() for tag in scenario.tags]
        if any(error_type in scenario_tags for error_type in error_types):
            alignment_score += 0.2
        
        # Check for realistic failure progression
        description_lower = scenario.description.lower()
        progression_indicators = ["step", "during", "when", "then", "after"]
        progression_score = min(0.3, sum(0.06 for indicator in progression_indicators if indicator in description_lower))
        alignment_score += progression_score
        
        # Check for metadata completeness
        metadata = scenario.metadata or {}
        if metadata.get("source_pattern"):
            alignment_score += 0.1
        if metadata.get("trail_category"):
            alignment_score += 0.1
        
        return min(1.0, alignment_score)
    
    def _assess_error_realism(self, scenario: Scenario) -> float:
        """Assess how realistic the error pattern is based on TRAIL data."""
        # Implementation would analyze error patterns against TRAIL statistics
        description = scenario.description.lower()
        
        # Reasoning errors are most common in TRAIL (60%)
        reasoning_indicators = ["calculation", "logic", "assumption", "interpret"]
        if any(indicator in description for indicator in reasoning_indicators):
            return 0.9
        
        # Execution errors are moderately common (30%)
        execution_indicators = ["timeout", "connection", "service", "api"]
        if any(indicator in description for indicator in execution_indicators):
            return 0.8
        
        # Planning errors are least common but high impact (10%)
        planning_indicators = ["coordination", "workflow", "deadlock", "sequence"]
        if any(indicator in description for indicator in planning_indicators):
            return 0.7
        
        return 0.5  # Default for unclassified
    
    def _assess_failure_mode_accuracy(self, scenario: Scenario) -> float:
        """Assess if the failure mode matches the error type."""
        expected_failure = (scenario.expected_failure_mode or "").lower()
        description = scenario.description.lower()
        
        # Check consistency between description and expected failure
        if not expected_failure:
            return 0.3
        
        # Look for consistency indicators
        consistency_score = 0.5
        
        # If description mentions timeout, failure should mention timeout
        if "timeout" in description and "timeout" in expected_failure:
            consistency_score += 0.3
        
        # If description mentions data issues, failure should mention data
        if any(term in description for term in ["data", "parse", "format"]) and \
           any(term in expected_failure for term in ["data", "parse", "format"]):
            consistency_score += 0.3
        
        return min(1.0, consistency_score)
    
    def _assess_domain_specificity(self, scenario: Scenario) -> float:
        """Assess how well scenario fits target domain."""
        description = scenario.description.lower()
        
        # Check for domain-specific terms
        domain_score = 0.0
        
        # Finance domain (primary target)
        finance_terms = ["currency", "financial", "trading", "portfolio", "transaction"]
        if any(term in description for term in finance_terms):
            domain_score += 0.8
        
        # Technical terms (secondary)
        tech_terms = ["api", "service", "data", "system", "agent"]
        if any(term in description for term in tech_terms):
            domain_score += 0.3
        
        return min(1.0, domain_score)
    
    def _assess_recovery_feasibility(self, scenario: Scenario) -> float:
        """Assess if agent could potentially recover from this failure."""
        description = scenario.description.lower()
        
        # Positive indicators for recovery
        recovery_indicators = ["retry", "fallback", "alternative", "recover"]
        recovery_score = 0.3  # Base score
        
        if any(indicator in description for indicator in recovery_indicators):
            recovery_score += 0.4
        
        # Negative indicators (permanent failures)
        permanent_indicators = ["destroyed", "corrupted", "lost", "irreversible"]
        if any(indicator in description for indicator in permanent_indicators):
            recovery_score -= 0.5
        
        return max(0.0, min(1.0, recovery_score))
    
    def _assess_assumption_clarity(self, scenario: Scenario) -> float:
        """Assess how clearly the assumption violation is described."""
        description = scenario.description.lower()
        metadata = scenario.metadata or {}
        
        clarity_score = 0.0
        
        # Check for explicit assumption language
        assumption_terms = ["assumes", "expects", "configured", "default"]
        if any(term in description for term in assumption_terms):
            clarity_score += 0.4
        
        # Check for metadata about assumptions
        if metadata.get("assumption_type"):
            clarity_score += 0.3
        
        if metadata.get("assumption_value"):
            clarity_score += 0.3
        
        return min(1.0, clarity_score)
    
    def _generate_improvement_suggestions(
        self,
        scenario: Scenario,
        criteria_scores: Dict[ValidationCriteria, float]
    ) -> List[str]:
        """Generate specific improvement suggestions based on validation scores."""
        suggestions = []
        
        # Suggestions based on low scores
        if criteria_scores[ValidationCriteria.REALISTIC_ERROR_PATTERN] < 3.0:
            suggestions.append("Add more realistic error patterns based on common agent failures")
        
        if criteria_scores[ValidationCriteria.APPROPRIATE_COMPLEXITY] < 3.0:
            suggestions.append("Increase scenario complexity with multi-step failure progression")
        
        if criteria_scores[ValidationCriteria.RECOVERABLE_FAILURE] < 3.0:
            suggestions.append("Ensure failure mode allows for potential recovery mechanisms")
        
        if criteria_scores[ValidationCriteria.DOMAIN_RELEVANCE] < 3.0:
            suggestions.append("Add more domain-specific terminology and context")
        
        if criteria_scores[ValidationCriteria.ASSUMPTION_SPECIFICITY] < 3.0:
            suggestions.append("Clarify which specific assumption is being violated")
        
        return suggestions
    
    def _load_trail_quality_patterns(self) -> Dict[str, Any]:
        """Load quality patterns based on TRAIL analysis."""
        return {
            "common_error_patterns": [
                "API timeout during operation",
                "Data parsing failure with malformed input",
                "Authentication failure with expired token",
                "Rate limit exceeded during batch operation",
                "Circular dependency in task execution"
            ],
            "complexity_indicators": [
                "multi-step", "during", "when", "after", "sequence", "workflow"
            ],
            "recovery_patterns": [
                "retry", "fallback", "alternative", "backup", "handle"
            ]
        }
    
    def _load_domain_vocabularies(self) -> Dict[str, List[str]]:
        """Load domain-specific vocabularies for validation."""
        return {
            "finance": [
                "currency", "transaction", "portfolio", "trading", "investment",
                "market", "price", "exchange", "conversion", "settlement"
            ],
            "software": [
                "api", "service", "database", "endpoint", "request", "response",
                "authentication", "authorization", "deployment", "configuration"
            ],
            "agent": [
                "agent", "workflow", "task", "coordination", "planning", "execution",
                "tool", "capability", "decision", "reasoning"
            ]
        }
    
    def _load_error_signatures(self) -> Dict[str, List[str]]:
        """Load error pattern signatures from TRAIL analysis."""
        return {
            "reasoning_errors": [
                "misinterpret", "assume", "calculate", "logic", "condition"
            ],
            "execution_errors": [
                "timeout", "connection", "service", "api", "network", "resource"
            ],
            "planning_errors": [
                "coordination", "sequence", "deadlock", "dependency", "workflow"
            ]
        }