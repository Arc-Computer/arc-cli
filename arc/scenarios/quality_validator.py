"""
TRAIL-based Quality Validator for scenario generation.

Validates scenario quality using criteria derived from real-world agent failure
patterns to ensure realistic and effective testing scenarios.
"""

import re
import logging
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum

from ..core.models.scenario import Scenario
from .trail_loader import FailurePattern

logger = logging.getLogger(__name__)


class QualityDimension(Enum):
    """Quality dimensions for scenario validation."""
    
    ASSUMPTION_CLARITY = "assumption_clarity"
    FAILURE_REALISM = "failure_realism"
    RECOVERABILITY = "recoverability"
    DOMAIN_RELEVANCE = "domain_relevance"
    COMPLEXITY_BALANCE = "complexity_balance"
    INSTRUCTION_CLARITY = "instruction_clarity"
    EXPECTED_OUTPUT_QUALITY = "expected_output_quality"


@dataclass
class QualityScore:
    """Individual quality dimension score."""
    
    dimension: QualityDimension
    score: float  # 0.0 to 1.0
    feedback: str
    weight: float = 1.0
    
    def weighted_score(self) -> float:
        """Get weighted score."""
        return self.score * self.weight


@dataclass
class ScenarioQualityResult:
    """Complete quality assessment result for a scenario."""
    
    scenario_id: str
    overall_score: float
    dimension_scores: List[QualityScore]
    passed_threshold: bool
    recommendations: List[str]
    trail_based_insights: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "scenario_id": self.scenario_id,
            "overall_score": round(self.overall_score, 3),
            "passed_threshold": self.passed_threshold,
            "dimension_scores": {
                score.dimension.value: {
                    "score": round(score.score, 3),
                    "weighted_score": round(score.weighted_score(), 3),
                    "feedback": score.feedback
                }
                for score in self.dimension_scores
            },
            "recommendations": self.recommendations,
            "trail_insights": self.trail_based_insights
        }


class TrailQualityValidator:
    """Validates scenario quality using TRAIL-based criteria."""
    
    def __init__(
        self,
        quality_threshold: float = 0.7,
        trail_patterns: Optional[List[FailurePattern]] = None
    ):
        """Initialize the quality validator.
        
        Args:
            quality_threshold: Minimum quality score (0.0-1.0)
            trail_patterns: TRAIL patterns for validation insights
        """
        self.quality_threshold = quality_threshold
        self.trail_patterns = trail_patterns or []
        
        # Quality dimension weights based on TRAIL analysis
        self.dimension_weights = {
            QualityDimension.ASSUMPTION_CLARITY: 1.2,  # Most important for assumption testing
            QualityDimension.FAILURE_REALISM: 1.1,     # Critical for TRAIL-based scenarios
            QualityDimension.RECOVERABILITY: 1.0,      # Standard importance
            QualityDimension.DOMAIN_RELEVANCE: 0.9,    # Slightly less critical
            QualityDimension.COMPLEXITY_BALANCE: 0.8,  # Important but not critical
            QualityDimension.INSTRUCTION_CLARITY: 1.0, # Standard importance
            QualityDimension.EXPECTED_OUTPUT_QUALITY: 0.9  # Less critical than inputs
        }
        
        # TRAIL pattern insights for validation
        self._build_trail_insights()
        
        # Quality patterns learned from TRAIL data
        self.quality_patterns = {
            "assumption_violation_indicators": [
                "assumes", "expects", "default", "typical", "standard",
                "usual", "normal", "common", "always", "never"
            ],
            "failure_realism_indicators": [
                "timeout", "unavailable", "error", "fails", "missing",
                "incorrect", "mismatch", "conflict", "corruption"
            ],
            "recovery_indicators": [
                "retry", "fallback", "alternative", "backup", "recover",
                "handle", "graceful", "degrade", "warn", "detect"
            ],
            "clarity_indicators": [
                "specific", "clear", "explicit", "detailed", "precise",
                "unambiguous", "concrete", "measurable"
            ]
        }
    
    def _build_trail_insights(self) -> None:
        """Build insights from TRAIL patterns for validation."""
        self.trail_insights = {
            "common_failure_modes": {},
            "recovery_patterns": {},
            "complexity_distribution": {},
            "domain_characteristics": {}
        }
        
        if not self.trail_patterns:
            return
        
        # Analyze TRAIL patterns for validation insights
        for pattern in self.trail_patterns:
            # Common failure modes
            failure_mode = pattern.failure_mode
            self.trail_insights["common_failure_modes"][failure_mode] = (
                self.trail_insights["common_failure_modes"].get(failure_mode, 0) + 1
            )
            
            # Recovery patterns
            recovery_key = f"{pattern.error_type}_{pattern.recovery_possible}"
            self.trail_insights["recovery_patterns"][recovery_key] = (
                self.trail_insights["recovery_patterns"].get(recovery_key, 0) + 1
            )
            
            # Complexity distribution
            complexity = pattern.complexity
            self.trail_insights["complexity_distribution"][complexity] = (
                self.trail_insights["complexity_distribution"].get(complexity, 0) + 1
            )
            
            # Domain characteristics
            domain = pattern.domain
            if domain not in self.trail_insights["domain_characteristics"]:
                self.trail_insights["domain_characteristics"][domain] = {
                    "error_types": set(),
                    "common_words": set(),
                    "complexity_levels": set()
                }
            
            self.trail_insights["domain_characteristics"][domain]["error_types"].add(pattern.error_type)
            self.trail_insights["domain_characteristics"][domain]["complexity_levels"].add(pattern.complexity)
            
            # Extract common words from descriptions
            words = re.findall(r'\b\w+\b', pattern.description.lower())
            self.trail_insights["domain_characteristics"][domain]["common_words"].update(
                word for word in words if len(word) > 3
            )
    
    def validate_scenario(self, scenario: Scenario) -> ScenarioQualityResult:
        """Validate scenario quality using TRAIL-based criteria.
        
        Args:
            scenario: Scenario to validate
            
        Returns:
            Quality assessment result
        """
        dimension_scores = []
        
        # Evaluate each quality dimension
        for dimension in QualityDimension:
            score = self._evaluate_dimension(scenario, dimension)
            score.weight = self.dimension_weights.get(dimension, 1.0)
            dimension_scores.append(score)
        
        # Calculate overall score
        total_weighted_score = sum(score.weighted_score() for score in dimension_scores)
        total_weight = sum(score.weight for score in dimension_scores)
        overall_score = total_weighted_score / total_weight if total_weight > 0 else 0.0
        
        # Determine if passed threshold
        passed_threshold = overall_score >= self.quality_threshold
        
        # Generate recommendations
        recommendations = self._generate_recommendations(scenario, dimension_scores)
        
        # Add TRAIL-based insights
        trail_insights = self._get_trail_insights_for_scenario(scenario)
        
        return ScenarioQualityResult(
            scenario_id=scenario.id,
            overall_score=overall_score,
            dimension_scores=dimension_scores,
            passed_threshold=passed_threshold,
            recommendations=recommendations,
            trail_based_insights=trail_insights
        )
    
    def _evaluate_dimension(self, scenario: Scenario, dimension: QualityDimension) -> QualityScore:
        """Evaluate a specific quality dimension."""
        evaluator_map = {
            QualityDimension.ASSUMPTION_CLARITY: self._evaluate_assumption_clarity,
            QualityDimension.FAILURE_REALISM: self._evaluate_failure_realism,
            QualityDimension.RECOVERABILITY: self._evaluate_recoverability,
            QualityDimension.DOMAIN_RELEVANCE: self._evaluate_domain_relevance,
            QualityDimension.COMPLEXITY_BALANCE: self._evaluate_complexity_balance,
            QualityDimension.INSTRUCTION_CLARITY: self._evaluate_instruction_clarity,
            QualityDimension.EXPECTED_OUTPUT_QUALITY: self._evaluate_expected_output_quality
        }
        
        evaluator = evaluator_map.get(dimension)
        if evaluator:
            return evaluator(scenario)
        else:
            return QualityScore(
                dimension=dimension,
                score=0.5,
                feedback="Evaluation method not implemented"
            )
    
    def _evaluate_assumption_clarity(self, scenario: Scenario) -> QualityScore:
        """Evaluate how clearly the scenario violates assumptions."""
        score = 0.0
        feedback_parts = []
        
        # Check for assumption violation indicators
        text_content = f"{scenario.description} {scenario.instructions}".lower()
        assumption_indicators = self.quality_patterns["assumption_violation_indicators"]
        
        indicator_count = sum(1 for indicator in assumption_indicators if indicator in text_content)
        if indicator_count > 0:
            score += 0.3
            feedback_parts.append(f"Found {indicator_count} assumption indicators")
        
        # Check metadata for assumption information
        assumption_metadata = scenario.metadata.get("assumption_violated")
        if assumption_metadata:
            score += 0.4
            feedback_parts.append("Clear assumption violation metadata")
        
        # Check for specific assumption types in tags
        assumption_tags = [tag for tag in scenario.tags if "assumption" in tag]
        if assumption_tags:
            score += 0.3
            feedback_parts.append(f"Assumption tags: {assumption_tags}")
        
        # Bonus for TRAIL-adapted scenarios (real-world basis)
        if "trail_adapted" in scenario.tags:
            score += 0.2
            feedback_parts.append("TRAIL-adapted scenario")
        
        # Cap at 1.0
        score = min(score, 1.0)
        
        feedback = "Assumption clarity: " + "; ".join(feedback_parts) if feedback_parts else "No clear assumption violation"
        
        return QualityScore(
            dimension=QualityDimension.ASSUMPTION_CLARITY,
            score=score,
            feedback=feedback
        )
    
    def _evaluate_failure_realism(self, scenario: Scenario) -> QualityScore:
        """Evaluate how realistic the failure scenario is."""
        score = 0.0
        feedback_parts = []
        
        # Check for realistic failure indicators
        text_content = f"{scenario.description} {scenario.instructions}".lower()
        failure_indicators = self.quality_patterns["failure_realism_indicators"]
        
        indicator_count = sum(1 for indicator in failure_indicators if indicator in text_content)
        if indicator_count > 0:
            score += min(indicator_count * 0.15, 0.6)
            feedback_parts.append(f"Found {indicator_count} failure realism indicators")
        
        # Check if based on TRAIL pattern (real-world failure)
        trail_pattern_id = scenario.metadata.get("trail_pattern_id")
        if trail_pattern_id:
            score += 0.4
            feedback_parts.append("Based on real TRAIL failure pattern")
            
            # Find the original pattern for more insights
            original_pattern = self._find_trail_pattern(trail_pattern_id)
            if original_pattern:
                # Boost for common failure modes
                if original_pattern.failure_mode in self.trail_insights.get("common_failure_modes", {}):
                    score += 0.2
                    feedback_parts.append("Common real-world failure mode")
        
        # Check for specific technical details
        technical_terms = ["api", "timeout", "error", "service", "network", "database", "authentication"]
        technical_count = sum(1 for term in technical_terms if term in text_content)
        if technical_count > 0:
            score += min(technical_count * 0.05, 0.2)
            feedback_parts.append(f"Technical specificity ({technical_count} terms)")
        
        score = min(score, 1.0)
        
        feedback = "Failure realism: " + "; ".join(feedback_parts) if feedback_parts else "Generic failure scenario"
        
        return QualityScore(
            dimension=QualityDimension.FAILURE_REALISM,
            score=score,
            feedback=feedback
        )
    
    def _evaluate_recoverability(self, scenario: Scenario) -> QualityScore:
        """Evaluate if the scenario allows for recovery/graceful handling."""
        score = 0.0
        feedback_parts = []
        
        # Check for recovery indicators
        text_content = f"{scenario.description} {scenario.instructions}".lower()
        recovery_indicators = self.quality_patterns["recovery_indicators"]
        
        recovery_count = sum(1 for indicator in recovery_indicators if indicator in text_content)
        if recovery_count > 0:
            score += min(recovery_count * 0.2, 0.6)
            feedback_parts.append(f"Found {recovery_count} recovery indicators")
        
        # Check expected outputs for recovery expectations
        for output in scenario.expected_outputs:
            output_lower = output.lower()
            if any(indicator in output_lower for indicator in recovery_indicators):
                score += 0.3
                feedback_parts.append("Expected outputs mention recovery")
                break
        
        # Check TRAIL pattern recovery possibility
        recovery_possible = scenario.metadata.get("recovery_possible")
        if recovery_possible is True:
            score += 0.3
            feedback_parts.append("TRAIL pattern indicates recovery possible")
        elif recovery_possible is False:
            score += 0.1  # Still some value for catastrophic scenarios
            feedback_parts.append("TRAIL pattern indicates limited recovery")
        
        # Avoid scenarios that are too easy or too impossible
        if score > 0.9:
            score = 0.8  # Too easy
            feedback_parts.append("May be too easily recoverable")
        elif score < 0.2:
            feedback_parts.append("May be too difficult to recover from")
        
        score = min(score, 1.0)
        
        feedback = "Recoverability: " + "; ".join(feedback_parts) if feedback_parts else "Recovery potential unclear"
        
        return QualityScore(
            dimension=QualityDimension.RECOVERABILITY,
            score=score,
            feedback=feedback
        )
    
    def _evaluate_domain_relevance(self, scenario: Scenario) -> QualityScore:
        """Evaluate domain relevance and consistency."""
        score = 0.0
        feedback_parts = []
        
        # Check domain tags
        domain_tags = [tag for tag in scenario.tags if tag in ["finance", "web", "api", "data", "software"]]
        if domain_tags:
            score += 0.3
            feedback_parts.append(f"Domain tags: {domain_tags}")
        
        # Check for domain-specific terminology
        text_content = f"{scenario.description} {scenario.instructions}".lower()
        
        domain_terms = {
            "finance": ["currency", "payment", "transaction", "trading", "investment", "portfolio"],
            "web": ["browser", "html", "scraping", "navigation", "page", "website"],
            "api": ["endpoint", "request", "response", "rest", "http", "service"],
            "data": ["database", "query", "analysis", "processing", "extraction", "format"]
        }
        
        for domain, terms in domain_terms.items():
            term_count = sum(1 for term in terms if term in text_content)
            if term_count > 0:
                score += min(term_count * 0.1, 0.4)
                feedback_parts.append(f"{domain} domain terms: {term_count}")
                break
        
        # Check TRAIL pattern domain consistency
        pattern_domain = scenario.metadata.get("pattern_domain")
        if pattern_domain and any(pattern_domain in tag for tag in scenario.tags):
            score += 0.3
            feedback_parts.append("TRAIL pattern domain consistency")
        
        score = min(score, 1.0)
        
        feedback = "Domain relevance: " + "; ".join(feedback_parts) if feedback_parts else "Domain unclear"
        
        return QualityScore(
            dimension=QualityDimension.DOMAIN_RELEVANCE,
            score=score,
            feedback=feedback
        )
    
    def _evaluate_complexity_balance(self, scenario: Scenario) -> QualityScore:
        """Evaluate appropriate complexity balance."""
        score = 0.0
        feedback_parts = []
        
        # Analyze text complexity
        text_content = f"{scenario.description} {scenario.instructions}"
        
        # Length considerations
        description_length = len(scenario.description)
        instruction_length = len(scenario.instructions)
        
        if 50 <= description_length <= 300:
            score += 0.3
            feedback_parts.append("Appropriate description length")
        elif description_length < 50:
            feedback_parts.append("Description may be too brief")
        else:
            feedback_parts.append("Description may be too verbose")
        
        if 100 <= instruction_length <= 500:
            score += 0.3
            feedback_parts.append("Appropriate instruction length")
        
        # Check for step complexity
        instruction_sentences = len(re.findall(r'[.!?]+', scenario.instructions))
        if 2 <= instruction_sentences <= 5:
            score += 0.2
            feedback_parts.append("Good instruction complexity")
        
        # Check TRAIL pattern complexity
        pattern_complexity = scenario.metadata.get("pattern_complexity")
        if pattern_complexity == "medium":
            score += 0.2
            feedback_parts.append("TRAIL pattern has medium complexity")
        elif pattern_complexity in ["low", "high"]:
            score += 0.1
            feedback_parts.append(f"TRAIL pattern has {pattern_complexity} complexity")
        
        score = min(score, 1.0)
        
        feedback = "Complexity balance: " + "; ".join(feedback_parts) if feedback_parts else "Complexity assessment unclear"
        
        return QualityScore(
            dimension=QualityDimension.COMPLEXITY_BALANCE,
            score=score,
            feedback=feedback
        )
    
    def _evaluate_instruction_clarity(self, scenario: Scenario) -> QualityScore:
        """Evaluate clarity and specificity of instructions."""
        score = 0.0
        feedback_parts = []
        
        instructions = scenario.instructions.lower()
        
        # Check for clarity indicators
        clarity_indicators = self.quality_patterns["clarity_indicators"]
        clarity_count = sum(1 for indicator in clarity_indicators if indicator in instructions)
        if clarity_count > 0:
            score += min(clarity_count * 0.15, 0.4)
            feedback_parts.append(f"Found {clarity_count} clarity indicators")
        
        # Check for actionable verbs
        action_verbs = ["process", "handle", "execute", "analyze", "generate", "convert", "calculate"]
        action_count = sum(1 for verb in action_verbs if verb in instructions)
        if action_count > 0:
            score += min(action_count * 0.1, 0.3)
            feedback_parts.append(f"Clear action verbs: {action_count}")
        
        # Check for specific values/examples
        if re.search(r'\$\d+|\d+\.\d+|[A-Z]{3}|\d+s', scenario.instructions):
            score += 0.3
            feedback_parts.append("Contains specific values/examples")
        
        score = min(score, 1.0)
        
        feedback = "Instruction clarity: " + "; ".join(feedback_parts) if feedback_parts else "Instructions need clarity"
        
        return QualityScore(
            dimension=QualityDimension.INSTRUCTION_CLARITY,
            score=score,
            feedback=feedback
        )
    
    def _evaluate_expected_output_quality(self, scenario: Scenario) -> QualityScore:
        """Evaluate quality of expected outputs."""
        score = 0.0
        feedback_parts = []
        
        if not scenario.expected_outputs:
            return QualityScore(
                dimension=QualityDimension.EXPECTED_OUTPUT_QUALITY,
                score=0.0,
                feedback="No expected outputs specified"
            )
        
        # Check number of expected outputs
        output_count = len(scenario.expected_outputs)
        if 2 <= output_count <= 5:
            score += 0.3
            feedback_parts.append(f"Good number of outputs ({output_count})")
        elif output_count == 1:
            score += 0.1
            feedback_parts.append("Single expected output")
        else:
            feedback_parts.append("Too many expected outputs")
        
        # Check for specific, measurable outputs
        specific_outputs = 0
        for output in scenario.expected_outputs:
            output_lower = output.lower()
            if any(indicator in output_lower for indicator in ["should", "must", "detect", "recognize"]):
                specific_outputs += 1
        
        if specific_outputs > 0:
            score += min(specific_outputs * 0.2, 0.4)
            feedback_parts.append(f"Specific expectations: {specific_outputs}")
        
        # Check for assumption-related outputs
        assumption_outputs = sum(
            1 for output in scenario.expected_outputs
            if "assumption" in output.lower() or "violation" in output.lower()
        )
        if assumption_outputs > 0:
            score += 0.3
            feedback_parts.append("Assumption-related outputs present")
        
        score = min(score, 1.0)
        
        feedback = "Expected output quality: " + "; ".join(feedback_parts)
        
        return QualityScore(
            dimension=QualityDimension.EXPECTED_OUTPUT_QUALITY,
            score=score,
            feedback=feedback
        )
    
    def _generate_recommendations(
        self,
        scenario: Scenario,
        dimension_scores: List[QualityScore]
    ) -> List[str]:
        """Generate improvement recommendations."""
        recommendations = []
        
        # Find dimensions with low scores
        low_scoring_dimensions = [
            score for score in dimension_scores
            if score.score < 0.6
        ]
        
        for low_score in low_scoring_dimensions:
            if low_score.dimension == QualityDimension.ASSUMPTION_CLARITY:
                recommendations.append(
                    "Make assumption violation more explicit in description and metadata"
                )
            elif low_score.dimension == QualityDimension.FAILURE_REALISM:
                recommendations.append(
                    "Add more realistic technical details or base on real failure patterns"
                )
            elif low_score.dimension == QualityDimension.RECOVERABILITY:
                recommendations.append(
                    "Clarify how the agent can potentially recover from this scenario"
                )
            elif low_score.dimension == QualityDimension.DOMAIN_RELEVANCE:
                recommendations.append(
                    "Add domain-specific terminology and context"
                )
            elif low_score.dimension == QualityDimension.COMPLEXITY_BALANCE:
                recommendations.append(
                    "Adjust scenario complexity - may be too simple or too complex"
                )
            elif low_score.dimension == QualityDimension.INSTRUCTION_CLARITY:
                recommendations.append(
                    "Make instructions more specific with concrete examples"
                )
            elif low_score.dimension == QualityDimension.EXPECTED_OUTPUT_QUALITY:
                recommendations.append(
                    "Improve expected outputs with specific, measurable criteria"
                )
        
        return recommendations
    
    def _get_trail_insights_for_scenario(self, scenario: Scenario) -> Dict[str, Any]:
        """Get TRAIL-based insights for the scenario."""
        insights = {}
        
        # Get pattern information if available
        trail_pattern_id = scenario.metadata.get("trail_pattern_id")
        if trail_pattern_id:
            original_pattern = self._find_trail_pattern(trail_pattern_id)
            if original_pattern:
                insights["original_pattern"] = {
                    "error_type": original_pattern.error_type,
                    "domain": original_pattern.domain,
                    "complexity": original_pattern.complexity,
                    "failure_mode": original_pattern.failure_mode,
                    "recovery_possible": original_pattern.recovery_possible
                }
                
                # Add commonality insights
                failure_mode_count = self.trail_insights.get("common_failure_modes", {}).get(
                    original_pattern.failure_mode, 0
                )
                if failure_mode_count > 1:
                    insights["failure_commonality"] = f"This failure mode appears in {failure_mode_count} TRAIL patterns"
        
        # Add general TRAIL insights
        if self.trail_insights:
            insights["dataset_context"] = {
                "total_patterns": len(self.trail_patterns),
                "common_domains": list(self.trail_insights.get("domain_characteristics", {}).keys())[:3],
                "complexity_distribution": self.trail_insights.get("complexity_distribution", {})
            }
        
        return insights
    
    def _find_trail_pattern(self, pattern_id: str) -> Optional[FailurePattern]:
        """Find TRAIL pattern by ID."""
        for pattern in self.trail_patterns:
            if pattern.id == pattern_id:
                return pattern
        return None
    
    def get_quality_statistics(self) -> Dict[str, Any]:
        """Get quality validation statistics."""
        return {
            "quality_threshold": self.quality_threshold,
            "dimension_weights": {dim.value: weight for dim, weight in self.dimension_weights.items()},
            "trail_patterns_loaded": len(self.trail_patterns),
            "trail_insights_available": bool(self.trail_insights),
            "quality_patterns": {k: len(v) for k, v in self.quality_patterns.items()}
        }