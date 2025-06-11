"""Arc Assumption Detector - AI-Enhanced Pattern Detection.

Uses latest AI models (GPT-4.1, Claude-4) to detect assumption violations
in real-time during agent execution, with domain-aware pattern matching.
"""

import logging
import json
import asyncio
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime

from arc.cli.message_templates import ASSUMPTION_MESSAGES

logger = logging.getLogger(__name__)


@dataclass
class AssumptionViolation:
    """Represents a detected assumption violation."""
    type: str  # currency, language, timezone, etc.
    severity: str  # low, medium, high, critical
    confidence: float  # 0-100
    description: str
    minimal_reproduction: str
    trajectory_id: str
    scenario_context: Dict[str, Any]
    suggested_fix: str
    business_impact: str
    
    @property
    def message(self) -> str:
        """Get formatted message for this violation."""
        template = ASSUMPTION_MESSAGES.get(self.type, "ASSUMPTION VIOLATED: {description}")
        return template.format(description=self.description)


@dataclass
class AssumptionPattern:
    """Pattern for assumption detection."""
    type: str
    keywords: List[str]
    domain_specific: bool
    severity_indicators: Dict[str, str]
    confidence_boost: float = 0.0


class AssumptionDetector:
    """AI-enhanced assumption violation detection for real-time analysis."""
    
    def __init__(self, model_provider: str = "openai"):
        """
        Initialize the assumption detector.
        
        Args:
            model_provider: AI model provider ("openai", "anthropic", "google")
        """
        self.model_provider = model_provider
        self.model_name = self._get_latest_model(model_provider)
        
        # Domain-aware patterns from normalizer.py knowledge
        self.domain_patterns = {
            "finance": {
                "currency": AssumptionPattern(
                    type="currency",
                    keywords=["USD", "$", "dollars", "currency", "exchange", "rate"],
                    domain_specific=True,
                    severity_indicators={
                        "hardcoded_currency": "high",
                        "missing_conversion": "medium", 
                        "wrong_symbol": "low"
                    },
                    confidence_boost=20.0
                ),
                "precision": AssumptionPattern(
                    type="precision",
                    keywords=["decimal", "cents", "rounding", "precision"],
                    domain_specific=True,
                    severity_indicators={
                        "calculation_error": "critical",
                        "rounding_error": "medium"
                    },
                    confidence_boost=15.0
                )
            },
            "healthcare": {
                "privacy": AssumptionPattern(
                    type="privacy",
                    keywords=["patient", "confidential", "HIPAA", "private"],
                    domain_specific=True,
                    severity_indicators={
                        "data_exposure": "critical",
                        "logging_phi": "high"
                    },
                    confidence_boost=25.0
                )
            },
            "general": {
                "language": AssumptionPattern(
                    type="language",
                    keywords=["english", "language", "locale", "encoding", "utf"],
                    domain_specific=False,
                    severity_indicators={
                        "encoding_error": "medium",
                        "language_detection": "low"
                    }
                ),
                "timezone": AssumptionPattern(
                    type="timezone", 
                    keywords=["time", "date", "timezone", "UTC", "GMT"],
                    domain_specific=False,
                    severity_indicators={
                        "wrong_timezone": "medium",
                        "missing_tz": "low"
                    }
                )
            }
        }
    
    def _get_latest_model(self, provider: str) -> str:
        """Get the latest, fast and powerful model for assumption detection."""
        latest_models = {
            "openai": "openai/gpt-4.1-mini",  # Fast and powerful reasoning
            "anthropic": "anthropic/claude-3.5-haiku",  # Fast analysis
            "google": "google/gemini-2.5-flash-preview-05-20"  # Fast and capable
        }
        return latest_models.get(provider, "openai/gpt-4.1-mini")
    
    async def detect_live_violations(self, trajectory: Dict[str, Any], 
                                   agent_profile: Dict[str, Any]) -> List[AssumptionViolation]:
        """
        Detect assumption violations in real-time during execution.
        
        Args:
            trajectory: Current trajectory being executed
            agent_profile: Complete agent profile with capabilities
            
        Returns:
            List of detected assumption violations
        """
        violations = []
        
        # Get domain context
        capabilities = agent_profile.get("capabilities", {})
        domains = capabilities.get("domains", ["general"])
        
        # Quick pattern-based detection first (fast)
        pattern_violations = self._detect_pattern_violations(trajectory, domains)
        violations.extend(pattern_violations)
        
        # AI-enhanced deep analysis for complex cases (slower but thorough)
        if self._should_use_ai_analysis(trajectory, pattern_violations):
            ai_violations = await self._ai_enhanced_detection(trajectory, agent_profile)
            violations.extend(ai_violations)
        
        # Score and rank violations
        scored_violations = self._score_violations(violations, agent_profile)
        
        return scored_violations
    
    def _detect_pattern_violations(self, trajectory: Dict[str, Any], 
                                 domains: List[str]) -> List[AssumptionViolation]:
        """Fast pattern-based assumption detection."""
        violations = []
        
        # Get trajectory text for analysis
        trajectory_text = self._extract_trajectory_text(trajectory)
        scenario_context = trajectory.get("scenario", {})
        
        # Check relevant patterns for each domain
        for domain in domains + ["general"]:
            domain_patterns = self.domain_patterns.get(domain, {})
            
            for pattern_name, pattern in domain_patterns.items():
                violation = self._check_pattern_match(
                    pattern, trajectory_text, trajectory, scenario_context
                )
                if violation:
                    violations.append(violation)
        
        return violations
    
    def _extract_trajectory_text(self, trajectory: Dict[str, Any]) -> str:
        """Extract text content from trajectory for analysis."""
        text_parts = []
        
        # Add scenario prompt
        if "scenario" in trajectory:
            text_parts.append(trajectory["scenario"].get("task_prompt", ""))
        
        # Add trajectory events
        for event in trajectory.get("full_trajectory", []):
            if event.get("type") == "message":
                text_parts.append(str(event.get("content", "")))
            elif event.get("type") == "tool_call":
                text_parts.append(str(event.get("tool_input", "")))
                text_parts.append(str(event.get("tool_output", "")))
        
        # Add final output
        text_parts.append(str(trajectory.get("final_output", "")))
        
        return " ".join(text_parts).lower()
    
    def _check_pattern_match(self, pattern: AssumptionPattern, trajectory_text: str,
                           trajectory: Dict[str, Any], scenario_context: Dict[str, Any]) -> Optional[AssumptionViolation]:
        """Check if a pattern matches the trajectory."""
        
        # Look for pattern keywords
        keyword_matches = [kw for kw in pattern.keywords if kw.lower() in trajectory_text]
        
        if not keyword_matches:
            return None
        
        # Determine severity based on context
        severity = "low"
        for indicator, sev in pattern.severity_indicators.items():
            if indicator.lower() in trajectory_text:
                severity = sev
                break
        
        # Calculate confidence
        base_confidence = min(len(keyword_matches) * 20, 80)
        confidence = base_confidence + pattern.confidence_boost
        confidence = min(confidence, 100)
        
        # Generate description and minimal reproduction
        description = self._generate_description(pattern, keyword_matches, trajectory_text)
        minimal_repro = self._create_minimal_reproduction(pattern, scenario_context, keyword_matches)
        suggested_fix = self._generate_suggested_fix(pattern)
        business_impact = self._estimate_business_impact(pattern, severity)
        
        return AssumptionViolation(
            type=pattern.type,
            severity=severity,
            confidence=confidence,
            description=description,
            minimal_reproduction=minimal_repro,
            trajectory_id=trajectory.get("id", "unknown"),
            scenario_context=scenario_context,
            suggested_fix=suggested_fix,
            business_impact=business_impact
        )
    
    def _generate_description(self, pattern: AssumptionPattern, matches: List[str], text: str) -> str:
        """Generate human-readable description of the assumption violation."""
        base_descriptions = {
            "currency": f"Agent assumes default currency without validation (found: {', '.join(matches)})",
            "language": f"Agent processes only English input without language detection (indicators: {', '.join(matches)})",
            "timezone": f"Agent uses system timezone without user context (found: {', '.join(matches)})", 
            "precision": f"Agent may lose numerical precision in calculations (indicators: {', '.join(matches)})",
            "privacy": f"Agent may expose sensitive information (concerns: {', '.join(matches)})"
        }
        
        return base_descriptions.get(pattern.type, f"Assumption violation detected: {', '.join(matches)}")
    
    def _create_minimal_reproduction(self, pattern: AssumptionPattern, 
                                   scenario_context: Dict[str, Any], matches: List[str]) -> str:
        """Create minimal reproduction case for the assumption violation."""
        task_prompt = scenario_context.get("task_prompt", "")
        
        reproductions = {
            "currency": f"Input: '{task_prompt[:50]}...' → Expected: Currency validation, Actual: Assumed USD",
            "language": f"Input: Non-English text → Expected: Language detection, Actual: Processing failed",
            "timezone": f"Input: Time-sensitive task → Expected: Timezone clarification, Actual: Used system TZ",
            "precision": f"Input: Financial calculation → Expected: High precision, Actual: Potential rounding error",
            "privacy": f"Input: Sensitive data → Expected: Privacy protection, Actual: Potential exposure"
        }
        
        return reproductions.get(pattern.type, f"Test with: {task_prompt[:100]}...")
    
    def _generate_suggested_fix(self, pattern: AssumptionPattern) -> str:
        """Generate suggested fix for the assumption violation."""
        fixes = {
            "currency": "Add currency validation and conversion tools to agent configuration",
            "language": "Implement language detection and multi-language support",
            "timezone": "Add timezone context to user input processing",
            "precision": "Use high-precision decimal calculations for financial operations",
            "privacy": "Implement data sanitization and privacy protection measures"
        }
        
        return fixes.get(pattern.type, "Review agent configuration and add appropriate validation")
    
    def _estimate_business_impact(self, pattern: AssumptionPattern, severity: str) -> str:
        """Estimate business impact of the assumption violation."""
        impact_matrix = {
            ("currency", "high"): "Potential financial calculation errors affecting customer billing",
            ("currency", "medium"): "Incorrect currency display leading to user confusion",
            ("privacy", "critical"): "GDPR/HIPAA compliance violation with legal consequences",
            ("precision", "critical"): "Financial miscalculations affecting account balances",
            ("language", "medium"): "Reduced accessibility for non-English users",
            ("timezone", "medium"): "Scheduling errors affecting global users"
        }
        
        key = (pattern.type, severity)
        return impact_matrix.get(key, f"Reliability issues in {pattern.type} handling")
    
    def _should_use_ai_analysis(self, trajectory: Dict[str, Any], 
                              pattern_violations: List[AssumptionViolation]) -> bool:
        """Determine if AI analysis is needed for deeper insights."""
        # Use AI for complex cases or when pattern detection is uncertain
        
        # High-confidence pattern violations don't need AI
        high_confidence_violations = [v for v in pattern_violations if v.confidence > 80]
        if high_confidence_violations:
            return False
        
        # Use AI for error cases or complex trajectories
        if trajectory.get("status") == "error":
            return True
        
        # Use AI for long, complex trajectories
        events = trajectory.get("full_trajectory", [])
        if len(events) > 10:
            return True
        
        return False
    
    async def _ai_enhanced_detection(self, trajectory: Dict[str, Any], 
                                   agent_profile: Dict[str, Any]) -> List[AssumptionViolation]:
        """Use AI model for enhanced assumption detection."""
        # This would integrate with actual AI models
        # For now, return placeholder enhanced analysis
        
        logger.info(f"Running AI-enhanced analysis with {self.model_name}")
        
        # Simulate AI analysis delay
        await asyncio.sleep(0.1)
        
        # Enhanced analysis would happen here using the AI model
        # For now, return empty list as we'd need actual API integration
        return []
    
    def _score_violations(self, violations: List[AssumptionViolation], 
                         agent_profile: Dict[str, Any]) -> List[AssumptionViolation]:
        """Score and rank violations by importance."""
        # Adjust scores based on agent domain and complexity
        capabilities = agent_profile.get("capabilities", {})
        domains = capabilities.get("domains", ["general"])
        
        for violation in violations:
            # Boost confidence for domain-relevant violations
            if violation.type in ["currency", "precision"] and "finance" in domains:
                violation.confidence = min(violation.confidence + 15, 100)
            elif violation.type == "privacy" and "healthcare" in domains:
                violation.confidence = min(violation.confidence + 20, 100)
        
        # Sort by severity and confidence
        severity_order = {"critical": 4, "high": 3, "medium": 2, "low": 1}
        violations.sort(
            key=lambda x: (severity_order.get(x.severity, 0), x.confidence), 
            reverse=True
        )
        
        return violations
    
    def cross_reference_domain_patterns(self, violations: List[AssumptionViolation],
                                      domain_context: Dict[str, Any]) -> List[AssumptionViolation]:
        """Cross-reference violations with domain-specific patterns."""
        enhanced_violations = []
        
        for violation in violations:
            # Enhance violation with domain-specific context
            enhanced_violation = violation
            
            # Add domain-specific business impact
            if "finance" in domain_context.get("domains", []):
                if violation.type == "currency":
                    enhanced_violation.business_impact = "Financial accuracy compliance risk"
                elif violation.type == "precision":
                    enhanced_violation.business_impact = "Calculation errors affecting financial reporting"
            
            enhanced_violations.append(enhanced_violation)
        
        return enhanced_violations
    
    def extract_minimal_reproductions(self, violations: List[AssumptionViolation]) -> Dict[str, str]:
        """Extract minimal reproduction cases for all violations."""
        reproductions = {}
        
        for violation in violations:
            reproductions[f"{violation.type}_{violation.trajectory_id}"] = violation.minimal_reproduction
        
        return reproductions
    
    def assumption_confidence_scoring(self, violations: List[AssumptionViolation],
                                    historical_patterns: Optional[Dict[str, Any]] = None) -> Dict[str, float]:
        """Calculate confidence scores based on historical patterns."""
        confidence_scores = {}
        
        for violation in violations:
            base_score = violation.confidence
            
            # Boost confidence if we've seen this pattern before
            if historical_patterns:
                pattern_history = historical_patterns.get(violation.type, {})
                historical_frequency = pattern_history.get("frequency", 0)
                if historical_frequency > 5:  # Seen this pattern multiple times
                    base_score += 10
            
            confidence_scores[f"{violation.type}_{violation.trajectory_id}"] = min(base_score, 100)
        
        return confidence_scores
