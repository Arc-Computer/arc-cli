"""
TRAIL pattern adapter for creating assumption-specific violation scenarios.

Adapts TRAIL's real-world failure patterns to test specific agent assumptions,
transforming patterns to the finance domain while preserving failure modes.
"""

import re
import random
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from collections import defaultdict

from ..core.models.scenario import Scenario
from .failure_patterns.library import FailurePattern
from .assumption_extractor import AgentAssumptions
from .trail_loader import TrailPattern


@dataclass
class AdaptationRule:
    """Rule for adapting TRAIL patterns to specific domains/assumptions."""
    pattern_type: str  # reasoning, execution, planning
    assumption_category: str  # currency, data_format, api_version, etc.
    transformation_template: str
    example_scenarios: List[str] = field(default_factory=list)
    priority: int = 5  # 1-10, higher means more relevant


@dataclass
class AdaptationMetrics:
    """Metrics for pattern adaptation process."""
    total_trail_patterns: int = 0
    adapted_scenarios: int = 0
    adaptation_rules_used: Dict[str, int] = field(default_factory=dict)
    assumption_coverage: Dict[str, int] = field(default_factory=dict)
    adaptation_time_seconds: float = 0.0


class TrailPatternAdapter:
    """Adapts TRAIL patterns to create assumption violation scenarios."""
    
    def __init__(self):
        """Initialize the adapter with transformation rules."""
        self.adaptation_rules = self._load_adaptation_rules()
        self.finance_domain_terms = self._load_finance_terms()
        self.scenario_templates = self._load_scenario_templates()
    
    async def adapt_patterns_to_assumptions(
        self,
        trail_patterns: List[FailurePattern],
        assumptions: AgentAssumptions,
        target_count: int = 35,
        assumption_types: Optional[List[str]] = None
    ) -> Tuple[List[Scenario], AdaptationMetrics]:
        """Adapt TRAIL patterns to create assumption violation scenarios.
        
        Args:
            trail_patterns: TRAIL failure patterns to adapt
            assumptions: Agent assumptions to test
            target_count: Number of scenarios to generate
            assumption_types: Specific assumption types to focus on
            
        Returns:
            Tuple of (adapted_scenarios, metrics)
        """
        import asyncio
        from datetime import datetime
        
        start_time = datetime.now()
        metrics = AdaptationMetrics(total_trail_patterns=len(trail_patterns))
        
        # Group patterns by error type for balanced selection
        patterns_by_type = defaultdict(list)
        for pattern in trail_patterns:
            patterns_by_type[pattern.category].append(pattern)
        
        # Distribute scenarios across TRAIL categories (reasoning, execution, planning)
        reasoning_count = int(target_count * 0.4)  # 40% reasoning errors (most common in TRAIL)
        execution_count = int(target_count * 0.4)   # 40% execution errors  
        planning_count = target_count - reasoning_count - execution_count  # 20% planning
        
        distribution = {
            "logic": reasoning_count,           # Maps to reasoning errors
            "infrastructure": execution_count,  # Maps to execution errors
            "multi_agent": planning_count,      # Maps to planning errors
            "data": reasoning_count // 2,       # Additional reasoning category
            "navigation": execution_count // 2  # Additional execution category
        }
        
        adapted_scenarios = []
        
        # Generate scenarios for each category
        for category, count in distribution.items():
            if count <= 0:
                continue
                
            category_patterns = patterns_by_type.get(category, [])
            if not category_patterns:
                # Use patterns from similar categories
                category_patterns = self._find_similar_patterns(category, patterns_by_type)
            
            category_scenarios = await self._adapt_category_patterns(
                category_patterns[:count * 2],  # Get more patterns than needed for selection
                assumptions,
                count,
                assumption_types
            )
            
            adapted_scenarios.extend(category_scenarios)
            
            # Update metrics
            for scenario in category_scenarios:
                rule_used = scenario.metadata.get("adaptation_rule", "unknown")
                metrics.adaptation_rules_used[rule_used] = metrics.adaptation_rules_used.get(rule_used, 0) + 1
        
        # Ensure we hit the target count
        if len(adapted_scenarios) < target_count:
            # Fill remaining with high-priority patterns
            remaining_count = target_count - len(adapted_scenarios)
            remaining_patterns = [p for p in trail_patterns if p not in [s.metadata.get("source_pattern") for s in adapted_scenarios]]
            
            additional_scenarios = await self._adapt_category_patterns(
                remaining_patterns[:remaining_count * 2],
                assumptions,
                remaining_count,
                assumption_types
            )
            adapted_scenarios.extend(additional_scenarios)
        
        # Trim to exact count and ensure diversity
        adapted_scenarios = self._ensure_diversity(adapted_scenarios, target_count)
        
        # Calculate final metrics
        metrics.adapted_scenarios = len(adapted_scenarios)
        metrics.adaptation_time_seconds = (datetime.now() - start_time).total_seconds()
        
        # Track assumption coverage
        for scenario in adapted_scenarios:
            assumption_type = scenario.metadata.get("assumption_type", "unknown")
            metrics.assumption_coverage[assumption_type] = metrics.assumption_coverage.get(assumption_type, 0) + 1
        
        return adapted_scenarios, metrics
    
    async def _adapt_category_patterns(
        self,
        patterns: List[FailurePattern],
        assumptions: AgentAssumptions,
        target_count: int,
        assumption_types: Optional[List[str]] = None
    ) -> List[Scenario]:
        """Adapt patterns from a specific category."""
        scenarios = []
        
        # Get relevant assumptions for adaptation
        relevant_assumptions = self._get_relevant_assumptions(assumptions, assumption_types)
        
        for pattern in patterns:
            if len(scenarios) >= target_count:
                break
            
            # Find best assumption to violate with this pattern
            best_assumption = self._find_best_assumption_match(pattern, relevant_assumptions)
            if not best_assumption:
                continue
            
            # Adapt pattern to create assumption violation scenario
            scenario = await self._adapt_single_pattern(pattern, best_assumption, assumptions)
            if scenario:
                scenarios.append(scenario)
        
        return scenarios[:target_count]
    
    async def _adapt_single_pattern(
        self,
        pattern: FailurePattern,
        assumption_info: Tuple[str, Any],  # (assumption_type, assumption_value)
        full_assumptions: AgentAssumptions
    ) -> Optional[Scenario]:
        """Adapt a single TRAIL pattern to create assumption violation scenario."""
        assumption_type, assumption_value = assumption_info
        
        # Get appropriate adaptation rule
        adaptation_rule = self._get_adaptation_rule(pattern.category, assumption_type)
        if not adaptation_rule:
            return None
        
        # Generate scenario using the adaptation rule
        scenario_text = self._apply_adaptation_rule(
            adaptation_rule,
            pattern,
            assumption_type,
            assumption_value,
            full_assumptions
        )
        
        if not scenario_text:
            return None
        
        # Create scenario object
        scenario = Scenario(
            scenario_id=f"trail_adapted_{pattern.id}_{assumption_type}",
            description=scenario_text,
            expected_failure_mode=pattern.expected_error,
            tags=["trail_adapted", "assumption_violation", pattern.category, assumption_type],
            metadata={
                "source_pattern": pattern.id,
                "adaptation_rule": adaptation_rule.pattern_type,
                "assumption_type": assumption_type,
                "assumption_value": str(assumption_value),
                "original_description": pattern.description,
                "trail_category": pattern.category,
                "severity": pattern.severity
            }
        )
        
        return scenario
    
    def _get_relevant_assumptions(
        self,
        assumptions: AgentAssumptions,
        assumption_types: Optional[List[str]] = None
    ) -> List[Tuple[str, Any]]:
        """Extract relevant assumptions for testing."""
        relevant = []
        
        # Currency assumptions (high priority for finance agents)
        for currency in assumptions.currencies:
            if not assumption_types or "currency" in assumption_types:
                relevant.append(("currency", currency))
        
        # Data format assumptions
        for format_type in assumptions.data_formats:
            if not assumption_types or "data_format" in assumption_types:
                relevant.append(("data_format", format_type))
        
        # API version assumptions
        for version in assumptions.api_versions:
            if not assumption_types or "api_version" in assumption_types:
                relevant.append(("api_version", version))
        
        # Timeout assumptions
        for timeout_name, timeout_value in assumptions.timeouts.items():
            if not assumption_types or "timeout" in assumption_types:
                relevant.append(("timeout", timeout_value))
        
        # Rate limit assumptions
        for limit_name, limit_value in assumptions.rate_limits.items():
            if not assumption_types or "rate_limit" in assumption_types:
                relevant.append(("rate_limit", limit_value))
        
        # Tool assumptions
        for tool in assumptions.tools:
            if not assumption_types or "tool" in assumption_types:
                relevant.append(("tool", tool))
        
        return relevant
    
    def _find_best_assumption_match(
        self,
        pattern: FailurePattern,
        assumptions: List[Tuple[str, Any]]
    ) -> Optional[Tuple[str, Any]]:
        """Find the best assumption to violate with this pattern."""
        if not assumptions:
            return None
        
        # Score assumptions based on relevance to pattern
        scored_assumptions = []
        
        for assumption_type, assumption_value in assumptions:
            score = self._calculate_assumption_relevance(pattern, assumption_type)
            scored_assumptions.append((score, assumption_type, assumption_value))
        
        # Sort by relevance and return best match
        scored_assumptions.sort(key=lambda x: x[0], reverse=True)
        
        if scored_assumptions and scored_assumptions[0][0] > 0:
            _, assumption_type, assumption_value = scored_assumptions[0]
            return (assumption_type, assumption_value)
        
        return None
    
    def _calculate_assumption_relevance(self, pattern: FailurePattern, assumption_type: str) -> int:
        """Calculate how relevant an assumption type is to a pattern."""
        relevance_map = {
            # Currency assumptions are highly relevant to calculation patterns
            ("calculation", "currency"): 10,
            ("logic", "currency"): 8,
            
            # Data format assumptions relevant to data patterns
            ("data", "data_format"): 10,
            ("infrastructure", "data_format"): 7,
            
            # API versions relevant to infrastructure patterns
            ("infrastructure", "api_version"): 9,
            ("navigation", "api_version"): 8,
            
            # Timeouts relevant to infrastructure patterns
            ("infrastructure", "timeout"): 10,
            ("navigation", "timeout"): 8,
            
            # Rate limits relevant to infrastructure patterns
            ("infrastructure", "rate_limit"): 9,
            ("navigation", "rate_limit"): 7,
            
            # Tools relevant to navigation patterns
            ("navigation", "tool"): 10,
            ("multi_agent", "tool"): 8,
        }
        
        return relevance_map.get((pattern.category, assumption_type), 3)  # Default low relevance
    
    def _get_adaptation_rule(self, pattern_category: str, assumption_type: str) -> Optional[AdaptationRule]:
        """Get adaptation rule for pattern category and assumption type."""
        key = f"{pattern_category}_{assumption_type}"
        return self.adaptation_rules.get(key)
    
    def _apply_adaptation_rule(
        self,
        rule: AdaptationRule,
        pattern: FailurePattern,
        assumption_type: str,
        assumption_value: Any,
        full_assumptions: AgentAssumptions
    ) -> str:
        """Apply adaptation rule to generate scenario text."""
        # Get finance-specific context
        finance_context = self._get_finance_context(assumption_type, assumption_value)
        
        # Apply template substitution
        scenario_text = rule.transformation_template
        
        # Substitute placeholders
        substitutions = {
            "{pattern_description}": pattern.description,
            "{assumption_value}": str(assumption_value),
            "{finance_context}": finance_context,
            "{error_description}": pattern.expected_error,
            "{violation_context}": self._generate_violation_context(assumption_type, assumption_value)
        }
        
        for placeholder, value in substitutions.items():
            scenario_text = scenario_text.replace(placeholder, value)
        
        return scenario_text
    
    def _get_finance_context(self, assumption_type: str, assumption_value: Any) -> str:
        """Generate finance-specific context for scenarios."""
        contexts = {
            "currency": [
                "processing international transaction",
                "calculating portfolio value",
                "generating financial report",
                "reconciling multi-currency accounts"
            ],
            "data_format": [
                "parsing market data feed",
                "processing trading signals",
                "analyzing financial statements",
                "importing transaction records"
            ],
            "timeout": [
                "fetching real-time market data",
                "executing high-frequency trade",
                "calculating risk metrics",
                "processing large transaction batch"
            ],
            "api_version": [
                "connecting to trading platform",
                "accessing market data service",
                "interfacing with payment processor",
                "integrating with banking API"
            ]
        }
        
        context_options = contexts.get(assumption_type, ["performing financial operation"])
        return random.choice(context_options)
    
    def _generate_violation_context(self, assumption_type: str, assumption_value: Any) -> str:
        """Generate context describing how the assumption is violated."""
        violations = {
            "currency": f"system receives amount in unexpected currency (not {assumption_value})",
            "data_format": f"data arrives in different format than expected {assumption_value}",
            "timeout": f"operation exceeds assumed {assumption_value}s timeout limit",
            "api_version": f"API returns response incompatible with expected version {assumption_value}",
            "tool": f"required tool {assumption_value} becomes unavailable or behaves differently"
        }
        
        return violations.get(assumption_type, f"assumption about {assumption_type} is violated")
    
    def _load_adaptation_rules(self) -> Dict[str, AdaptationRule]:
        """Load adaptation rules for different pattern-assumption combinations."""
        rules = {}
        
        # Currency-related adaptation rules
        rules["calculation_currency"] = AdaptationRule(
            pattern_type="reasoning",
            assumption_category="currency",
            transformation_template="While {finance_context}, agent encounters {pattern_description} when {violation_context}. Expected behavior assumes {assumption_value} but {error_description}.",
            example_scenarios=[
                "Calculate portfolio value for €10,000 but system assumes USD",
                "Process ¥150,000 transaction but currency converter expects USD"
            ],
            priority=10
        )
        
        rules["logic_currency"] = AdaptationRule(
            pattern_type="reasoning", 
            assumption_category="currency",
            transformation_template="During {finance_context}, {pattern_description} occurs because {violation_context}. System logic assumes {assumption_value} currency format.",
            priority=8
        )
        
        # Data format adaptation rules
        rules["data_data_format"] = AdaptationRule(
            pattern_type="execution",
            assumption_category="data_format", 
            transformation_template="When {finance_context}, {pattern_description} happens due to {violation_context}. Parser expects {assumption_value} format but receives different structure.",
            priority=9
        )
        
        rules["infrastructure_data_format"] = AdaptationRule(
            pattern_type="execution",
            assumption_category="data_format",
            transformation_template="While {finance_context}, infrastructure fails because {violation_context}. API contract assumes {assumption_value} but {error_description}.",
            priority=8
        )
        
        # Timeout adaptation rules
        rules["infrastructure_timeout"] = AdaptationRule(
            pattern_type="execution",
            assumption_category="timeout",
            transformation_template="During {finance_context}, {pattern_description} when {violation_context}. Service call timeout set to {assumption_value}s proves insufficient.",
            priority=9
        )
        
        # API version adaptation rules
        rules["infrastructure_api_version"] = AdaptationRule(
            pattern_type="execution",
            assumption_category="api_version",
            transformation_template="While {finance_context}, {pattern_description} because {violation_context}. Client configured for API {assumption_value} but service updated.",
            priority=8
        )
        
        rules["navigation_api_version"] = AdaptationRule(
            pattern_type="execution",
            assumption_category="api_version", 
            transformation_template="When {finance_context}, navigation fails due to {violation_context}. Tool expects API {assumption_value} but encounters version mismatch.",
            priority=7
        )
        
        # Tool adaptation rules
        rules["navigation_tool"] = AdaptationRule(
            pattern_type="planning",
            assumption_category="tool",
            transformation_template="During {finance_context}, {pattern_description} when {violation_context}. Workflow assumes {assumption_value} availability but tool is modified.",
            priority=8
        )
        
        # Generic fallback rules
        rules["multi_agent_currency"] = AdaptationRule(
            pattern_type="planning",
            assumption_category="currency",
            transformation_template="In multi-agent {finance_context}, coordination fails when {violation_context}. Agents assume consistent {assumption_value} handling.",
            priority=6
        )
        
        return rules
    
    def _load_finance_terms(self) -> Dict[str, List[str]]:
        """Load finance domain terminology for scenario generation."""
        return {
            "amounts": ["$1,000", "€2,500", "£800", "¥150,000", "$50,000", "€10,000"],
            "operations": ["trade", "settlement", "reconciliation", "valuation", "conversion"],
            "instruments": ["stocks", "bonds", "options", "futures", "currencies", "portfolios"],
            "timeframes": ["real-time", "end-of-day", "monthly", "quarterly", "intraday"],
            "regions": ["US", "European", "Asian", "global", "domestic", "international"]
        }
    
    def _load_scenario_templates(self) -> Dict[str, List[str]]:
        """Load scenario templates for different error types."""
        return {
            "reasoning": [
                "Agent misinterprets {context} due to {assumption} assumption",
                "Logic error in {context} when {assumption} is violated", 
                "Calculation fails during {context} because {assumption} is wrong"
            ],
            "execution": [
                "Service fails during {context} when {assumption} doesn't hold",
                "API call times out in {context} violating {assumption}",
                "Connection error in {context} due to {assumption} mismatch"
            ],
            "planning": [
                "Workflow breaks in {context} when {assumption} is invalid",
                "Coordination fails during {context} because {assumption} changes",
                "Task ordering wrong in {context} violating {assumption}"
            ]
        }
    
    def _find_similar_patterns(
        self,
        target_category: str,
        patterns_by_type: Dict[str, List[FailurePattern]]
    ) -> List[FailurePattern]:
        """Find patterns from similar categories when target category is empty."""
        # Category similarity mapping
        similar_categories = {
            "logic": ["data", "calculation"],
            "infrastructure": ["navigation", "auth", "security"],
            "multi_agent": ["temporal", "logic"],
            "data": ["logic", "calculation"],
            "navigation": ["infrastructure", "security"]
        }
        
        similar = similar_categories.get(target_category, [])
        patterns = []
        
        for category in similar:
            patterns.extend(patterns_by_type.get(category, []))
        
        return patterns
    
    def _ensure_diversity(self, scenarios: List[Scenario], target_count: int) -> List[Scenario]:
        """Ensure diversity in selected scenarios."""
        if len(scenarios) <= target_count:
            return scenarios
        
        # Group by assumption type for diversity
        by_assumption = defaultdict(list)
        for scenario in scenarios:
            assumption_type = scenario.metadata.get("assumption_type", "unknown")
            by_assumption[assumption_type].append(scenario)
        
        # Select diverse scenarios
        selected = []
        assumption_types = list(by_assumption.keys())
        
        # Round-robin selection for diversity
        for i in range(target_count):
            assumption_type = assumption_types[i % len(assumption_types)]
            if by_assumption[assumption_type]:
                selected.append(by_assumption[assumption_type].pop(0))
        
        return selected[:target_count]