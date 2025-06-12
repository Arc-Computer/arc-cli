"""
TRAIL Pattern Adapter for assumption-based scenario generation.

Adapts TRAIL failure patterns to create assumption violation scenarios
tailored to specific agent configurations and domains.
"""

import random
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
import logging

from .trail_loader import FailurePattern, TrailDatasetLoader
from .assumption_extractor import AgentAssumptions
from ..core.models.scenario import Scenario

logger = logging.getLogger(__name__)


@dataclass
class AdaptationMapping:
    """Maps TRAIL patterns to assumption violations."""
    
    trail_pattern: FailurePattern
    target_assumption: str
    assumption_type: str  # currency, api_version, timeout, etc.
    adaptation_strategy: str
    confidence_score: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AdaptationResult:
    """Result of adapting TRAIL patterns to scenarios."""
    
    scenarios: List[Scenario]
    adaptations_used: List[AdaptationMapping]
    adaptation_metrics: Dict[str, Any]
    success_rate: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "scenario_count": len(self.scenarios),
            "adaptations_count": len(self.adaptations_used),
            "metrics": self.adaptation_metrics,
            "success_rate": round(self.success_rate, 3)
        }


class TrailPatternAdapter:
    """Adapts TRAIL patterns to agent assumptions for scenario generation."""
    
    def __init__(self, trail_loader: TrailDatasetLoader, random_seed: Optional[int] = None):
        """Initialize the adapter.
        
        Args:
            trail_loader: Loaded TRAIL dataset
            random_seed: Seed for reproducible adaptation
        """
        self.trail_loader = trail_loader
        if random_seed is not None:
            random.seed(random_seed)
        
        # Adaptation strategies for different assumption types
        self.adaptation_strategies = {
            "currency": self._adapt_currency_assumption,
            "api_version": self._adapt_api_version_assumption,
            "timeout": self._adapt_timeout_assumption,
            "data_format": self._adapt_data_format_assumption,
            "error_handling": self._adapt_error_handling_assumption,
            "rate_limit": self._adapt_rate_limit_assumption
        }
        
        # Template phrases for different violation types
        self.violation_templates = {
            "currency": [
                "Process payment in {currency}",
                "Convert {amount} to {target_currency}",
                "Calculate tax on {currency} amount",
                "Generate report with {currency} totals",
                "Analyze budget in {currency}"
            ],
            "api_version": [
                "Call API using version {version}",
                "Integrate with {service} v{version}",
                "Handle deprecated endpoint {version}",
                "Process response from {service} {version}"
            ],
            "timeout": [
                "Process request with {timeout}s timeout",
                "Handle slow response after {timeout} seconds",
                "Execute operation within {timeout}s limit",
                "Retry failed call after {timeout}s"
            ],
            "data_format": [
                "Parse {format} data response",
                "Convert data to {format} format",
                "Process malformed {format} input",
                "Validate {format} structure"
            ]
        }
    
    async def adapt_patterns_to_assumptions(
        self,
        assumptions: AgentAssumptions,
        agent_config: Dict[str, Any],
        target_count: int = 35,
        focus_types: Optional[List[str]] = None
    ) -> AdaptationResult:
        """Adapt TRAIL patterns to create assumption violation scenarios.
        
        Args:
            assumptions: Extracted agent assumptions
            agent_config: Original agent configuration
            target_count: Number of scenarios to generate
            focus_types: Error types to focus on (reasoning, execution, planning)
            
        Returns:
            Adaptation result with generated scenarios
        """
        if not self.trail_loader.is_loaded():
            await self.trail_loader.load_patterns()
        
        # Find relevant TRAIL patterns for each assumption
        adaptation_mappings = self._map_patterns_to_assumptions(
            assumptions, focus_types or ["reasoning", "execution", "planning"]
        )
        
        # Generate scenarios from mappings
        scenarios = []
        adaptations_used = []
        adaptation_metrics = {
            "patterns_considered": len(self.trail_loader.get_all_patterns()),
            "mappings_found": len(adaptation_mappings),
            "assumptions_covered": set(),
            "error_types_used": set(),
            "adaptation_strategies": {}
        }
        
        # Select best mappings and adapt to scenarios
        selected_mappings = self._select_best_mappings(
            adaptation_mappings, target_count
        )
        
        for mapping in selected_mappings:
            try:
                scenario = await self._create_scenario_from_mapping(
                    mapping, assumptions, agent_config
                )
                if scenario:
                    scenarios.append(scenario)
                    adaptations_used.append(mapping)
                    
                    # Update metrics
                    adaptation_metrics["assumptions_covered"].add(mapping.target_assumption)
                    adaptation_metrics["error_types_used"].add(mapping.trail_pattern.error_type)
                    
                    strategy = mapping.adaptation_strategy
                    adaptation_metrics["adaptation_strategies"][strategy] = (
                        adaptation_metrics["adaptation_strategies"].get(strategy, 0) + 1
                    )
                    
            except Exception as e:
                logger.warning(f"Failed to adapt pattern {mapping.trail_pattern.id}: {e}")
        
        # Convert sets to lists for serialization
        adaptation_metrics["assumptions_covered"] = list(adaptation_metrics["assumptions_covered"])
        adaptation_metrics["error_types_used"] = list(adaptation_metrics["error_types_used"])
        
        success_rate = len(scenarios) / max(len(selected_mappings), 1)
        
        return AdaptationResult(
            scenarios=scenarios,
            adaptations_used=adaptations_used,
            adaptation_metrics=adaptation_metrics,
            success_rate=success_rate
        )
    
    def _map_patterns_to_assumptions(
        self,
        assumptions: AgentAssumptions,
        focus_types: List[str]
    ) -> List[AdaptationMapping]:
        """Map TRAIL patterns to agent assumptions."""
        mappings = []
        
        # Get patterns by focused error types
        relevant_patterns = []
        for error_type in focus_types:
            relevant_patterns.extend(
                self.trail_loader.get_patterns_by_type(error_type)
            )
        
        # Map patterns to each assumption type
        assumption_mappings = {
            "currencies": ("currency", list(assumptions.currencies)),
            "api_versions": ("api_version", list(assumptions.api_versions)),
            "timeouts": ("timeout", list(assumptions.timeouts.keys())),
            "data_formats": ("data_format", list(assumptions.data_formats)),
            "error_handling": ("error_handling", list(assumptions.error_handling)),
            "rate_limits": ("rate_limit", list(assumptions.rate_limits.keys()))
        }
        
        for assumption_attr, (assumption_type, assumption_values) in assumption_mappings.items():
            if not assumption_values:
                continue
                
            # Find patterns that can be adapted to this assumption type
            for pattern in relevant_patterns:
                confidence = self._calculate_adaptation_confidence(
                    pattern, assumption_type, assumption_values
                )
                
                if confidence > 0.3:  # Minimum confidence threshold
                    for assumption_value in assumption_values[:3]:  # Limit per assumption
                        mappings.append(AdaptationMapping(
                            trail_pattern=pattern,
                            target_assumption=assumption_value,
                            assumption_type=assumption_type,
                            adaptation_strategy=self._get_adaptation_strategy(
                                pattern, assumption_type
                            ),
                            confidence_score=confidence,
                            metadata={
                                "assumption_attr": assumption_attr,
                                "pattern_domain": pattern.domain
                            }
                        ))
        
        # Sort by confidence score
        mappings.sort(key=lambda m: m.confidence_score, reverse=True)
        return mappings
    
    def _calculate_adaptation_confidence(
        self,
        pattern: FailurePattern,
        assumption_type: str,
        assumption_values: List[str]
    ) -> float:
        """Calculate confidence score for adapting pattern to assumption."""
        confidence = 0.0
        
        # Base confidence by pattern type and assumption compatibility
        compatibility_matrix = {
            ("reasoning", "currency"): 0.8,
            ("reasoning", "data_format"): 0.9,
            ("execution", "api_version"): 0.9,
            ("execution", "timeout"): 0.95,
            ("execution", "rate_limit"): 0.9,
            ("planning", "error_handling"): 0.8,
            ("planning", "timeout"): 0.7
        }
        
        base_confidence = compatibility_matrix.get(
            (pattern.error_type, assumption_type), 0.5
        )
        confidence += base_confidence
        
        # Boost for finance domain patterns
        if pattern.domain == "finance" and assumption_type == "currency":
            confidence += 0.2
        
        # Boost for recoverable patterns (better for testing)
        if pattern.recovery_possible:
            confidence += 0.1
        
        # Boost for medium complexity (not too simple, not too complex)
        if pattern.complexity == "medium":
            confidence += 0.1
        
        # Penalize if pattern is too generic
        if len(pattern.description) < 50:
            confidence -= 0.2
        
        return min(confidence, 1.0)
    
    def _get_adaptation_strategy(self, pattern: FailurePattern, assumption_type: str) -> str:
        """Get adaptation strategy for pattern and assumption type."""
        strategies = {
            ("reasoning", "currency"): "currency_confusion",
            ("reasoning", "data_format"): "format_misinterpretation", 
            ("execution", "api_version"): "version_mismatch",
            ("execution", "timeout"): "timeout_violation",
            ("execution", "rate_limit"): "limit_exceeded",
            ("planning", "error_handling"): "recovery_failure"
        }
        
        return strategies.get(
            (pattern.error_type, assumption_type),
            "generic_adaptation"
        )
    
    def _select_best_mappings(
        self,
        mappings: List[AdaptationMapping],
        target_count: int
    ) -> List[AdaptationMapping]:
        """Select best mappings for scenario generation."""
        if len(mappings) <= target_count:
            return mappings
        
        # Ensure diversity across assumption types and error types
        selected = []
        seen_combinations = set()
        
        # First pass: select highest confidence mappings with diversity
        for mapping in mappings:
            if len(selected) >= target_count:
                break
                
            combo_key = (mapping.assumption_type, mapping.trail_pattern.error_type)
            
            # Prefer diverse combinations
            if combo_key not in seen_combinations or len(selected) < target_count // 2:
                selected.append(mapping)
                seen_combinations.add(combo_key)
        
        # Second pass: fill remaining slots with highest confidence
        remaining_slots = target_count - len(selected)
        if remaining_slots > 0:
            remaining_mappings = [m for m in mappings if m not in selected]
            selected.extend(remaining_mappings[:remaining_slots])
        
        return selected[:target_count]
    
    async def _create_scenario_from_mapping(
        self,
        mapping: AdaptationMapping,
        assumptions: AgentAssumptions,
        agent_config: Dict[str, Any]
    ) -> Optional[Scenario]:
        """Create scenario from adaptation mapping."""
        try:
            # Get adaptation strategy
            strategy_func = self.adaptation_strategies.get(
                mapping.assumption_type,
                self._adapt_generic_assumption
            )
            
            # Apply adaptation strategy
            scenario_data = strategy_func(mapping, assumptions, agent_config)
            
            if not scenario_data:
                return None
            
            # Create scenario object
            scenario = Scenario(
                id=f"trail_{mapping.trail_pattern.id}_{mapping.assumption_type}",
                description=scenario_data["description"],
                instructions=scenario_data["instructions"],
                expected_outputs=scenario_data.get("expected_outputs", []),
                tools_required=assumptions.tools,
                complexity="medium",
                tags=[
                    "trail_adapted",
                    f"assumption_{mapping.assumption_type}",
                    f"error_{mapping.trail_pattern.error_type}",
                    mapping.trail_pattern.domain
                ],
                metadata={
                    "trail_pattern_id": mapping.trail_pattern.id,
                    "assumption_violated": mapping.target_assumption,
                    "assumption_type": mapping.assumption_type,
                    "adaptation_strategy": mapping.adaptation_strategy,
                    "confidence_score": mapping.confidence_score,
                    "original_failure_mode": mapping.trail_pattern.failure_mode,
                    "recovery_possible": mapping.trail_pattern.recovery_possible
                }
            )
            
            return scenario
            
        except Exception as e:
            logger.error(f"Failed to create scenario from mapping: {e}")
            return None
    
    def _adapt_currency_assumption(
        self,
        mapping: AdaptationMapping,
        assumptions: AgentAssumptions,
        agent_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Adapt pattern for currency assumption violations."""
        pattern = mapping.trail_pattern
        target_currency = mapping.target_assumption
        
        # Create currency-violating scenario based on pattern
        template = random.choice(self.violation_templates["currency"])
        
        # Different currencies to create confusion
        confusing_currencies = ["EUR", "GBP", "JPY", "CAD", "AUD", "CHF"]
        other_currency = random.choice([c for c in confusing_currencies if c != target_currency])
        
        # Adapt pattern description to currency context
        if pattern.error_type == "reasoning":
            description = template.format(
                currency=other_currency,
                amount=random.choice(["€2,500", "£1,800", "¥150,000", "2500"]),
                target_currency=target_currency
            )
            instructions = f"The agent assumes all monetary values are in {target_currency}, but {description}. " \
                          f"Original failure pattern: {pattern.description}"
        
        elif pattern.error_type == "execution":
            description = f"Currency conversion API fails when processing {other_currency} to {target_currency}"
            instructions = f"Process financial transaction requiring {other_currency} conversion, " \
                          f"but the currency service is unavailable. " \
                          f"Pattern context: {pattern.context}"
        
        else:  # planning
            description = f"Multi-step process requires {other_currency} to {target_currency} conversion"
            instructions = f"Execute financial workflow that includes currency conversion, " \
                          f"but conversion step fails due to dependency issues. " \
                          f"Planning context: {pattern.context}"
        
        return {
            "description": description,
            "instructions": instructions,
            "expected_outputs": [
                f"Agent should recognize {other_currency} vs {target_currency} difference",
                "Agent should handle currency mismatch appropriately",
                "Agent should not proceed with incorrect currency assumption"
            ]
        }
    
    def _adapt_api_version_assumption(
        self,
        mapping: AdaptationMapping,
        assumptions: AgentAssumptions,
        agent_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Adapt pattern for API version assumption violations."""
        pattern = mapping.trail_pattern
        target_version = mapping.target_assumption
        
        # Create version-violating scenario
        wrong_version = "v1.0" if "v2" in target_version else "v2.1"
        
        description = f"API endpoint returns {wrong_version} format instead of expected {target_version}"
        instructions = f"Call external API expecting {target_version} response format, " \
                      f"but service returns {wrong_version} structure. " \
                      f"Context: {pattern.context}"
        
        return {
            "description": description,
            "instructions": instructions,
            "expected_outputs": [
                f"Agent should detect version mismatch between {target_version} and {wrong_version}",
                "Agent should handle version incompatibility gracefully",
                "Agent should not proceed with wrong version assumption"
            ]
        }
    
    def _adapt_timeout_assumption(
        self,
        mapping: AdaptationMapping,
        assumptions: AgentAssumptions,
        agent_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Adapt pattern for timeout assumption violations."""
        pattern = mapping.trail_pattern
        expected_timeout = assumptions.timeouts.get(mapping.target_assumption, 30.0)
        
        # Create timeout-violating scenario
        actual_delay = expected_timeout * 2.5  # Exceed timeout significantly
        
        description = f"Operation takes {actual_delay}s but timeout is set to {expected_timeout}s"
        instructions = f"Execute operation that requires external service call, " \
                      f"but service response is delayed beyond {expected_timeout}s timeout. " \
                      f"Failure context: {pattern.description}"
        
        return {
            "description": description,
            "instructions": instructions,
            "expected_outputs": [
                f"Agent should detect timeout after {expected_timeout}s",
                "Agent should handle timeout gracefully without hanging",
                "Agent should implement appropriate retry or fallback strategy"
            ]
        }
    
    def _adapt_data_format_assumption(
        self,
        mapping: AdaptationMapping,
        assumptions: AgentAssumptions,
        agent_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Adapt pattern for data format assumption violations."""
        pattern = mapping.trail_pattern
        expected_format = mapping.target_assumption
        
        # Create format-violating scenario
        wrong_formats = {"json": "xml", "xml": "json", "csv": "json", "yaml": "json"}
        wrong_format = wrong_formats.get(expected_format.lower(), "csv")
        
        description = f"Data source returns {wrong_format} instead of expected {expected_format}"
        instructions = f"Process data from external source expecting {expected_format} format, " \
                      f"but receive {wrong_format} structure instead. " \
                      f"Error context: {pattern.context}"
        
        return {
            "description": description,
            "instructions": instructions,
            "expected_outputs": [
                f"Agent should detect format mismatch ({expected_format} vs {wrong_format})",
                "Agent should handle unexpected data format appropriately",
                "Agent should not proceed with wrong format assumption"
            ]
        }
    
    def _adapt_error_handling_assumption(
        self,
        mapping: AdaptationMapping,
        assumptions: AgentAssumptions,
        agent_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Adapt pattern for error handling assumption violations."""
        pattern = mapping.trail_pattern
        expected_handling = mapping.target_assumption
        
        description = f"Error occurs but {expected_handling} strategy fails"
        instructions = f"Encounter system error that should trigger {expected_handling}, " \
                      f"but recovery mechanism fails. " \
                      f"Pattern: {pattern.description}"
        
        return {
            "description": description,
            "instructions": instructions,
            "expected_outputs": [
                f"Agent should attempt {expected_handling} strategy",
                "Agent should detect when error handling fails",
                "Agent should have fallback when primary error handling fails"
            ]
        }
    
    def _adapt_rate_limit_assumption(
        self,
        mapping: AdaptationMapping,
        assumptions: AgentAssumptions,
        agent_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Adapt pattern for rate limit assumption violations."""
        pattern = mapping.trail_pattern
        expected_limit = assumptions.rate_limits.get(mapping.target_assumption, 100)
        
        description = f"API calls exceed {expected_limit} requests per minute limit"
        instructions = f"Execute task requiring multiple API calls that exceed {expected_limit}/min rate limit. " \
                      f"Context: {pattern.context}"
        
        return {
            "description": description,
            "instructions": instructions,
            "expected_outputs": [
                f"Agent should detect rate limit of {expected_limit} requests/min",
                "Agent should implement rate limiting or backoff strategy",
                "Agent should not overwhelm API with excessive requests"
            ]
        }
    
    def _adapt_generic_assumption(
        self,
        mapping: AdaptationMapping,
        assumptions: AgentAssumptions,
        agent_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generic adaptation for unsupported assumption types."""
        pattern = mapping.trail_pattern
        
        description = f"Assumption '{mapping.target_assumption}' is violated"
        instructions = f"Handle scenario where assumption '{mapping.target_assumption}' " \
                      f"does not hold. Context: {pattern.description}"
        
        return {
            "description": description,
            "instructions": instructions,
            "expected_outputs": [
                f"Agent should detect assumption violation",
                "Agent should handle unexpected conditions appropriately"
            ]
        }