"""
Coordinates parallel scenario generation using the two-stage pipeline.
Manages pattern selection, instantiation, and quality control.
"""

import asyncio
import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass, field
import os

logger = logging.getLogger(__name__)

from ..core.models.scenario import Scenario
from .failure_patterns import PatternLibrary
from .assumption_extractor import AssumptionExtractor, AgentAssumptions
from .pattern_selector import PatternSelector
from .scenario_instantiator import ScenarioInstantiator
from .quality_scorer import ScenarioQualityScorer
from .deduplicator import ScenarioDeduplicator
from .trail_loader import TrailDatasetLoader
from .trail_adapter import TrailPatternAdapter
from .pattern_library import EnhancedPatternLibrary
from .quality_validator import TrailQualityValidator


@dataclass
class GenerationMetrics:
    """Enhanced metrics for scenario generation with TRAIL integration."""
    total_scenarios_requested: int
    total_scenarios_generated: int = 0
    patterns_selected: List[str] = field(default_factory=list)
    scenarios_passed_quality: int = 0
    scenarios_failed_quality: int = 0
    duplicates_removed: int = 0
    generation_time_seconds: float = 0.0
    pattern_selection_time: float = 0.0
    instantiation_time: float = 0.0
    quality_scoring_time: float = 0.0
    estimated_cost: float = 0.0
    # TRAIL-specific metrics
    trail_patterns_used: int = 0
    currency_scenarios_generated: int = 0
    trail_scenarios_generated: int = 0
    adaptation_metrics: Dict[str, Any] = field(default_factory=dict)
    assumption_coverage: Dict[str, int] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_requested": self.total_scenarios_requested,
            "total_generated": self.total_scenarios_generated,
            "patterns_used": len(set(self.patterns_selected)),
            "quality_passed": self.scenarios_passed_quality,
            "quality_failed": self.scenarios_failed_quality,
            "duplicates_removed": self.duplicates_removed,
            "generation_time": round(self.generation_time_seconds, 2),
            "breakdown": {
                "pattern_selection": round(self.pattern_selection_time, 2),
                "instantiation": round(self.instantiation_time, 2),
                "quality_scoring": round(self.quality_scoring_time, 2)
            },
            "estimated_cost": round(self.estimated_cost, 4),
            "efficiency": {
                "success_rate": round(self.scenarios_passed_quality / max(self.total_scenarios_generated, 1), 2),
                "scenarios_per_second": round(self.total_scenarios_generated / max(self.generation_time_seconds, 1), 2)
            },
            # TRAIL-specific metrics
            "trail_integration": {
                "trail_patterns_used": self.trail_patterns_used,
                "currency_scenarios": self.currency_scenarios_generated,
                "trail_scenarios": self.trail_scenarios_generated,
                "adaptation_metrics": self.adaptation_metrics,
                "assumption_coverage": self.assumption_coverage
            }
        }


class GenerationCoordinator:
    """Enhanced coordination for scenario generation with TRAIL integration."""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        use_llm: bool = True,
        quality_threshold: float = 0.7,
        patterns_per_batch: int = 3,
        scenarios_per_pattern: int = 7,
        enable_trail: bool = True,
        random_seed: Optional[int] = None
    ):
        """Initialize the enhanced generation coordinator.
        
        Args:
            api_key: OpenRouter API key
            use_llm: Whether to use LLM for generation
            quality_threshold: Minimum quality score (0.0-1.0)
            patterns_per_batch: Number of patterns per batch
            scenarios_per_pattern: Scenarios to generate per pattern
            enable_trail: Whether to enable TRAIL integration
            random_seed: Seed for reproducible generation
        """
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        self.use_llm = use_llm and bool(self.api_key)
        self.quality_threshold = quality_threshold
        self.patterns_per_batch = patterns_per_batch
        self.scenarios_per_pattern = scenarios_per_pattern
        self.enable_trail = enable_trail
        
        # Initialize enhanced components
        if enable_trail:
            self.pattern_library = EnhancedPatternLibrary()
            self.trail_loader = TrailDatasetLoader()
            self.trail_adapter = TrailPatternAdapter(self.trail_loader, random_seed)
            self.quality_validator = None  # Will be initialized after TRAIL loading
        else:
            # Fallback to basic components
            self.pattern_library = PatternLibrary()
            self.trail_loader = None
            self.trail_adapter = None
            self.quality_validator = None
        
        self.assumption_extractor = AssumptionExtractor()
        self.pattern_selector = PatternSelector(
            self.pattern_library,
            api_key=self.api_key,
            use_llm=self.use_llm
        )
        self.scenario_instantiator = ScenarioInstantiator(
            api_key=self.api_key,
            use_llm=self.use_llm,
            quality_threshold=self.quality_threshold
        )
        self.quality_scorer = ScenarioQualityScorer(min_threshold=self.quality_threshold)
        self.deduplicator = ScenarioDeduplicator()
        
        # Track initialization state
        self._trail_initialized = False
        
        # Cost estimation (rough estimates per call)
        self.cost_estimates = {
            "pattern_selection_llm": 0.0002,  # GPT-4.1-mini
            "scenario_instantiation_llm": 0.001,  # GPT-4.1
            "pattern_selection_heuristic": 0.0,
            "scenario_instantiation_adapter": 0.0
        }
    
    async def _ensure_trail_initialized(self) -> None:
        """Ensure TRAIL components are initialized."""
        if not self.enable_trail or self._trail_initialized:
            return
        
        try:
            # Initialize enhanced pattern library with TRAIL data
            await self.pattern_library.initialize()
            
            # Initialize quality validator with TRAIL patterns
            trail_patterns = self.trail_loader.get_all_patterns()
            self.quality_validator = TrailQualityValidator(
                quality_threshold=self.quality_threshold,
                trail_patterns=trail_patterns
            )
            
            self._trail_initialized = True
            logger.info(f"TRAIL integration initialized with {len(trail_patterns)} patterns")
            
        except Exception as e:
            logger.warning(f"Failed to initialize TRAIL components: {e}")
            # Graceful fallback to basic components
            self.enable_trail = False
            if hasattr(self, 'pattern_library') and hasattr(self.pattern_library, '__class__'):
                if 'Enhanced' in self.pattern_library.__class__.__name__:
                    self.pattern_library = PatternLibrary()
    
    async def generate_scenarios(
        self,
        agent_config: Dict[str, Any],
        total_scenarios: int = 50,
        focus_on_assumptions: bool = True,
        currency_focus: bool = False
    ) -> Tuple[List[Scenario], GenerationMetrics]:
        """Generate scenarios using enhanced pipeline with TRAIL integration.
        
        Args:
            agent_config: Agent configuration
            total_scenarios: Total number of scenarios to generate
            focus_on_assumptions: Whether to focus on assumption violations
            currency_focus: Whether to focus on currency-related scenarios
            
        Returns:
            Tuple of (scenarios, metrics)
        """
        start_time = datetime.now()
        metrics = GenerationMetrics(total_scenarios_requested=total_scenarios)
        
        # Ensure TRAIL components are initialized
        await self._ensure_trail_initialized()
        
        # Extract assumptions
        assumptions = self.assumption_extractor.extract(agent_config)
        
        # Prepare batches
        scenarios_per_batch = self.patterns_per_batch * self.scenarios_per_pattern
        num_batches = (total_scenarios + scenarios_per_batch - 1) // scenarios_per_batch
        
        all_scenarios = []
        seen_hashes = set()
        
        # Process batches
        for batch_num in range(num_batches):
            if len(all_scenarios) >= total_scenarios:
                break
            
            # Stage A: Pattern Selection
            selection_start = datetime.now()
            
            if currency_focus and batch_num == 0:
                # First batch focuses on currency patterns
                currency_patterns = [
                    p for p in self.pattern_library.get_patterns_by_category("calculation")
                    if "precision" in p.id or "currency" in p.description.lower()
                ]
                selection_result = await self.pattern_selector.select_patterns(
                    assumptions,
                    agent_config,
                    count=min(self.patterns_per_batch, len(currency_patterns)),
                    prioritize_assumptions=True
                )
            else:
                # Regular pattern selection
                selection_result = await self.pattern_selector.select_patterns(
                    assumptions,
                    agent_config,
                    count=self.patterns_per_batch,
                    prioritize_assumptions=focus_on_assumptions
                )
            
            metrics.pattern_selection_time += (datetime.now() - selection_start).total_seconds()
            metrics.patterns_selected.extend([p.id for p in selection_result.selected_patterns])
            
            # Update cost estimate
            if selection_result.selection_method == "llm":
                metrics.estimated_cost += self.cost_estimates["pattern_selection_llm"]
            
            # Stage B: Scenario Instantiation
            instantiation_start = datetime.now()
            
            instantiation_result = await self.scenario_instantiator.instantiate_scenarios(
                selection_result.selected_patterns,
                assumptions,
                agent_config,
                scenarios_per_pattern=self.scenarios_per_pattern,
                use_hybrid=True
            )
            
            metrics.instantiation_time += (datetime.now() - instantiation_start).total_seconds()
            
            # Update cost estimate
            if instantiation_result.generation_method in ["llm", "hybrid"]:
                metrics.estimated_cost += (
                    self.cost_estimates["scenario_instantiation_llm"] * 
                    len(selection_result.selected_patterns) * 
                    (self.scenarios_per_pattern // 2 if instantiation_result.generation_method == "hybrid" else self.scenarios_per_pattern)
                )
            
            # Quality scoring and deduplication
            scoring_start = datetime.now()
            
            # Quality score scenarios
            passed_scenarios = []
            for scenario in instantiation_result.scenarios:
                quality_metrics = self.quality_scorer.score_scenario(scenario)
                
                # Add quality metrics to scenario metadata
                scenario.metadata["quality_metrics"] = quality_metrics.to_dict()
                
                if quality_metrics.passed_threshold:
                    # Check for duplicates
                    scenario_hash = self.deduplicator.get_scenario_hash(scenario)
                    if scenario_hash not in seen_hashes:
                        seen_hashes.add(scenario_hash)
                        passed_scenarios.append(scenario)
                        metrics.scenarios_passed_quality += 1
                    else:
                        metrics.duplicates_removed += 1
                else:
                    metrics.scenarios_failed_quality += 1
            
            metrics.quality_scoring_time += (datetime.now() - scoring_start).total_seconds()
            
            # Add passed scenarios to results
            all_scenarios.extend(passed_scenarios)
            
            # Stop if we have enough
            if len(all_scenarios) >= total_scenarios:
                all_scenarios = all_scenarios[:total_scenarios]
                break
        
        # Final metrics
        metrics.total_scenarios_generated = len(all_scenarios)
        metrics.generation_time_seconds = (datetime.now() - start_time).total_seconds()
        
        return all_scenarios, metrics
    
    async def generate_currency_scenarios(
        self,
        agent_config: Dict[str, Any],
        count: int = 15
    ) -> Tuple[List[Scenario], GenerationMetrics]:
        """Generate currency assumption violation scenarios.
        
        Args:
            agent_config: Agent configuration
            count: Number of currency scenarios
            
        Returns:
            Tuple of (scenarios, metrics)
        """
        # Temporarily adjust parameters for currency focus
        original_patterns = self.patterns_per_batch
        original_scenarios = self.scenarios_per_pattern
        
        self.patterns_per_batch = 2  # Focus on fewer patterns
        self.scenarios_per_pattern = 8  # More variations per pattern
        
        try:
            scenarios, metrics = await self.generate_scenarios(
                agent_config,
                total_scenarios=count,
                focus_on_assumptions=True,
                currency_focus=True
            )
            
            # Tag all scenarios as currency-focused
            for scenario in scenarios:
                if "currency" not in scenario.tags:
                    scenario.tags.append("currency")
                scenario.tags.append("assumption_violation")
            
            # Update TRAIL-specific metrics
            metrics.currency_scenarios_generated = len(scenarios)
            
            return scenarios, metrics
            
        finally:
            # Restore original parameters
            self.patterns_per_batch = original_patterns
            self.scenarios_per_pattern = original_scenarios
    
    async def generate_trail_scenarios(
        self,
        agent_config: Dict[str, Any],
        count: int = 35
    ) -> Tuple[List[Scenario], GenerationMetrics]:
        """Generate TRAIL-based general capability test scenarios.
        
        Args:
            agent_config: Agent configuration
            count: Number of TRAIL scenarios
            
        Returns:
            Tuple of (scenarios, metrics)
        """
        start_time = datetime.now()
        metrics = GenerationMetrics(total_scenarios_requested=count)
        
        # Ensure TRAIL components are initialized
        await self._ensure_trail_initialized()
        
        if not self.enable_trail or not self.trail_adapter:
            # Fallback to original implementation
            logger.warning("TRAIL not available, falling back to basic generation")
            return await self._generate_trail_fallback(agent_config, count)
        
        try:
            # Extract assumptions for adaptation
            assumptions = self.assumption_extractor.extract(agent_config)
            
            # Use TRAIL adapter to create assumption violation scenarios
            adaptation_result = await self.trail_adapter.adapt_patterns_to_assumptions(
                assumptions=assumptions,
                agent_config=agent_config,
                target_count=count,
                focus_types=["reasoning", "execution", "planning"]
            )
            
            scenarios = adaptation_result.scenarios
            
            # Update metrics with TRAIL-specific data
            metrics.trail_scenarios_generated = len(scenarios)
            metrics.trail_patterns_used = len(adaptation_result.adaptations_used)
            metrics.adaptation_metrics = adaptation_result.adaptation_metrics
            
            # Track assumption coverage
            for adaptation in adaptation_result.adaptations_used:
                assumption_type = adaptation.assumption_type
                metrics.assumption_coverage[assumption_type] = (
                    metrics.assumption_coverage.get(assumption_type, 0) + 1
                )
            
            # Apply quality validation if available
            if self.quality_validator:
                validated_scenarios = []
                for scenario in scenarios:
                    quality_result = self.quality_validator.validate_scenario(scenario)
                    
                    # Add quality metrics to scenario metadata
                    scenario.metadata["quality_result"] = quality_result.to_dict()
                    
                    if quality_result.passed_threshold:
                        validated_scenarios.append(scenario)
                        metrics.scenarios_passed_quality += 1
                    else:
                        metrics.scenarios_failed_quality += 1
                
                scenarios = validated_scenarios
            else:
                # Use basic quality check
                for scenario in scenarios:
                    if len(scenario.description) > 20 and len(scenario.instructions) > 30:
                        metrics.scenarios_passed_quality += 1
                    else:
                        metrics.scenarios_failed_quality += 1
            
            # Deduplication
            seen_hashes = set()
            deduplicated_scenarios = []
            for scenario in scenarios:
                scenario_hash = self.deduplicator.get_scenario_hash(scenario)
                if scenario_hash not in seen_hashes:
                    seen_hashes.add(scenario_hash)
                    deduplicated_scenarios.append(scenario)
                else:
                    metrics.duplicates_removed += 1
            
            scenarios = deduplicated_scenarios[:count]  # Limit to requested count
            
            # Final metrics
            metrics.total_scenarios_generated = len(scenarios)
            metrics.generation_time_seconds = (datetime.now() - start_time).total_seconds()
            
            logger.info(f"Generated {len(scenarios)} TRAIL-based scenarios using {metrics.trail_patterns_used} patterns")
            
            return scenarios, metrics
            
        except Exception as e:
            logger.error(f"TRAIL scenario generation failed: {e}")
            # Fallback to basic generation
            return await self._generate_trail_fallback(agent_config, count)
    
    async def _generate_trail_fallback(
        self,
        agent_config: Dict[str, Any],
        count: int
    ) -> Tuple[List[Scenario], GenerationMetrics]:
        """Fallback implementation when TRAIL is not available."""
        # Use original implementation as fallback
        original_patterns = self.patterns_per_batch
        self.patterns_per_batch = 5
        
        try:
            scenarios, metrics = await self.generate_scenarios(
                agent_config,
                total_scenarios=count,
                focus_on_assumptions=False,
                currency_focus=False
            )
            
            # Tag as fallback scenarios
            for scenario in scenarios:
                scenario.tags.append("trail_fallback")
                scenario.tags.append("general_capability")
            
            return scenarios, metrics
            
        finally:
            self.patterns_per_batch = original_patterns
    
    def estimate_generation_time(self, total_scenarios: int) -> Dict[str, float]:
        """Estimate generation time for given number of scenarios.
        
        Args:
            total_scenarios: Number of scenarios to generate
            
        Returns:
            Time estimates in seconds
        """
        # Based on empirical measurements
        time_per_pattern_selection = 0.5 if self.use_llm else 0.01
        time_per_scenario_llm = 0.3
        time_per_scenario_adapter = 0.01
        time_per_quality_check = 0.005
        
        num_batches = (total_scenarios + (self.patterns_per_batch * self.scenarios_per_pattern) - 1) // (self.patterns_per_batch * self.scenarios_per_pattern)
        
        pattern_selection_time = num_batches * time_per_pattern_selection
        
        if self.use_llm:
            # Hybrid mode: half LLM, half adapter
            instantiation_time = total_scenarios * (time_per_scenario_llm / 2 + time_per_scenario_adapter / 2)
        else:
            instantiation_time = total_scenarios * time_per_scenario_adapter
        
        quality_time = total_scenarios * time_per_quality_check
        
        return {
            "pattern_selection": pattern_selection_time,
            "instantiation": instantiation_time,
            "quality_scoring": quality_time,
            "total": pattern_selection_time + instantiation_time + quality_time,
            "parallel_speedup": 0.7  # Estimated parallel efficiency
        }
    
    def get_generation_stats(self) -> Dict[str, Any]:
        """Get statistics about generation capabilities."""
        return {
            "pattern_library": self.pattern_library.get_pattern_stats(),
            "pattern_selector": self.pattern_selector.get_selection_stats(),
            "quality_scorer": {
                "threshold": self.quality_threshold,
                "dimensions": ["specific_error", "edge_cases", "multi_tool", "novelty"]
            },
            "cost_estimates": self.cost_estimates,
            "capabilities": {
                "llm_enabled": self.use_llm,
                "hybrid_generation": True,
                "parallel_batching": True,
                "quality_filtering": True,
                "deduplication": True
            }
        }