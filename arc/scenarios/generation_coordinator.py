"""
Coordinates parallel scenario generation using the two-stage pipeline.
Manages pattern selection, instantiation, and quality control.
"""

import asyncio
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass, field
import os

from ..core.models.scenario import Scenario
from .failure_patterns import PatternLibrary
from .assumption_extractor import AssumptionExtractor, AgentAssumptions
from .pattern_selector import PatternSelector
from .scenario_instantiator import ScenarioInstantiator
from .quality_scorer import ScenarioQualityScorer
from .deduplicator import ScenarioDeduplicator


@dataclass
class GenerationMetrics:
    """Metrics for scenario generation."""
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
            }
        }


class GenerationCoordinator:
    """Coordinates the two-stage scenario generation pipeline."""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        use_llm: bool = True,
        quality_threshold: float = 3.0,
        patterns_per_batch: int = 3,
        scenarios_per_pattern: int = 7
    ):
        """Initialize the generation coordinator.
        
        Args:
            api_key: OpenRouter API key
            use_llm: Whether to use LLM for generation
            quality_threshold: Minimum quality score
            patterns_per_batch: Number of patterns per batch
            scenarios_per_pattern: Scenarios to generate per pattern
        """
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        self.use_llm = use_llm and bool(self.api_key)
        self.quality_threshold = quality_threshold
        self.patterns_per_batch = patterns_per_batch
        self.scenarios_per_pattern = scenarios_per_pattern
        
        # Initialize components
        self.pattern_library = PatternLibrary()
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
        
        # Cost estimation (rough estimates per call)
        self.cost_estimates = {
            "pattern_selection_llm": 0.0002,  # GPT-4.1-mini
            "scenario_instantiation_llm": 0.001,  # GPT-4.1
            "pattern_selection_heuristic": 0.0,
            "scenario_instantiation_adapter": 0.0
        }
    
    async def generate_scenarios(
        self,
        agent_config: Dict[str, Any],
        total_scenarios: int = 50,
        focus_on_assumptions: bool = True,
        currency_focus: bool = False
    ) -> Tuple[List[Scenario], GenerationMetrics]:
        """Generate scenarios using the two-stage pipeline.
        
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
        """Generate TRAIL-inspired general capability test scenarios.
        
        Args:
            agent_config: Agent configuration
            count: Number of TRAIL scenarios
            
        Returns:
            Tuple of (scenarios, metrics)
        """
        # Ensure diverse coverage across TRAIL categories
        original_patterns = self.patterns_per_batch
        
        self.patterns_per_batch = 5  # More patterns for diversity
        
        try:
            scenarios, metrics = await self.generate_scenarios(
                agent_config,
                total_scenarios=count,
                focus_on_assumptions=False,  # General capability testing
                currency_focus=False
            )
            
            # Tag all scenarios as TRAIL-inspired
            for scenario in scenarios:
                scenario.tags.append("trail_inspired")
                
                # Add TRAIL category tags based on pattern
                pattern_category = scenario.metadata.get("pattern_category", "")
                if pattern_category in ["data", "logic", "calculation"]:
                    scenario.tags.append("reasoning_error")
                elif pattern_category in ["infrastructure", "auth", "security"]:
                    scenario.tags.append("execution_error")
                elif pattern_category in ["navigation", "temporal", "multi_agent"]:
                    scenario.tags.append("planning_error")
            
            return scenarios, metrics
            
        finally:
            # Restore original parameters
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