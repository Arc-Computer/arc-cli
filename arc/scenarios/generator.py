"""
Enhanced Scenario Generator for Arc-Eval Production
Combines LLM-based and pattern-based generation approaches
Production version adapted from experiments/generation/enhanced_generator.py
"""

import os
import json
import yaml
import asyncio
from typing import List, Dict, Any, Optional, Union
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field
import hashlib

# Import production components
from arc.core.models.scenario import Scenario
from arc.core.utils.constants import DEFAULT_QUALITY_THRESHOLD
from arc.scenarios.quality_scorer import ScenarioQualityScorer, ScenarioDeduplicator


@dataclass
class GenerationMetrics:
    """Metrics for scenario generation"""
    total_generated: int = 0
    generation_method: str = "hybrid"
    patterns_used: List[str] = field(default_factory=list)
    quality_scores: List[float] = field(default_factory=list)
    duplicates_removed: int = 0
    scenarios_rejected: int = 0
    generation_cost: float = 0.0
    time_taken: float = 0.0
    tokens_used: int = 0
    domains_found: set = field(default_factory=set)


class ScenarioGenerator:
    """Production scenario generator with hybrid approach"""
    
    def __init__(
        self,
        agent_config_path: str,
        api_key: Optional[str] = None,
        use_patterns: bool = True,
        quality_threshold: float = DEFAULT_QUALITY_THRESHOLD
    ):
        """
        Initialize scenario generator
        
        Args:
            agent_config_path: Path to agent configuration YAML
            api_key: API key for LLM provider
            use_patterns: Whether to use pattern-based generation
            quality_threshold: Minimum quality score for scenarios
        """
        self.agent_config_path = agent_config_path
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        self.use_patterns = use_patterns
        self.quality_threshold = quality_threshold
        
        # Load agent configuration
        with open(agent_config_path, 'r') as f:
            self.agent_config = yaml.safe_load(f)
        
        # Initialize components
        if self.use_patterns:
            self.quality_scorer = ScenarioQualityScorer(min_threshold=quality_threshold)
            self.deduplicator = ScenarioDeduplicator()
        
        # Generation metrics
        self.metrics = GenerationMetrics()
        self.generated_scenarios: List[Scenario] = []
    
    async def generate_scenarios_batch(
        self,
        count: int = 100,
        batch_size: int = 20,
        pattern_ratio: float = 0.7
    ) -> List[Scenario]:
        """
        Generate scenarios using hybrid approach
        
        Args:
            count: Total number of scenarios to generate
            batch_size: Scenarios per batch
            pattern_ratio: Ratio of pattern-based vs pure LLM scenarios
        
        Returns:
            List of Scenario objects
        """
        start_time = datetime.now()
        all_scenarios = []
        
        if self.use_patterns and pattern_ratio > 0:
            # Generate pattern-based scenarios
            pattern_count = int(count * pattern_ratio)
            print(f"\nGenerating {pattern_count} pattern-based scenarios...")
            
            pattern_scenarios = await self._generate_pattern_based_scenarios(pattern_count)
            all_scenarios.extend(pattern_scenarios)
            
            # Generate remaining with LLM
            llm_count = count - len(pattern_scenarios)
            if llm_count > 0:
                print(f"\nGenerating {llm_count} additional LLM scenarios...")
                llm_scenarios = await self._generate_llm_scenarios(llm_count, batch_size)
                all_scenarios.extend(llm_scenarios)
        else:
            # Use LLM generation only
            all_scenarios = await self._generate_llm_scenarios(count, batch_size)
        
        # Apply quality filtering
        if self.use_patterns:
            all_scenarios = self._apply_quality_filtering(all_scenarios)
        
        # Update metrics
        self.metrics.total_generated = len(all_scenarios)
        self.metrics.time_taken = (datetime.now() - start_time).total_seconds()
        self.generated_scenarios = all_scenarios
        
        return all_scenarios
    
    async def _generate_pattern_based_scenarios(self, count: int) -> List[Scenario]:
        """Generate scenarios using failure patterns"""
        # TODO: Implement pattern-based generation using failure patterns
        # For now, return empty list to allow migration to complete
        return []
    
    async def _generate_llm_scenarios(self, count: int, batch_size: int) -> List[Scenario]:
        """Generate scenarios using LLM"""
        # TODO: Implement LLM-based generation
        # For now, return empty list to allow migration to complete
        return []
    
    def _apply_quality_filtering(self, scenarios: List[Scenario]) -> List[Scenario]:
        """Apply quality scoring and deduplication"""
        print(f"\nApplying quality filtering to {len(scenarios)} scenarios...")
        
        # Convert to dict format for scoring
        scenario_dicts = [s.to_dict() for s in scenarios]
        
        # Score scenarios
        passed_dicts, failed_dicts = self.quality_scorer.score_batch(scenario_dicts)
        self.metrics.scenarios_rejected = len(failed_dicts)
        
        print(f"  ✓ {len(passed_dicts)} passed quality threshold")
        print(f"  ✗ {len(failed_dicts)} rejected")
        
        # Deduplicate
        unique_dicts = self.deduplicator.deduplicate_batch(passed_dicts)
        self.metrics.duplicates_removed = len(passed_dicts) - len(unique_dicts)
        
        print(f"  • {self.metrics.duplicates_removed} duplicates removed")
        
        # Convert back to Scenario objects
        filtered_scenarios = []
        for scenario_dict in unique_dicts:
            scenario = Scenario(**scenario_dict)
            if 'quality_metrics' in scenario_dict:
                scenario.quality_score = scenario_dict['quality_metrics']['total_score']
                self.metrics.quality_scores.append(scenario.quality_score)
            filtered_scenarios.append(scenario)
        
        return filtered_scenarios
    
    def prepare_for_sandbox(self, scenarios: List[Scenario]) -> Dict[str, Any]:
        """Prepare scenarios for sandbox execution"""
        return {
            "scenarios": [s.to_dict() for s in scenarios],
            "metadata": {
                "total_scenarios": len(scenarios),
                "generation_timestamp": datetime.now().isoformat(),
                "generation_method": self.metrics.generation_method,
                "patterns_used": self.metrics.patterns_used,
                "quality_threshold": self.quality_threshold if self.use_patterns else None,
                "average_quality_score": (
                    sum(self.metrics.quality_scores) / len(self.metrics.quality_scores)
                    if self.metrics.quality_scores else None
                ),
                "duplicates_removed": self.metrics.duplicates_removed,
                "scenarios_rejected": self.metrics.scenarios_rejected
            }
        }
    
    def save_scenarios(self, scenarios: List[Scenario], output_path: str):
        """Save scenarios to file"""
        data = self.prepare_for_sandbox(scenarios)
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"Saved {len(scenarios)} scenarios to {output_path}")
    
    def get_generation_summary(self) -> Dict[str, Any]:
        """Get comprehensive generation summary"""
        summary = {
            "total_generated": self.metrics.total_generated,
            "generation_method": self.metrics.generation_method,
            "domains_found": list(self.metrics.domains_found),
            "generation_cost": f"${self.metrics.generation_cost:.4f}",
            "time_taken": f"{self.metrics.time_taken:.2f}s",
            "tokens_used": self.metrics.tokens_used
        }
        
        if self.use_patterns:
            summary.update({
                "patterns_used": len(self.metrics.patterns_used),
                "pattern_list": self.metrics.patterns_used,
                "quality_metrics": {
                    "average_score": (
                        sum(self.metrics.quality_scores) / len(self.metrics.quality_scores)
                        if self.metrics.quality_scores else 0
                    ),
                    "scenarios_rejected": self.metrics.scenarios_rejected,
                    "duplicates_removed": self.metrics.duplicates_removed
                }
            })
        
        return summary


# Convenience function for high-quality scenario generation
async def generate_high_quality_scenarios(
    agent_config_path: str,
    count: int = 100,
    use_patterns: bool = True,
    pattern_ratio: float = 0.7,
    quality_threshold: float = DEFAULT_QUALITY_THRESHOLD,
    output_path: Optional[str] = None
) -> List[Scenario]:
    """
    High-level function to generate quality scenarios
    
    Args:
        agent_config_path: Path to agent configuration
        count: Number of scenarios to generate
        use_patterns: Whether to use pattern-based generation
        pattern_ratio: Ratio of pattern-based scenarios (0-1)
        quality_threshold: Minimum quality score
        output_path: Optional path to save scenarios
    
    Returns:
        List of generated scenarios
    """
    print(f"Generating {count} high-quality scenarios")
    print(f"   Pattern-based: {use_patterns} ({pattern_ratio*100:.0f}% if enabled)")
    print(f"   Quality threshold: {quality_threshold}")
    
    generator = ScenarioGenerator(
        agent_config_path,
        use_patterns=use_patterns,
        quality_threshold=quality_threshold
    )
    
    scenarios = await generator.generate_scenarios_batch(
        count=count,
        pattern_ratio=pattern_ratio
    )
    
    if output_path:
        generator.save_scenarios(scenarios, output_path)
    
    # Print summary
    summary = generator.get_generation_summary()
    print(f"\n✓ Generation complete")
    print(f"   Generated: {summary['total_generated']} scenarios")
    print(f"   Cost: {summary['generation_cost']}")
    if use_patterns:
        print(f"   Patterns used: {summary['patterns_used']}")
        print(f"   Average quality: {summary['quality_metrics']['average_score']:.2f}")
        print(f"   Rejected: {summary['quality_metrics']['scenarios_rejected']}")
    
    return scenarios