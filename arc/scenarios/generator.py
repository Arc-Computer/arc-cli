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
from ..core.models.scenario import Scenario
from ..core.utils.constants import DEFAULT_QUALITY_THRESHOLD
from .generation_coordinator import GenerationCoordinator, GenerationMetrics as CoordinatorMetrics


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
        
        # Initialize generation coordinator
        self.coordinator = GenerationCoordinator(
            api_key=self.api_key,
            use_llm=bool(self.api_key),
            quality_threshold=quality_threshold
        )
        
        # Generation metrics
        self.metrics = GenerationMetrics()
        self.generated_scenarios: List[Scenario] = []
    
    async def generate_scenarios_batch(
        self,
        count: int = 100,
        batch_size: int = 20,
        pattern_ratio: float = 0.7,
        currency_focus: bool = False
    ) -> List[Scenario]:
        """
        Generate scenarios using hybrid approach
        
        Args:
            count: Total number of scenarios to generate
            batch_size: Scenarios per batch (not used with coordinator)
            pattern_ratio: Ratio of pattern-based vs pure LLM scenarios
            currency_focus: Whether to focus on currency-related scenarios
        
        Returns:
            List of Scenario objects
        """
        # Determine generation approach
        if currency_focus and count <= 15:
            # Generate currency-focused scenarios
            scenarios, coord_metrics = await self.coordinator.generate_currency_scenarios(
                self.agent_config,
                count=count
            )
        elif count >= 35 and not currency_focus:
            # Generate TRAIL-inspired scenarios
            scenarios, coord_metrics = await self.coordinator.generate_trail_scenarios(
                self.agent_config,
                count=count
            )
        else:
            # General scenario generation
            scenarios, coord_metrics = await self.coordinator.generate_scenarios(
                self.agent_config,
                total_scenarios=count,
                focus_on_assumptions=True,
                currency_focus=currency_focus
            )
        
        # Update our metrics from coordinator metrics
        self.metrics.total_generated = coord_metrics.total_scenarios_generated
        self.metrics.time_taken = coord_metrics.generation_time_seconds
        self.metrics.patterns_used = list(set(coord_metrics.patterns_selected))
        self.metrics.duplicates_removed = coord_metrics.duplicates_removed
        self.metrics.scenarios_rejected = coord_metrics.scenarios_failed_quality
        self.metrics.generation_cost = coord_metrics.estimated_cost
        
        # Extract quality scores
        for scenario in scenarios:
            if 'quality_metrics' in scenario.metadata:
                score = scenario.metadata['quality_metrics'].get('total', 0)
                self.metrics.quality_scores.append(score)
        
        self.generated_scenarios = scenarios
        return scenarios
    
    
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