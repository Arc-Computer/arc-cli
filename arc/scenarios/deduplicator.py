"""
Scenario deduplication to ensure uniqueness and diversity.
"""

import hashlib
from typing import List, Dict, Any, Set, Union
from dataclasses import dataclass

from ..core.models.scenario import Scenario


@dataclass
class DeduplicationStats:
    """Statistics from deduplication process."""
    total_input: int
    unique_scenarios: int
    duplicates_removed: int
    duplicate_rate: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_input": self.total_input,
            "unique_scenarios": self.unique_scenarios,
            "duplicates_removed": self.duplicates_removed,
            "duplicate_rate": round(self.duplicate_rate, 3)
        }


class ScenarioDeduplicator:
    """Handles scenario deduplication using content hashing."""
    
    def __init__(self):
        """Initialize deduplicator."""
        self.seen_hashes: Dict[str, str] = {}  # hash -> scenario_id
        self.duplicate_count = 0
    
    def get_scenario_hash(self, scenario: Union[Scenario, Dict[str, Any]]) -> str:
        """Generate hash for scenario based on task and tools.
        
        Args:
            scenario: Scenario object or dictionary
            
        Returns:
            SHA256 hash (truncated to 16 chars)
        """
        if isinstance(scenario, Scenario):
            task = scenario.task_prompt
            tools = sorted(scenario.expected_tools)  # Sort for consistency
        else:
            task = scenario.get('task_prompt', '') or scenario.get('task', '')
            tools = sorted(scenario.get('expected_tools', []) or scenario.get('tools', []))
        
        # Create content string
        tools_str = ':'.join(tools)
        content = f"{task}:{tools_str}"
        
        # Generate hash
        return hashlib.sha256(content.encode()).hexdigest()[:16]
    
    def is_duplicate(self, scenario: Union[Scenario, Dict[str, Any]]) -> bool:
        """Check if scenario is a duplicate.
        
        Args:
            scenario: Scenario to check
            
        Returns:
            True if duplicate, False otherwise
        """
        scenario_hash = self.get_scenario_hash(scenario)
        
        if scenario_hash in self.seen_hashes:
            self.duplicate_count += 1
            return True
        
        # Get scenario ID
        if isinstance(scenario, Scenario):
            scenario_id = scenario.id or scenario_hash
        else:
            scenario_id = scenario.get('id', scenario_hash)
        
        self.seen_hashes[scenario_hash] = scenario_id
        return False
    
    def deduplicate_batch(
        self,
        scenarios: List[Union[Scenario, Dict[str, Any]]]
    ) -> List[Union[Scenario, Dict[str, Any]]]:
        """Remove duplicates from a batch of scenarios.
        
        Args:
            scenarios: List of scenarios
            
        Returns:
            List of unique scenarios
        """
        unique_scenarios = []
        
        for scenario in scenarios:
            if not self.is_duplicate(scenario):
                # Add hash to scenario metadata
                scenario_hash = self.get_scenario_hash(scenario)
                
                if isinstance(scenario, Scenario):
                    scenario.metadata['scenario_hash'] = scenario_hash
                else:
                    scenario['scenario_hash'] = scenario_hash
                
                unique_scenarios.append(scenario)
        
        return unique_scenarios
    
    def get_stats(self) -> DeduplicationStats:
        """Get deduplication statistics.
        
        Returns:
            Deduplication statistics
        """
        total = len(self.seen_hashes) + self.duplicate_count
        
        return DeduplicationStats(
            total_input=total,
            unique_scenarios=len(self.seen_hashes),
            duplicates_removed=self.duplicate_count,
            duplicate_rate=self.duplicate_count / total if total > 0 else 0.0
        )
    
    def reset(self) -> None:
        """Reset deduplicator state."""
        self.seen_hashes.clear()
        self.duplicate_count = 0
    
    def merge_similar_scenarios(
        self,
        scenarios: List[Union[Scenario, Dict[str, Any]]],
        similarity_threshold: float = 0.85
    ) -> List[Union[Scenario, Dict[str, Any]]]:
        """Merge scenarios that are too similar (advanced deduplication).
        
        This is a more aggressive deduplication that considers semantic similarity.
        Currently just does exact deduplication - similarity checking could be
        added with embedding models or fuzzy matching.
        
        Args:
            scenarios: List of scenarios
            similarity_threshold: Threshold for considering scenarios similar
            
        Returns:
            List of unique scenarios
        """
        # For now, just do exact deduplication
        # Future: Add semantic similarity with embeddings
        return self.deduplicate_batch(scenarios)