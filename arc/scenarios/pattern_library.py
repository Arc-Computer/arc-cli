"""
Enhanced Pattern Library with TRAIL dataset integration.

Combines local failure patterns with TRAIL dataset patterns for comprehensive
scenario generation with systematic violation scenarios.
"""

import asyncio
import json
import logging
from typing import Dict, List, Any, Optional, Set, Union
from dataclasses import dataclass, field
from pathlib import Path
from datetime import datetime

from .trail_loader import TrailDatasetLoader, FailurePattern as TrailPattern
from .failure_patterns.library import PatternLibrary as LocalPatternLibrary
from ..core.models.scenario import Scenario

logger = logging.getLogger(__name__)


@dataclass
class PatternMetrics:
    """Metrics for pattern usage and effectiveness."""
    
    total_patterns: int = 0
    local_patterns: int = 0
    trail_patterns: int = 0
    patterns_by_category: Dict[str, int] = field(default_factory=dict)
    patterns_by_domain: Dict[str, int] = field(default_factory=dict)
    usage_frequency: Dict[str, int] = field(default_factory=dict)
    effectiveness_scores: Dict[str, float] = field(default_factory=dict)
    last_updated: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_patterns": self.total_patterns,
            "local_patterns": self.local_patterns,
            "trail_patterns": self.trail_patterns,
            "patterns_by_category": self.patterns_by_category,
            "patterns_by_domain": self.patterns_by_domain,
            "usage_frequency": self.usage_frequency,
            "effectiveness_scores": self.effectiveness_scores,
            "last_updated": self.last_updated
        }


@dataclass
class PatternSelectionResult:
    """Result of pattern selection for scenario generation."""
    
    selected_patterns: List[Union[dict, TrailPattern]]
    selection_method: str
    selection_criteria: Dict[str, Any]
    coverage_metrics: Dict[str, Any]
    estimated_diversity: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "pattern_count": len(self.selected_patterns),
            "selection_method": self.selection_method,
            "selection_criteria": self.selection_criteria,
            "coverage_metrics": self.coverage_metrics,
            "estimated_diversity": round(self.estimated_diversity, 3)
        }


class EnhancedPatternLibrary:
    """Enhanced pattern library combining local and TRAIL patterns."""
    
    def __init__(self, cache_dir: Optional[str] = None, version: str = "1.0"):
        """Initialize the enhanced pattern library.
        
        Args:
            cache_dir: Directory for caching patterns
            version: Pattern library version for versioning
        """
        self.version = version
        self.cache_dir = Path(cache_dir or Path.home() / ".arc_cache")
        
        # Initialize component libraries
        self.local_library = LocalPatternLibrary()
        self.trail_loader = TrailDatasetLoader(cache_dir=cache_dir)
        
        # Combined pattern storage
        self._all_patterns: List[Union[dict, TrailPattern]] = []
        self._patterns_by_category: Dict[str, List] = {}
        self._patterns_by_domain: Dict[str, List] = {}
        self._patterns_by_type: Dict[str, List] = {}
        
        # Library state
        self._loaded = False
        self._metrics = PatternMetrics()
        
        # Pattern versioning for reproducibility
        self._pattern_version_hash: Optional[str] = None
        
        # Selection optimization
        self._selection_cache: Dict[str, List] = {}
        self._effectiveness_weights = {
            "assumption_violation": 1.0,
            "real_world_pattern": 0.9,
            "domain_relevance": 0.8,
            "complexity_appropriate": 0.7,
            "recovery_possible": 0.6
        }
    
    async def initialize(self, force_refresh: bool = False) -> None:
        """Initialize the pattern library with all sources.
        
        Args:
            force_refresh: Force refresh of cached patterns
        """
        if self._loaded and not force_refresh:
            return
        
        logger.info("Initializing enhanced pattern library...")
        start_time = datetime.now()
        
        # Load local patterns
        local_patterns = self.local_library.get_all_patterns()
        logger.info(f"Loaded {len(local_patterns)} local patterns")
        
        # Load TRAIL patterns
        trail_patterns = await self.trail_loader.load_patterns(force_refresh)
        logger.info(f"Loaded {len(trail_patterns)} TRAIL patterns")
        
        # Combine patterns
        self._all_patterns = local_patterns + trail_patterns
        
        # Build indices
        self._build_pattern_indices()
        
        # Update metrics
        self._update_metrics()
        
        # Generate version hash for reproducibility
        self._generate_version_hash()
        
        self._loaded = True
        
        load_time = (datetime.now() - start_time).total_seconds()
        logger.info(f"Pattern library initialized in {load_time:.2f}s with {len(self._all_patterns)} total patterns")
    
    def _build_pattern_indices(self) -> None:
        """Build indices for fast pattern lookup."""
        self._patterns_by_category = {}
        self._patterns_by_domain = {}
        self._patterns_by_type = {}
        
        for pattern in self._all_patterns:
            # Handle both local patterns (dict) and TRAIL patterns (TrailPattern)
            if isinstance(pattern, dict):
                # Local pattern
                category = pattern.get("category", "unknown")
                domain = pattern.get("domain", "general")
                pattern_type = "local"
            else:
                # TRAIL pattern
                category = pattern.error_type
                domain = pattern.domain
                pattern_type = "trail"
            
            # Index by category
            if category not in self._patterns_by_category:
                self._patterns_by_category[category] = []
            self._patterns_by_category[category].append(pattern)
            
            # Index by domain
            if domain not in self._patterns_by_domain:
                self._patterns_by_domain[domain] = []
            self._patterns_by_domain[domain].append(pattern)
            
            # Index by type (local vs TRAIL)
            if pattern_type not in self._patterns_by_type:
                self._patterns_by_type[pattern_type] = []
            self._patterns_by_type[pattern_type].append(pattern)
    
    def _update_metrics(self) -> None:
        """Update pattern library metrics."""
        self._metrics.total_patterns = len(self._all_patterns)
        self._metrics.local_patterns = len(self._patterns_by_type.get("local", []))
        self._metrics.trail_patterns = len(self._patterns_by_type.get("trail", []))
        
        # Count by category
        for category, patterns in self._patterns_by_category.items():
            self._metrics.patterns_by_category[category] = len(patterns)
        
        # Count by domain
        for domain, patterns in self._patterns_by_domain.items():
            self._metrics.patterns_by_domain[domain] = len(patterns)
        
        self._metrics.last_updated = datetime.now().isoformat()
    
    def _generate_version_hash(self) -> None:
        """Generate version hash for reproducible pattern selection."""
        import hashlib
        
        # Create hash from pattern IDs and version
        pattern_ids = []
        for pattern in self._all_patterns:
            if isinstance(pattern, dict):
                pattern_ids.append(pattern.get("id", "unknown"))
            else:
                pattern_ids.append(pattern.id)
        
        hash_content = f"{self.version}:{sorted(pattern_ids)}"
        self._pattern_version_hash = hashlib.md5(hash_content.encode()).hexdigest()[:8]
    
    async def select_patterns_for_assumptions(
        self,
        assumption_types: List[str],
        domain_preference: Optional[str] = None,
        count: int = 10,
        diversity_weight: float = 0.5
    ) -> PatternSelectionResult:
        """Select patterns optimized for testing specific assumptions.
        
        Args:
            assumption_types: Types of assumptions to test (currency, api_version, etc.)
            domain_preference: Preferred domain (finance, web, etc.)
            count: Number of patterns to select
            diversity_weight: Weight for diversity vs relevance (0-1)
            
        Returns:
            Pattern selection result
        """
        if not self._loaded:
            await self.initialize()
        
        # Create selection cache key
        cache_key = f"{assumption_types}:{domain_preference}:{count}:{diversity_weight}"
        if cache_key in self._selection_cache:
            cached_patterns = self._selection_cache[cache_key]
            return PatternSelectionResult(
                selected_patterns=cached_patterns,
                selection_method="cached",
                selection_criteria={"cache_key": cache_key},
                coverage_metrics={},
                estimated_diversity=0.0
            )
        
        # Score all patterns for assumption testing
        scored_patterns = []
        for pattern in self._all_patterns:
            score = self._score_pattern_for_assumptions(
                pattern, assumption_types, domain_preference
            )
            if score > 0:
                scored_patterns.append((pattern, score))
        
        # Sort by score
        scored_patterns.sort(key=lambda x: x[1], reverse=True)
        
        # Select patterns with diversity consideration
        selected_patterns = self._select_diverse_patterns(
            scored_patterns, count, diversity_weight
        )
        
        # Cache selection
        self._selection_cache[cache_key] = selected_patterns
        
        # Calculate coverage metrics
        coverage_metrics = self._calculate_coverage_metrics(
            selected_patterns, assumption_types
        )
        
        # Estimate diversity
        estimated_diversity = self._estimate_pattern_diversity(selected_patterns)
        
        return PatternSelectionResult(
            selected_patterns=selected_patterns,
            selection_method="optimized",
            selection_criteria={
                "assumption_types": assumption_types,
                "domain_preference": domain_preference,
                "diversity_weight": diversity_weight
            },
            coverage_metrics=coverage_metrics,
            estimated_diversity=estimated_diversity
        )
    
    def _score_pattern_for_assumptions(
        self,
        pattern: Union[dict, TrailPattern],
        assumption_types: List[str],
        domain_preference: Optional[str]
    ) -> float:
        """Score pattern for assumption testing effectiveness."""
        score = 0.0
        
        if isinstance(pattern, dict):
            # Local pattern scoring
            category = pattern.get("category", "")
            domain = pattern.get("domain", "general")
            description = pattern.get("description", "")
        else:
            # TRAIL pattern scoring
            category = pattern.error_type
            domain = pattern.domain
            description = pattern.description
        
        # Base score for assumption type relevance
        assumption_relevance_map = {
            "currency": ["calculation", "reasoning", "finance"],
            "api_version": ["execution", "infrastructure", "api"],
            "timeout": ["execution", "infrastructure"],
            "data_format": ["data", "reasoning"],
            "error_handling": ["execution", "planning"],
            "rate_limit": ["execution", "infrastructure"]
        }
        
        for assumption_type in assumption_types:
            relevant_categories = assumption_relevance_map.get(assumption_type, [])
            if any(cat in category.lower() for cat in relevant_categories):
                score += self._effectiveness_weights["assumption_violation"]
        
        # Boost for TRAIL patterns (real-world data)
        if isinstance(pattern, TrailPattern):
            score += self._effectiveness_weights["real_world_pattern"]
        
        # Domain preference boost
        if domain_preference and domain == domain_preference:
            score += self._effectiveness_weights["domain_relevance"]
        
        # Complexity consideration
        if isinstance(pattern, TrailPattern):
            if pattern.complexity == "medium":
                score += self._effectiveness_weights["complexity_appropriate"]
        
        # Recovery possibility boost
        if isinstance(pattern, TrailPattern) and pattern.recovery_possible:
            score += self._effectiveness_weights["recovery_possible"]
        
        return score
    
    def _select_diverse_patterns(
        self,
        scored_patterns: List[tuple],
        count: int,
        diversity_weight: float
    ) -> List[Union[dict, TrailPattern]]:
        """Select patterns balancing score and diversity."""
        if len(scored_patterns) <= count:
            return [pattern for pattern, score in scored_patterns]
        
        selected = []
        remaining = scored_patterns.copy()
        
        # Always select top-scored pattern
        selected.append(remaining.pop(0)[0])
        
        # Select remaining patterns with diversity consideration
        while len(selected) < count and remaining:
            best_candidate = None
            best_combined_score = -1
            best_index = -1
            
            for i, (pattern, relevance_score) in enumerate(remaining):
                # Calculate diversity bonus
                diversity_bonus = self._calculate_diversity_bonus(pattern, selected)
                
                # Combined score
                combined_score = (
                    (1 - diversity_weight) * relevance_score +
                    diversity_weight * diversity_bonus
                )
                
                if combined_score > best_combined_score:
                    best_combined_score = combined_score
                    best_candidate = pattern
                    best_index = i
            
            if best_candidate:
                selected.append(best_candidate)
                remaining.pop(best_index)
            else:
                break
        
        return selected
    
    def _calculate_diversity_bonus(
        self,
        candidate: Union[dict, TrailPattern],
        selected: List[Union[dict, TrailPattern]]
    ) -> float:
        """Calculate diversity bonus for candidate pattern."""
        if not selected:
            return 1.0
        
        # Extract features for diversity calculation
        def extract_features(pattern):
            if isinstance(pattern, dict):
                return {
                    "category": pattern.get("category", "unknown"),
                    "domain": pattern.get("domain", "general"),
                    "type": "local"
                }
            else:
                return {
                    "category": pattern.error_type,
                    "domain": pattern.domain,
                    "type": "trail",
                    "complexity": pattern.complexity
                }
        
        candidate_features = extract_features(candidate)
        
        # Calculate diversity score
        diversity_score = 0.0
        feature_weights = {"category": 0.4, "domain": 0.3, "type": 0.2, "complexity": 0.1}
        
        for feature, weight in feature_weights.items():
            if feature in candidate_features:
                # Count how many selected patterns share this feature
                matches = sum(
                    1 for selected_pattern in selected
                    if extract_features(selected_pattern).get(feature) == candidate_features[feature]
                )
                
                # Higher diversity bonus for less common features
                feature_diversity = 1.0 - (matches / len(selected))
                diversity_score += weight * feature_diversity
        
        return diversity_score
    
    def _calculate_coverage_metrics(
        self,
        selected_patterns: List[Union[dict, TrailPattern]],
        assumption_types: List[str]
    ) -> Dict[str, Any]:
        """Calculate coverage metrics for selected patterns."""
        coverage = {
            "assumption_types_covered": set(),
            "error_types_covered": set(),
            "domains_covered": set(),
            "pattern_sources": {"local": 0, "trail": 0}
        }
        
        for pattern in selected_patterns:
            if isinstance(pattern, dict):
                coverage["error_types_covered"].add(pattern.get("category", "unknown"))
                coverage["domains_covered"].add(pattern.get("domain", "general"))
                coverage["pattern_sources"]["local"] += 1
            else:
                coverage["error_types_covered"].add(pattern.error_type)
                coverage["domains_covered"].add(pattern.domain)
                coverage["pattern_sources"]["trail"] += 1
        
        # Convert sets to lists for serialization
        return {
            "assumption_types_covered": list(coverage["assumption_types_covered"]),
            "error_types_covered": list(coverage["error_types_covered"]),
            "domains_covered": list(coverage["domains_covered"]),
            "pattern_sources": coverage["pattern_sources"],
            "coverage_ratios": {
                "assumption_coverage": len(coverage["assumption_types_covered"]) / max(len(assumption_types), 1),
                "error_type_diversity": len(coverage["error_types_covered"]) / max(len(self._patterns_by_category), 1),
                "domain_diversity": len(coverage["domains_covered"]) / max(len(self._patterns_by_domain), 1)
            }
        }
    
    def _estimate_pattern_diversity(
        self,
        selected_patterns: List[Union[dict, TrailPattern]]
    ) -> float:
        """Estimate diversity of selected patterns."""
        if len(selected_patterns) <= 1:
            return 1.0
        
        # Calculate diversity based on feature distribution
        features = {
            "categories": [],
            "domains": [],
            "types": []
        }
        
        for pattern in selected_patterns:
            if isinstance(pattern, dict):
                features["categories"].append(pattern.get("category", "unknown"))
                features["domains"].append(pattern.get("domain", "general"))
                features["types"].append("local")
            else:
                features["categories"].append(pattern.error_type)
                features["domains"].append(pattern.domain)
                features["types"].append("trail")
        
        # Calculate diversity as ratio of unique values to total
        diversity_scores = []
        for feature_type, values in features.items():
            unique_ratio = len(set(values)) / len(values)
            diversity_scores.append(unique_ratio)
        
        return sum(diversity_scores) / len(diversity_scores)
    
    def get_patterns_by_category(self, category: str) -> List[Union[dict, TrailPattern]]:
        """Get patterns by category."""
        return self._patterns_by_category.get(category, [])
    
    def get_patterns_by_domain(self, domain: str) -> List[Union[dict, TrailPattern]]:
        """Get patterns by domain."""
        return self._patterns_by_domain.get(domain, [])
    
    def get_trail_patterns(self) -> List[TrailPattern]:
        """Get all TRAIL patterns."""
        return self._patterns_by_type.get("trail", [])
    
    def get_local_patterns(self) -> List[dict]:
        """Get all local patterns."""
        return self._patterns_by_type.get("local", [])
    
    def get_all_patterns(self) -> List[Union[dict, TrailPattern]]:
        """Get all patterns."""
        return self._all_patterns.copy()
    
    def get_metrics(self) -> PatternMetrics:
        """Get pattern library metrics."""
        return self._metrics
    
    def get_version_info(self) -> Dict[str, Any]:
        """Get version information for reproducibility."""
        return {
            "library_version": self.version,
            "pattern_version_hash": self._pattern_version_hash,
            "total_patterns": len(self._all_patterns),
            "last_updated": self._metrics.last_updated,
            "trail_dataset_stats": self.trail_loader.get_stats().to_dict() if self.trail_loader.is_loaded() else None
        }
    
    def clear_cache(self) -> None:
        """Clear selection cache."""
        self._selection_cache.clear()
        logger.info("Pattern selection cache cleared")
    
    def is_loaded(self) -> bool:
        """Check if library is loaded."""
        return self._loaded
    
    async def validate_patterns(self) -> Dict[str, Any]:
        """Validate pattern library integrity."""
        if not self._loaded:
            await self.initialize()
        
        validation_results = {
            "total_patterns": len(self._all_patterns),
            "valid_patterns": 0,
            "invalid_patterns": 0,
            "validation_errors": [],
            "category_distribution": {},
            "domain_distribution": {}
        }
        
        for i, pattern in enumerate(self._all_patterns):
            try:
                if isinstance(pattern, dict):
                    # Validate local pattern
                    required_fields = ["id", "category", "description"]
                    missing_fields = [field for field in required_fields if not pattern.get(field)]
                    
                    if missing_fields:
                        validation_results["invalid_patterns"] += 1
                        validation_results["validation_errors"].append(
                            f"Local pattern {i}: Missing fields {missing_fields}"
                        )
                    else:
                        validation_results["valid_patterns"] += 1
                        
                        # Update distribution
                        category = pattern.get("category", "unknown")
                        domain = pattern.get("domain", "general")
                        
                else:
                    # Validate TRAIL pattern
                    if not pattern.id or not pattern.description:
                        validation_results["invalid_patterns"] += 1
                        validation_results["validation_errors"].append(
                            f"TRAIL pattern {i}: Missing required fields"
                        )
                    else:
                        validation_results["valid_patterns"] += 1
                        
                        # Update distribution
                        category = pattern.error_type
                        domain = pattern.domain
                
                # Update distributions
                validation_results["category_distribution"][category] = (
                    validation_results["category_distribution"].get(category, 0) + 1
                )
                validation_results["domain_distribution"][domain] = (
                    validation_results["domain_distribution"].get(domain, 0) + 1
                )
                
            except Exception as e:
                validation_results["invalid_patterns"] += 1
                validation_results["validation_errors"].append(
                    f"Pattern {i}: Validation error - {str(e)}"
                )
        
        validation_results["validity_rate"] = (
            validation_results["valid_patterns"] / max(validation_results["total_patterns"], 1)
        )
        
        return validation_results