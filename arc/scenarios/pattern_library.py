"""
Enhanced pattern library that combines existing patterns with TRAIL dataset.

Manages a unified pattern library with versioning, systematic violation scenarios,
and integration between local patterns and real-world TRAIL failure patterns.
"""

import json
import asyncio
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from pathlib import Path
from datetime import datetime, timedelta
import hashlib

from .failure_patterns.library import FailurePattern, PatternLibrary
from .trail_loader import TrailDatasetLoader, TrailDatasetMetrics
from .assumption_extractor import AgentAssumptions


@dataclass
class PatternLibraryMetrics:
    """Metrics for the enhanced pattern library."""
    total_patterns: int = 0
    local_patterns: int = 0
    trail_patterns: int = 0
    by_category: Dict[str, int] = field(default_factory=dict)
    by_source: Dict[str, int] = field(default_factory=dict)
    library_version: str = "1.0.0"
    last_updated: str = ""
    trail_integration_time: float = 0.0


@dataclass
class PatternSelection:
    """Result of pattern selection with diversity metrics."""
    selected_patterns: List[FailurePattern]
    selection_method: str
    diversity_score: float
    coverage_by_category: Dict[str, int]
    assumptions_targeted: List[str]


class EnhancedPatternLibrary:
    """Enhanced pattern library combining local and TRAIL patterns."""
    
    def __init__(
        self,
        local_patterns_dir: Optional[Path] = None,
        trail_cache_dir: Optional[Path] = None,
        enable_trail_integration: bool = True
    ):
        """Initialize enhanced pattern library.
        
        Args:
            local_patterns_dir: Directory with existing patterns
            trail_cache_dir: Directory for TRAIL cache
            enable_trail_integration: Whether to load TRAIL patterns
        """
        self.enable_trail_integration = enable_trail_integration
        
        # Initialize local pattern library
        self.local_library = PatternLibrary(local_patterns_dir)
        
        # Initialize TRAIL loader
        self.trail_loader = TrailDatasetLoader(trail_cache_dir) if enable_trail_integration else None
        
        # Combined pattern storage
        self.all_patterns: Dict[str, FailurePattern] = {}
        self.patterns_by_category: Dict[str, List[str]] = {}
        self.patterns_by_source: Dict[str, List[str]] = {"local": [], "trail": []}
        
        # Versioning and caching
        self.library_version = self._generate_version()
        self.last_trail_update: Optional[datetime] = None
        
        # Performance tracking
        self.selection_history: List[Dict[str, Any]] = []
        
    async def initialize(self) -> PatternLibraryMetrics:
        """Initialize the library by loading all patterns.
        
        Returns:
            Metrics about the loaded patterns
        """
        start_time = datetime.now()
        
        # Load local patterns (already loaded in __init__)
        local_patterns = self.local_library.get_all_patterns()
        
        # Add local patterns to combined storage
        for pattern in local_patterns:
            pattern_id = f"local_{pattern.id}"
            self.all_patterns[pattern_id] = pattern
            self.patterns_by_source["local"].append(pattern_id)
            
            category = pattern.category
            if category not in self.patterns_by_category:
                self.patterns_by_category[category] = []
            self.patterns_by_category[category].append(pattern_id)
        
        # Load TRAIL patterns if enabled
        trail_patterns_count = 0
        trail_integration_time = 0.0
        
        if self.enable_trail_integration and self.trail_loader:
            try:
                trail_start = datetime.now()
                
                # Load TRAIL patterns (limit to manageable number for performance)
                trail_patterns, trail_metrics = await self.trail_loader.load_trail_patterns(limit=400)
                
                # Convert to failure patterns
                failure_patterns, conversion_time = await self.trail_loader.convert_to_failure_patterns(trail_patterns)
                
                # Add TRAIL patterns to combined storage
                for pattern in failure_patterns:
                    pattern_id = f"trail_{pattern.id}"
                    self.all_patterns[pattern_id] = pattern
                    self.patterns_by_source["trail"].append(pattern_id)
                    
                    category = pattern.category
                    if category not in self.patterns_by_category:
                        self.patterns_by_category[category] = []
                    self.patterns_by_category[category].append(pattern_id)
                
                trail_patterns_count = len(failure_patterns)
                trail_integration_time = (datetime.now() - trail_start).total_seconds()
                self.last_trail_update = datetime.now()
                
                print(f"✅ Integrated {trail_patterns_count} TRAIL patterns in {trail_integration_time:.2f}s")
                
            except Exception as e:
                print(f"⚠️ Failed to load TRAIL patterns: {e}")
                print("   Continuing with local patterns only")
        
        # Generate metrics
        metrics = PatternLibraryMetrics(
            total_patterns=len(self.all_patterns),
            local_patterns=len(self.patterns_by_source["local"]),
            trail_patterns=trail_patterns_count,
            by_category={cat: len(patterns) for cat, patterns in self.patterns_by_category.items()},
            by_source={src: len(patterns) for src, patterns in self.patterns_by_source.items()},
            library_version=self.library_version,
            last_updated=datetime.now().isoformat(),
            trail_integration_time=trail_integration_time
        )
        
        return metrics
    
    async def select_patterns_for_assumptions(
        self,
        assumptions: AgentAssumptions,
        count: int = 35,
        prioritize_trail: bool = True,
        assumption_types: Optional[List[str]] = None
    ) -> PatternSelection:
        """Select patterns optimized for testing specific assumptions.
        
        Args:
            assumptions: Agent assumptions to target
            count: Number of patterns to select
            prioritize_trail: Whether to prioritize TRAIL patterns
            assumption_types: Specific assumption types to focus on
            
        Returns:
            Pattern selection with diversity metrics
        """
        # Get relevant patterns based on tools and assumptions
        relevant_patterns = self._get_patterns_for_assumptions(assumptions, assumption_types)
        
        # Apply source prioritization
        if prioritize_trail and self.enable_trail_integration:
            relevant_patterns = self._prioritize_trail_patterns(relevant_patterns)
        
        # Select diverse patterns
        selected_patterns = self._select_diverse_patterns(
            relevant_patterns,
            count,
            assumptions
        )
        
        # Calculate diversity metrics
        diversity_score = self._calculate_diversity_score(selected_patterns)
        coverage_by_category = self._calculate_category_coverage(selected_patterns)
        assumptions_targeted = self._identify_targeted_assumptions(selected_patterns, assumptions)
        
        # Track selection for analytics
        selection_record = {
            "timestamp": datetime.now().isoformat(),
            "assumptions_count": len(assumptions.to_dict()),
            "selected_count": len(selected_patterns),
            "diversity_score": diversity_score,
            "prioritize_trail": prioritize_trail
        }
        self.selection_history.append(selection_record)
        
        return PatternSelection(
            selected_patterns=selected_patterns,
            selection_method="assumption_optimized",
            diversity_score=diversity_score,
            coverage_by_category=coverage_by_category,
            assumptions_targeted=assumptions_targeted
        )
    
    def get_pattern(self, pattern_id: str) -> Optional[FailurePattern]:
        """Get a specific pattern by ID."""
        return self.all_patterns.get(pattern_id)
    
    def get_patterns_by_category(self, category: str) -> List[FailurePattern]:
        """Get all patterns in a category."""
        pattern_ids = self.patterns_by_category.get(category, [])
        return [self.all_patterns[pid] for pid in pattern_ids if pid in self.all_patterns]
    
    def get_trail_patterns(self) -> List[FailurePattern]:
        """Get only TRAIL-sourced patterns."""
        pattern_ids = self.patterns_by_source.get("trail", [])
        return [self.all_patterns[pid] for pid in pattern_ids if pid in self.all_patterns]
    
    def get_local_patterns(self) -> List[FailurePattern]:
        """Get only locally-sourced patterns."""
        pattern_ids = self.patterns_by_source.get("local", [])
        return [self.all_patterns[pid] for pid in pattern_ids if pid in self.all_patterns]
    
    def _get_patterns_for_assumptions(
        self,
        assumptions: AgentAssumptions,
        assumption_types: Optional[List[str]] = None
    ) -> List[FailurePattern]:
        """Get patterns relevant to specific assumptions."""
        relevant_patterns = []
        
        # Map assumption types to relevant pattern categories
        assumption_category_map = {
            "currency": ["calculation", "logic", "data"],
            "data_format": ["data", "infrastructure", "navigation"],
            "api_version": ["infrastructure", "navigation", "auth"],
            "timeout": ["infrastructure", "temporal"],
            "rate_limit": ["infrastructure", "security"],
            "tool": ["navigation", "multi_agent"],
            "error_handling": ["logic", "infrastructure"],
            "region": ["infrastructure", "temporal"]
        }
        
        # Collect relevant categories
        relevant_categories = set()
        
        # Add categories based on assumptions
        if not assumption_types or "currency" in assumption_types:
            if assumptions.currencies:
                relevant_categories.update(assumption_category_map["currency"])
        
        if not assumption_types or "data_format" in assumption_types:
            if assumptions.data_formats:
                relevant_categories.update(assumption_category_map["data_format"])
        
        if not assumption_types or "api_version" in assumption_types:
            if assumptions.api_versions:
                relevant_categories.update(assumption_category_map["api_version"])
        
        if not assumption_types or "timeout" in assumption_types:
            if assumptions.timeouts:
                relevant_categories.update(assumption_category_map["timeout"])
        
        if not assumption_types or "rate_limit" in assumption_types:
            if assumptions.rate_limits:
                relevant_categories.update(assumption_category_map["rate_limit"])
        
        if not assumption_types or "tool" in assumption_types:
            if assumptions.tools:
                relevant_categories.update(assumption_category_map["tool"])
        
        # Add patterns from tools (using local library logic)
        tool_patterns = self.local_library.get_patterns_for_tools(assumptions.tools)
        relevant_patterns.extend(tool_patterns)
        
        # Add patterns from relevant categories
        for category in relevant_categories:
            category_patterns = self.get_patterns_by_category(category)
            relevant_patterns.extend(category_patterns)
        
        # Deduplicate while preserving order
        seen = set()
        unique_patterns = []
        for pattern in relevant_patterns:
            pattern_key = f"{pattern.id}_{pattern.title}"
            if pattern_key not in seen:
                seen.add(pattern_key)
                unique_patterns.append(pattern)
        
        return unique_patterns
    
    def _prioritize_trail_patterns(self, patterns: List[FailurePattern]) -> List[FailurePattern]:
        """Prioritize TRAIL patterns while maintaining some local patterns."""
        trail_patterns = [p for p in patterns if p.id.startswith("trail_")]
        local_patterns = [p for p in patterns if p.id.startswith("local_")]
        
        # Aim for 70% TRAIL, 30% local for diversity
        trail_ratio = 0.7
        total_patterns = len(patterns)
        trail_target = int(total_patterns * trail_ratio)
        local_target = total_patterns - trail_target
        
        # Select best patterns from each source
        prioritized = []
        
        # Add TRAIL patterns (sorted by relevance)
        trail_patterns.sort(key=lambda p: self._calculate_pattern_priority(p), reverse=True)
        prioritized.extend(trail_patterns[:trail_target])
        
        # Add local patterns (sorted by relevance) 
        local_patterns.sort(key=lambda p: self._calculate_pattern_priority(p), reverse=True)
        prioritized.extend(local_patterns[:local_target])
        
        # Fill remaining slots with remaining patterns
        remaining = trail_patterns[trail_target:] + local_patterns[local_target:]
        remaining.sort(key=lambda p: self._calculate_pattern_priority(p), reverse=True)
        
        prioritized.extend(remaining[:total_patterns - len(prioritized)])
        
        return prioritized[:total_patterns]
    
    def _select_diverse_patterns(
        self,
        available_patterns: List[FailurePattern],
        count: int,
        assumptions: AgentAssumptions
    ) -> List[FailurePattern]:
        """Select diverse patterns ensuring good coverage."""
        if len(available_patterns) <= count:
            return available_patterns
        
        # Group patterns by category for diversity
        by_category = {}
        for pattern in available_patterns:
            if pattern.category not in by_category:
                by_category[pattern.category] = []
            by_category[pattern.category].append(pattern)
        
        # Calculate target distribution
        categories = list(by_category.keys())
        patterns_per_category = max(1, count // len(categories))
        
        selected = []
        
        # First pass: select top patterns from each category
        for category in categories:
            category_patterns = by_category[category]
            # Sort by priority within category
            category_patterns.sort(key=lambda p: self._calculate_pattern_priority(p), reverse=True)
            
            # Select patterns from this category
            to_select = min(patterns_per_category, len(category_patterns), count - len(selected))
            selected.extend(category_patterns[:to_select])
        
        # Second pass: fill remaining slots with highest priority patterns
        if len(selected) < count:
            remaining_patterns = [p for p in available_patterns if p not in selected]
            remaining_patterns.sort(key=lambda p: self._calculate_pattern_priority(p), reverse=True)
            
            needed = count - len(selected)
            selected.extend(remaining_patterns[:needed])
        
        return selected[:count]
    
    def _calculate_pattern_priority(self, pattern: FailurePattern) -> float:
        """Calculate priority score for pattern selection."""
        score = 0.0
        
        # Base score from severity
        severity_scores = {"critical": 4.0, "high": 3.0, "medium": 2.0, "low": 1.0}
        score += severity_scores.get(pattern.severity, 2.0)
        
        # Boost for TRAIL patterns (real-world data)
        if pattern.id.startswith("trail_"):
            score += 1.0
        
        # Boost for frequency (more common = higher priority for testing)
        frequency_scores = {"very_common": 2.0, "common": 1.5, "moderate": 1.0, "rare": 0.5, "very_rare": 0.2}
        score += frequency_scores.get(pattern.frequency, 1.0)
        
        # Boost for patterns with good example instantiation
        if pattern.example_instantiation and len(pattern.example_instantiation) > 20:
            score += 0.5
        
        return score
    
    def _calculate_diversity_score(self, patterns: List[FailurePattern]) -> float:
        """Calculate diversity score for selected patterns."""
        if not patterns:
            return 0.0
        
        # Count unique categories
        categories = set(p.category for p in patterns)
        category_diversity = len(categories) / len(patterns)
        
        # Count unique sources
        sources = set("trail" if p.id.startswith("trail_") else "local" for p in patterns)
        source_diversity = len(sources) / 2.0  # Max 2 sources
        
        # Count severity distribution
        severities = set(p.severity for p in patterns)
        severity_diversity = len(severities) / 4.0  # Max 4 severities
        
        # Weighted average
        diversity_score = (
            category_diversity * 0.5 +
            source_diversity * 0.3 +
            severity_diversity * 0.2
        )
        
        return min(1.0, diversity_score)
    
    def _calculate_category_coverage(self, patterns: List[FailurePattern]) -> Dict[str, int]:
        """Calculate coverage by category."""
        coverage = {}
        for pattern in patterns:
            coverage[pattern.category] = coverage.get(pattern.category, 0) + 1
        return coverage
    
    def _identify_targeted_assumptions(
        self,
        patterns: List[FailurePattern],
        assumptions: AgentAssumptions
    ) -> List[str]:
        """Identify which assumptions are targeted by selected patterns."""
        targeted = set()
        
        # Map pattern categories back to assumptions
        for pattern in patterns:
            if pattern.category in ["calculation", "logic"] and assumptions.currencies:
                targeted.add("currency")
            if pattern.category in ["data", "infrastructure"] and assumptions.data_formats:
                targeted.add("data_format")
            if pattern.category in ["infrastructure", "navigation"] and assumptions.api_versions:
                targeted.add("api_version")
            if pattern.category == "infrastructure" and assumptions.timeouts:
                targeted.add("timeout")
            if pattern.category in ["navigation", "multi_agent"] and assumptions.tools:
                targeted.add("tool")
        
        return list(targeted)
    
    def _generate_version(self) -> str:
        """Generate version string for the library."""
        # Simple version based on timestamp
        timestamp = datetime.now().strftime("%Y%m%d")
        return f"1.0.{timestamp}"
    
    def get_library_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics about the library."""
        stats = {
            "total_patterns": len(self.all_patterns),
            "patterns_by_source": {src: len(patterns) for src, patterns in self.patterns_by_source.items()},
            "patterns_by_category": {cat: len(patterns) for cat, patterns in self.patterns_by_category.items()},
            "library_version": self.library_version,
            "trail_integration_enabled": self.enable_trail_integration,
            "last_trail_update": self.last_trail_update.isoformat() if self.last_trail_update else None,
            "selection_history_count": len(self.selection_history)
        }
        
        # Add pattern quality metrics
        all_patterns = list(self.all_patterns.values())
        if all_patterns:
            severity_counts = {}
            frequency_counts = {}
            
            for pattern in all_patterns:
                severity_counts[pattern.severity] = severity_counts.get(pattern.severity, 0) + 1
                frequency_counts[pattern.frequency] = frequency_counts.get(pattern.frequency, 0) + 1
            
            stats["quality_distribution"] = {
                "by_severity": severity_counts,
                "by_frequency": frequency_counts,
                "avg_description_length": sum(len(p.description) for p in all_patterns) / len(all_patterns)
            }
        
        return stats