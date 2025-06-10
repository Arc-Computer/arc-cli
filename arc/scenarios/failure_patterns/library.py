"""
Pattern library for loading and managing failure patterns.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass, field
import hashlib
from collections import defaultdict


@dataclass
class FailurePattern:
    """Represents a failure pattern from the library."""
    id: str
    title: str
    category: str
    severity: str = "medium"
    description: str = ""
    trigger_conditions: List[str] = field(default_factory=list)
    expected_error: str = ""
    recovery_patterns: Optional[List[str]] = None
    frequency: str = "moderate"
    example_instantiation: Optional[str] = None
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FailurePattern':
        """Create pattern from dictionary."""
        return cls(
            id=data.get('id', ''),
            title=data.get('title', ''),
            category=data.get('category', ''),
            severity=data.get('severity', 'medium'),
            description=data.get('description', ''),
            trigger_conditions=data.get('trigger_conditions', []),
            expected_error=data.get('expected_error', ''),
            recovery_patterns=data.get('recovery_patterns'),
            frequency=data.get('frequency', 'moderate'),
            example_instantiation=data.get('example_instantiation')
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert pattern to dictionary."""
        return {
            'id': self.id,
            'title': self.title,
            'category': self.category,
            'severity': self.severity,
            'description': self.description,
            'trigger_conditions': self.trigger_conditions,
            'expected_error': self.expected_error,
            'recovery_patterns': self.recovery_patterns,
            'frequency': self.frequency,
            'example_instantiation': self.example_instantiation
        }
    
    def get_frequency_weight(self) -> int:
        """Get numeric weight based on frequency."""
        frequency_map = {
            "very_common": 10,
            "common": 7,
            "moderate": 5,
            "rare": 3,
            "very_rare": 1
        }
        return frequency_map.get(self.frequency.lower(), 5)


class PatternLibrary:
    """Manages the failure pattern library."""
    
    def __init__(self, patterns_dir: Optional[Path] = None):
        """Initialize pattern library.
        
        Args:
            patterns_dir: Directory containing pattern files. If None, uses default location.
        """
        if patterns_dir is None:
            # Use production patterns directory
            patterns_dir = Path(__file__).parent / "patterns"
        
        self.patterns_dir = patterns_dir
        self.patterns: Dict[str, FailurePattern] = {}
        self.patterns_by_category: Dict[str, List[str]] = defaultdict(list)
        self.patterns_by_severity: Dict[str, List[str]] = defaultdict(list)
        self._recently_used: List[str] = []
        
        # Load patterns if directory exists
        if self.patterns_dir.exists():
            self._load_patterns()
    
    def _load_patterns(self) -> None:
        """Load all patterns from the library."""
        loaded_count = 0
        
        for category_dir in self.patterns_dir.iterdir():
            if category_dir.is_dir() and not category_dir.name.startswith('.'):
                category = category_dir.name
                
                for pattern_file in category_dir.glob("*.json"):
                    try:
                        with open(pattern_file, 'r') as f:
                            data = json.load(f)
                        
                        pattern = FailurePattern.from_dict(data)
                        pattern.category = category  # Ensure category matches directory
                        
                        self.patterns[pattern.id] = pattern
                        self.patterns_by_category[category].append(pattern.id)
                        self.patterns_by_severity[pattern.severity].append(pattern.id)
                        loaded_count += 1
                        
                    except Exception as e:
                        print(f"Warning: Failed to load pattern {pattern_file}: {e}")
        
        if loaded_count > 0:
            print(f"Loaded {loaded_count} patterns across {len(self.patterns_by_category)} categories")
    
    def get_pattern(self, pattern_id: str) -> Optional[FailurePattern]:
        """Get a specific pattern by ID."""
        return self.patterns.get(pattern_id)
    
    def get_patterns_by_category(self, category: str) -> List[FailurePattern]:
        """Get all patterns in a category."""
        pattern_ids = self.patterns_by_category.get(category, [])
        return [self.patterns[pid] for pid in pattern_ids if pid in self.patterns]
    
    def get_patterns_by_severity(self, severity: str) -> List[FailurePattern]:
        """Get all patterns with a specific severity."""
        pattern_ids = self.patterns_by_severity.get(severity, [])
        return [self.patterns[pid] for pid in pattern_ids if pid in self.patterns]
    
    def get_patterns_for_tools(self, tools: List[str]) -> List[FailurePattern]:
        """Get relevant patterns based on available tools.
        
        Maps tools to likely failure categories and returns relevant patterns.
        """
        # Map tools to relevant categories (based on TRAIL taxonomy)
        tool_category_map = {
            # Data tools
            "search": ["data", "infrastructure"],
            "database": ["data", "logic", "infrastructure"],
            "api": ["data", "infrastructure", "auth"],
            "file": ["data", "security", "auth"],
            
            # Calculation tools
            "calculator": ["calculation", "logic"],
            "math": ["calculation", "precision"],
            
            # Navigation tools
            "browser": ["navigation", "security", "infrastructure"],
            "web": ["navigation", "infrastructure", "temporal"],
            
            # Infrastructure tools
            "http": ["infrastructure", "auth", "security"],
            "shell": ["security", "infrastructure"],
            "terminal": ["security", "infrastructure"],
            
            # Multi-agent tools
            "agent": ["multi_agent", "logic", "temporal"],
            "coordinator": ["multi_agent", "temporal"],
            
            # General mappings
            "weather": ["data", "infrastructure", "temporal"],
            "currency": ["calculation", "data", "temporal"],
            "finance": ["calculation", "data", "security"]
        }
        
        relevant_categories = set()
        
        # Match tools to categories
        for tool in tools:
            tool_lower = tool.lower()
            
            # Direct matches
            if tool_lower in tool_category_map:
                relevant_categories.update(tool_category_map[tool_lower])
            
            # Partial matches
            for key, categories in tool_category_map.items():
                if key in tool_lower or tool_lower in key:
                    relevant_categories.update(categories)
        
        # Always include some general categories
        relevant_categories.update(["infrastructure", "data", "logic"])
        
        # Collect patterns from relevant categories
        relevant_patterns = []
        for category in relevant_categories:
            relevant_patterns.extend(self.get_patterns_by_category(category))
        
        # Remove duplicates while preserving order
        seen = set()
        unique_patterns = []
        for pattern in relevant_patterns:
            if pattern.id not in seen:
                seen.add(pattern.id)
                unique_patterns.append(pattern)
        
        return unique_patterns
    
    def select_diverse_patterns(
        self, 
        available_patterns: List[FailurePattern], 
        count: int = 3,
        prioritize_severity: bool = True
    ) -> List[FailurePattern]:
        """Select diverse patterns avoiding recent usage.
        
        Args:
            available_patterns: Patterns to select from
            count: Number of patterns to select
            prioritize_severity: Whether to prioritize high-severity patterns
        
        Returns:
            Selected patterns with diversity across categories
        """
        if not available_patterns:
            return []
        
        # Filter out recently used patterns
        candidates = [p for p in available_patterns if p.id not in self._recently_used[-10:]]
        
        # If all patterns were recently used, reset and use all
        if not candidates:
            self._recently_used = []
            candidates = available_patterns
        
        # Group by category for diversity
        by_category = defaultdict(list)
        for pattern in candidates:
            by_category[pattern.category].append(pattern)
        
        selected = []
        
        # First, try to get one from each category (for diversity)
        categories = list(by_category.keys())
        for i in range(min(count, len(categories))):
            category_patterns = by_category[categories[i]]
            
            if prioritize_severity:
                # Sort by severity and frequency
                category_patterns.sort(
                    key=lambda p: (
                        {"critical": 4, "high": 3, "medium": 2, "low": 1}.get(p.severity, 2),
                        p.get_frequency_weight()
                    ),
                    reverse=True
                )
            
            if category_patterns:
                selected.append(category_patterns[0])
                # Remove from available to avoid duplicates
                by_category[categories[i]] = category_patterns[1:]
        
        # Fill remaining slots with highest priority patterns
        if len(selected) < count:
            remaining = []
            for patterns in by_category.values():
                remaining.extend(patterns)
            
            if prioritize_severity:
                remaining.sort(
                    key=lambda p: (
                        {"critical": 4, "high": 3, "medium": 2, "low": 1}.get(p.severity, 2),
                        p.get_frequency_weight()
                    ),
                    reverse=True
                )
            
            selected.extend(remaining[:count - len(selected)])
        
        # Track selected patterns
        for pattern in selected:
            self._recently_used.append(pattern.id)
        
        # Keep only last 20 patterns in history
        self._recently_used = self._recently_used[-20:]
        
        return selected[:count]
    
    def get_all_patterns(self) -> List[FailurePattern]:
        """Get all patterns in the library."""
        return list(self.patterns.values())
    
    def get_pattern_stats(self) -> Dict[str, Any]:
        """Get statistics about the pattern library."""
        severity_counts = defaultdict(int)
        category_counts = defaultdict(int)
        
        for pattern in self.patterns.values():
            severity_counts[pattern.severity] += 1
            category_counts[pattern.category] += 1
        
        return {
            "total_patterns": len(self.patterns),
            "categories": dict(category_counts),
            "severities": dict(severity_counts),
            "recently_used": len(self._recently_used)
        }