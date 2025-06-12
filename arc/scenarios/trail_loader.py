"""
TRAIL dataset loader for integrating real-world agent failure patterns.

Loads and preprocesses the PatronusAI/TRAIL dataset containing 841 failure patterns
from 148 execution traces, categorizing them by error type for scenario generation.
"""

import json
import asyncio
from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass, field
from pathlib import Path
import hashlib
import httpx
from datetime import datetime, timedelta

from .failure_patterns.library import FailurePattern


@dataclass
class TrailPattern:
    """Represents a failure pattern from the TRAIL dataset."""
    id: str
    trace_id: str
    step_number: int
    error_type: str  # reasoning, execution, planning
    error_category: str  # hallucination, api_failure, coordination_error, etc.
    error_description: str
    failure_context: str
    recovery_attempted: bool
    severity_score: float = 0.0
    domain: str = ""
    tools_involved: List[str] = field(default_factory=list)
    
    def to_failure_pattern(self) -> FailurePattern:
        """Convert TRAIL pattern to internal FailurePattern format."""
        # Map TRAIL error types to our categories
        category_mapping = {
            "reasoning": "logic",
            "execution": "infrastructure", 
            "planning": "multi_agent",
            "api_failure": "infrastructure",
            "hallucination": "logic",
            "coordination_error": "multi_agent",
            "tool_error": "navigation",
            "data_error": "data"
        }
        
        # Map severity score to severity level
        if self.severity_score >= 0.8:
            severity = "critical"
        elif self.severity_score >= 0.6:
            severity = "high"
        elif self.severity_score >= 0.4:
            severity = "medium"
        else:
            severity = "low"
        
        # Generate trigger conditions based on context
        trigger_conditions = self._extract_trigger_conditions()
        
        # Generate recovery patterns
        recovery_patterns = self._extract_recovery_patterns()
        
        return FailurePattern(
            id=f"trail_{self.id}",
            title=self._generate_title(),
            category=category_mapping.get(self.error_type, "logic"),
            severity=severity,
            description=self._clean_description(),
            trigger_conditions=trigger_conditions,
            expected_error=self.error_description,
            recovery_patterns=recovery_patterns,
            frequency=self._estimate_frequency(),
            example_instantiation=self._generate_example()
        )
    
    def _extract_trigger_conditions(self) -> List[str]:
        """Extract trigger conditions from failure context."""
        conditions = []
        
        # Common patterns based on TRAIL analysis
        context_lower = self.failure_context.lower()
        
        if "api" in context_lower or "request" in context_lower:
            conditions.append("external API dependency")
        if "timeout" in context_lower or "slow" in context_lower:
            conditions.append("response delay")
        if "data" in context_lower or "parse" in context_lower:
            conditions.append("data processing")
        if "agent" in context_lower and "multiple" in context_lower:
            conditions.append("multi-agent coordination")
        if "tool" in context_lower:
            conditions.append("tool usage")
        if "auth" in context_lower or "permission" in context_lower:
            conditions.append("authentication required")
        
        # Add domain-specific conditions
        if self.domain:
            conditions.append(f"{self.domain} domain operations")
        
        return conditions[:3]  # Limit to most relevant
    
    def _extract_recovery_patterns(self) -> List[str]:
        """Extract potential recovery patterns."""
        patterns = []
        
        if self.recovery_attempted:
            patterns.append("retry operation")
            patterns.append("fallback strategy")
        
        # Add patterns based on error type
        if self.error_type == "reasoning":
            patterns.extend(["validate input", "check assumptions"])
        elif self.error_type == "execution":
            patterns.extend(["check connectivity", "verify permissions"])
        elif self.error_type == "planning":
            patterns.extend(["replan workflow", "coordinate agents"])
        
        return patterns[:3]  # Limit to most relevant
    
    def _clean_description(self) -> str:
        """Clean and format the error description."""
        desc = self.error_description.strip()
        if len(desc) > 200:
            desc = desc[:197] + "..."
        return desc
    
    def _generate_title(self) -> str:
        """Generate a human-readable title."""
        # Extract key terms from error description
        desc_words = self.error_description.lower().split()
        key_terms = []
        
        important_words = {
            "api", "timeout", "error", "failure", "invalid", "missing", 
            "unauthorized", "rate", "limit", "parse", "data", "agent"
        }
        
        for word in desc_words[:10]:  # Check first 10 words
            clean_word = word.strip(".,!?()[]")
            if clean_word in important_words:
                key_terms.append(clean_word.title())
        
        if key_terms:
            return f"{' '.join(key_terms)} in {self.error_type.title()}"
        else:
            return f"{self.error_type.title()} Error in Step {self.step_number}"
    
    def _estimate_frequency(self) -> str:
        """Estimate frequency based on TRAIL dataset patterns."""
        # Based on TRAIL analysis, reasoning errors are most common
        frequency_map = {
            "reasoning": "common",
            "execution": "moderate", 
            "planning": "rare"
        }
        
        # Adjust based on severity
        base_freq = frequency_map.get(self.error_type, "moderate")
        
        if self.severity_score >= 0.8:
            return "rare"  # High severity issues are typically rare
        elif self.severity_score <= 0.3:
            return "very_common"  # Low severity issues are common
        else:
            return base_freq
    
    def _generate_example(self) -> str:
        """Generate concrete example instantiation."""
        # Create example based on error type and context
        context_snippet = self.failure_context[:100] + "..." if len(self.failure_context) > 100 else self.failure_context
        
        return f"Agent encounters {self.error_description.lower()} during {context_snippet}"


@dataclass 
class TrailDatasetMetrics:
    """Metrics about the loaded TRAIL dataset."""
    total_patterns: int = 0
    by_error_type: Dict[str, int] = field(default_factory=dict)
    by_severity: Dict[str, int] = field(default_factory=dict)
    by_domain: Dict[str, int] = field(default_factory=dict)
    load_time_seconds: float = 0.0
    conversion_time_seconds: float = 0.0
    cache_hit: bool = False


class TrailDatasetLoader:
    """Loads and processes the TRAIL dataset from HuggingFace."""
    
    def __init__(self, cache_dir: Optional[Path] = None, cache_ttl_hours: int = 24):
        """Initialize the TRAIL dataset loader.
        
        Args:
            cache_dir: Directory for caching dataset. If None, uses temp directory.
            cache_ttl_hours: Hours to cache dataset before refresh
        """
        self.cache_dir = cache_dir or Path.home() / ".arc" / "cache" / "trail"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_ttl = timedelta(hours=cache_ttl_hours)
        
        # HuggingFace dataset API endpoints
        self.dataset_api_base = "https://datasets-server.huggingface.co"
        self.dataset_name = "PatronusAI/TRAIL" 
    
    async def load_trail_patterns(
        self, 
        limit: Optional[int] = None,
        error_types: Optional[List[str]] = None
    ) -> tuple[List[TrailPattern], TrailDatasetMetrics]:
        """Load TRAIL patterns from HuggingFace dataset.
        
        Args:
            limit: Maximum number of patterns to load
            error_types: Filter by error types (reasoning, execution, planning)
            
        Returns:
            Tuple of (patterns, metrics)
        """
        start_time = datetime.now()
        metrics = TrailDatasetMetrics()
        
        # Check cache first
        cache_file = self.cache_dir / "trail_patterns.json"
        if await self._is_cache_valid(cache_file):
            print("Loading TRAIL patterns from cache...")
            patterns = await self._load_from_cache(cache_file)
            metrics.cache_hit = True
        else:
            print("Fetching TRAIL patterns from HuggingFace...")
            patterns = await self._fetch_from_huggingface()
            await self._save_to_cache(cache_file, patterns)
            metrics.cache_hit = False
        
        metrics.load_time_seconds = (datetime.now() - start_time).total_seconds()
        
        # Filter patterns
        if error_types:
            patterns = [p for p in patterns if p.error_type in error_types]
        
        if limit:
            patterns = patterns[:limit]
        
        # Calculate metrics
        metrics.total_patterns = len(patterns)
        for pattern in patterns:
            # Count by error type
            metrics.by_error_type[pattern.error_type] = metrics.by_error_type.get(pattern.error_type, 0) + 1
            
            # Count by severity
            severity = "high" if pattern.severity_score >= 0.6 else "medium" if pattern.severity_score >= 0.3 else "low"
            metrics.by_severity[severity] = metrics.by_severity.get(severity, 0) + 1
            
            # Count by domain
            if pattern.domain:
                metrics.by_domain[pattern.domain] = metrics.by_domain.get(pattern.domain, 0) + 1
        
        print(f"Loaded {len(patterns)} TRAIL patterns: {dict(metrics.by_error_type)}")
        return patterns, metrics
    
    async def convert_to_failure_patterns(
        self, 
        trail_patterns: List[TrailPattern]
    ) -> tuple[List[FailurePattern], float]:
        """Convert TRAIL patterns to internal FailurePattern format.
        
        Args:
            trail_patterns: List of TRAIL patterns to convert
            
        Returns:
            Tuple of (failure_patterns, conversion_time_seconds)
        """
        start_time = datetime.now()
        
        failure_patterns = []
        seen_hashes = set()
        
        for trail_pattern in trail_patterns:
            try:
                failure_pattern = trail_pattern.to_failure_pattern()
                
                # Deduplicate based on pattern content
                pattern_hash = self._get_pattern_hash(failure_pattern)
                if pattern_hash not in seen_hashes:
                    seen_hashes.add(pattern_hash)
                    failure_patterns.append(failure_pattern)
                
            except Exception as e:
                print(f"Warning: Failed to convert TRAIL pattern {trail_pattern.id}: {e}")
        
        conversion_time = (datetime.now() - start_time).total_seconds()
        print(f"Converted {len(failure_patterns)} unique failure patterns ({len(trail_patterns) - len(failure_patterns)} duplicates removed)")
        
        return failure_patterns, conversion_time
    
    async def _fetch_from_huggingface(self) -> List[TrailPattern]:
        """Fetch TRAIL dataset from HuggingFace API."""
        patterns = []
        
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                # First, get dataset info
                info_url = f"{self.dataset_api_base}/info?dataset={self.dataset_name}"
                info_response = await client.get(info_url)
                info_response.raise_for_status()
                
                # Then fetch the actual data
                # Note: This is a simplified implementation - real TRAIL dataset
                # would require proper parsing of their format
                data_url = f"{self.dataset_api_base}/first-rows?dataset={self.dataset_name}&config=default&split=train"
                data_response = await client.get(data_url)
                data_response.raise_for_status()
                
                data = data_response.json()
                
                # Parse TRAIL data format (this would need to match actual TRAIL schema)
                if "rows" in data:
                    for i, row in enumerate(data["rows"][:841]):  # Limit to TRAIL's 841 patterns
                        patterns.append(self._parse_trail_row(row, i))
        
        except Exception as e:
            print(f"Warning: Could not fetch from HuggingFace ({e}), using synthetic TRAIL patterns")
            patterns = self._generate_synthetic_trail_patterns()
        
        return patterns
    
    def _parse_trail_row(self, row: Dict[str, Any], index: int) -> TrailPattern:
        """Parse a row from the TRAIL dataset."""
        # This would need to match the actual TRAIL dataset schema
        # For now, creating a synthetic pattern based on TRAIL characteristics
        
        error_types = ["reasoning", "execution", "planning"]
        domains = ["software_engineering", "information_retrieval", "finance", "general"]
        
        return TrailPattern(
            id=f"trail_{index:03d}",
            trace_id=row.get("trace_id", f"trace_{index}"),
            step_number=row.get("step", index % 10),
            error_type=row.get("error_type", error_types[index % len(error_types)]),
            error_category=row.get("category", "unknown"),
            error_description=row.get("error", f"Pattern {index} failure description"),
            failure_context=row.get("context", f"Context for pattern {index}"),
            recovery_attempted=row.get("recovery", False),
            severity_score=row.get("severity", (index * 0.1) % 1.0),
            domain=row.get("domain", domains[index % len(domains)])
        )
    
    def _generate_synthetic_trail_patterns(self) -> List[TrailPattern]:
        """Generate synthetic TRAIL patterns based on documented failure modes."""
        patterns = []
        
        # Based on TRAIL paper: 841 patterns across reasoning, execution, planning
        failure_templates = [
            # Reasoning errors (most common in TRAIL)
            {"type": "reasoning", "desc": "Agent hallucinates non-existent API endpoint", "severity": 0.7},
            {"type": "reasoning", "desc": "Incorrect parameter mapping in function call", "severity": 0.6},
            {"type": "reasoning", "desc": "Misinterprets user requirements", "severity": 0.5},
            {"type": "reasoning", "desc": "Logical inconsistency in multi-step plan", "severity": 0.8},
            
            # Execution errors
            {"type": "execution", "desc": "API timeout during critical operation", "severity": 0.6},
            {"type": "execution", "desc": "Permission denied accessing required resource", "severity": 0.7},
            {"type": "execution", "desc": "Network connectivity loss", "severity": 0.8},
            {"type": "execution", "desc": "Rate limit exceeded on external service", "severity": 0.4},
            
            # Planning errors (least common but high impact)
            {"type": "planning", "desc": "Circular dependency in task execution", "severity": 0.9},
            {"type": "planning", "desc": "Resource contention between parallel tasks", "severity": 0.7},
            {"type": "planning", "desc": "Incorrect task prioritization", "severity": 0.5},
            {"type": "planning", "desc": "Deadlock in multi-agent coordination", "severity": 0.9},
        ]
        
        domains = ["finance", "software_engineering", "information_retrieval", "general"]
        
        # Generate 841 patterns (as per TRAIL dataset size)
        for i in range(841):
            template = failure_templates[i % len(failure_templates)]
            domain = domains[i % len(domains)]
            
            patterns.append(TrailPattern(
                id=f"synthetic_trail_{i:03d}",
                trace_id=f"trace_{i // 10}",
                step_number=i % 15,  # Steps 0-14 as typical in agent execution
                error_type=template["type"],
                error_category=f"{template['type']}_error",
                error_description=template["desc"],
                failure_context=f"Agent operating in {domain} domain encountered {template['desc'].lower()}",
                recovery_attempted=(i % 3 == 0),  # ~33% have recovery attempts
                severity_score=template["severity"] + (i * 0.01) % 0.3,  # Add variation
                domain=domain,
                tools_involved=[f"tool_{i % 5}", f"tool_{(i+1) % 5}"]
            ))
        
        return patterns
    
    async def _is_cache_valid(self, cache_file: Path) -> bool:
        """Check if cache file exists and is within TTL."""
        if not cache_file.exists():
            return False
        
        file_age = datetime.now() - datetime.fromtimestamp(cache_file.stat().st_mtime)
        return file_age < self.cache_ttl
    
    async def _load_from_cache(self, cache_file: Path) -> List[TrailPattern]:
        """Load patterns from cache file."""
        with open(cache_file, 'r') as f:
            data = json.load(f)
        
        patterns = []
        for item in data:
            patterns.append(TrailPattern(**item))
        
        return patterns
    
    async def _save_to_cache(self, cache_file: Path, patterns: List[TrailPattern]) -> None:
        """Save patterns to cache file."""
        data = []
        for pattern in patterns:
            data.append({
                "id": pattern.id,
                "trace_id": pattern.trace_id,
                "step_number": pattern.step_number,
                "error_type": pattern.error_type,
                "error_category": pattern.error_category,
                "error_description": pattern.error_description,
                "failure_context": pattern.failure_context,
                "recovery_attempted": pattern.recovery_attempted,
                "severity_score": pattern.severity_score,
                "domain": pattern.domain,
                "tools_involved": pattern.tools_involved
            })
        
        with open(cache_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def _get_pattern_hash(self, pattern: FailurePattern) -> str:
        """Generate hash for pattern deduplication."""
        content = f"{pattern.title}_{pattern.description}_{pattern.category}"
        return hashlib.md5(content.encode()).hexdigest()