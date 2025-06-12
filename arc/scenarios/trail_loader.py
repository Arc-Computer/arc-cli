"""
TRAIL Dataset Loader for real-world agent failure patterns.

Loads and preprocesses the PatronusAI/TRAIL dataset containing 841 real agent 
failure patterns for enhanced scenario generation.
"""

import asyncio
import json
import logging
from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import httpx
from pathlib import Path
import hashlib

logger = logging.getLogger(__name__)


@dataclass
class FailurePattern:
    """Represents a failure pattern from TRAIL dataset."""
    
    id: str
    error_type: str  # reasoning, execution, planning
    description: str
    context: str
    failure_mode: str
    domain: str
    complexity: str
    recovery_possible: bool
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "error_type": self.error_type,
            "description": self.description,
            "context": self.context,
            "failure_mode": self.failure_mode,
            "domain": self.domain,
            "complexity": self.complexity,
            "recovery_possible": self.recovery_possible,
            "metadata": self.metadata
        }


@dataclass
class TrailDatasetStats:
    """Statistics about the loaded TRAIL dataset."""
    
    total_patterns: int = 0
    patterns_by_type: Dict[str, int] = field(default_factory=dict)
    patterns_by_domain: Dict[str, int] = field(default_factory=dict)
    patterns_by_complexity: Dict[str, int] = field(default_factory=dict)
    load_time: float = 0.0
    cache_status: str = "uncached"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_patterns": self.total_patterns,
            "patterns_by_type": self.patterns_by_type,
            "patterns_by_domain": self.patterns_by_domain,
            "patterns_by_complexity": self.patterns_by_complexity,
            "load_time": round(self.load_time, 2),
            "cache_status": self.cache_status
        }


class TrailDatasetLoader:
    """Loads and manages the TRAIL dataset for scenario generation."""
    
    def __init__(self, cache_dir: Optional[str] = None, cache_ttl_hours: int = 24):
        """Initialize the TRAIL dataset loader.
        
        Args:
            cache_dir: Directory for caching dataset (default: ~/.arc_cache)
            cache_ttl_hours: Hours to cache dataset (default: 24)
        """
        self.cache_dir = Path(cache_dir or Path.home() / ".arc_cache")
        self.cache_dir.mkdir(exist_ok=True)
        self.cache_ttl = timedelta(hours=cache_ttl_hours)
        
        self._patterns: List[FailurePattern] = []
        self._patterns_by_type: Dict[str, List[FailurePattern]] = {}
        self._patterns_by_domain: Dict[str, List[FailurePattern]] = {}
        self._loaded = False
        self._stats = TrailDatasetStats()
        
        # TRAIL dataset configuration
        self.dataset_url = "https://huggingface.co/datasets/PatronusAI/TRAIL/resolve/main/data.jsonl"
        self.cache_file = self.cache_dir / "trail_patterns.json"
        self.metadata_file = self.cache_dir / "trail_metadata.json"
    
    async def load_patterns(self, force_refresh: bool = False) -> List[FailurePattern]:
        """Load TRAIL patterns with caching.
        
        Args:
            force_refresh: Force download from source
            
        Returns:
            List of failure patterns
        """
        start_time = datetime.now()
        
        # Check cache first
        if not force_refresh and await self._load_from_cache():
            self._stats.cache_status = "hit"
            logger.info(f"Loaded {len(self._patterns)} TRAIL patterns from cache")
        else:
            # Download from source
            try:
                await self._download_and_cache_patterns()
                self._stats.cache_status = "miss"
                logger.info(f"Downloaded and cached {len(self._patterns)} TRAIL patterns")
            except Exception as e:
                logger.warning(f"Failed to download TRAIL dataset: {e}")
                # Try cache as fallback
                if await self._load_from_cache():
                    self._stats.cache_status = "fallback"
                    logger.info("Using cached TRAIL patterns as fallback")
                else:
                    # Use synthetic patterns as last resort
                    self._create_synthetic_patterns()
                    self._stats.cache_status = "synthetic"
                    logger.warning("Using synthetic patterns - TRAIL dataset unavailable")
        
        self._build_indices()
        self._compute_stats()
        self._stats.load_time = (datetime.now() - start_time).total_seconds()
        self._loaded = True
        
        return self._patterns
    
    async def _load_from_cache(self) -> bool:
        """Load patterns from cache if valid."""
        if not self.cache_file.exists() or not self.metadata_file.exists():
            return False
        
        # Check cache age
        cache_age = datetime.now() - datetime.fromtimestamp(self.cache_file.stat().st_mtime)
        if cache_age > self.cache_ttl:
            return False
        
        try:
            # Load patterns
            with open(self.cache_file, 'r') as f:
                patterns_data = json.load(f)
            
            self._patterns = [
                FailurePattern(**pattern_data) 
                for pattern_data in patterns_data
            ]
            
            return len(self._patterns) > 0
            
        except Exception as e:
            logger.warning(f"Failed to load cache: {e}")
            return False
    
    async def _download_and_cache_patterns(self) -> None:
        """Download TRAIL dataset and cache processed patterns."""
        async with httpx.AsyncClient(timeout=30.0) as client:
            try:
                response = await client.get(self.dataset_url)
                response.raise_for_status()
                
                # Process JSONL data
                patterns = []
                for line in response.text.strip().split('\n'):
                    if line.strip():
                        try:
                            item = json.loads(line)
                            pattern = self._convert_trail_item_to_pattern(item)
                            if pattern:
                                patterns.append(pattern)
                        except json.JSONDecodeError:
                            continue
                
                if not patterns:
                    raise ValueError("No valid patterns found in TRAIL dataset")
                
                self._patterns = patterns
                
                # Cache the processed patterns
                await self._save_to_cache()
                
            except httpx.RequestError as e:
                raise Exception(f"Failed to download TRAIL dataset: {e}")
    
    def _convert_trail_item_to_pattern(self, item: Dict[str, Any]) -> Optional[FailurePattern]:
        """Convert TRAIL dataset item to FailurePattern."""
        try:
            # Extract error information from TRAIL format
            # Adapt based on actual TRAIL dataset structure
            error_info = item.get('error', {})
            task_info = item.get('task', {})
            
            # Generate unique ID
            content_hash = hashlib.md5(
                json.dumps(item, sort_keys=True).encode()
            ).hexdigest()[:8]
            
            # Determine error type based on TRAIL categorization
            error_type = self._categorize_trail_error(error_info, task_info)
            
            # Extract description and context
            description = error_info.get('description', '') or str(error_info)
            context = task_info.get('context', '') or str(task_info)
            
            # Determine domain and complexity
            domain = self._extract_domain(item)
            complexity = self._assess_complexity(item)
            
            # Determine if recovery is possible
            recovery_possible = self._assess_recovery_possibility(error_info)
            
            return FailurePattern(
                id=f"trail_{content_hash}",
                error_type=error_type,
                description=description[:500],  # Limit length
                context=context[:500],
                failure_mode=error_info.get('type', 'unknown'),
                domain=domain,
                complexity=complexity,
                recovery_possible=recovery_possible,
                metadata={
                    'source': 'TRAIL',
                    'original_item': item
                }
            )
            
        except Exception as e:
            logger.warning(f"Failed to convert TRAIL item: {e}")
            return None
    
    def _categorize_trail_error(self, error_info: Dict, task_info: Dict) -> str:
        """Categorize TRAIL error into reasoning, execution, or planning."""
        error_str = str(error_info).lower()
        task_str = str(task_info).lower()
        combined = error_str + " " + task_str
        
        # Reasoning errors (hallucinations, logic errors)
        reasoning_keywords = [
            'hallucination', 'incorrect', 'wrong', 'logic', 'reasoning',
            'misunderstanding', 'confusion', 'interpretation'
        ]
        
        # Execution errors (API failures, timeouts)
        execution_keywords = [
            'timeout', 'api', 'network', 'connection', 'service',
            'unavailable', 'error', 'exception', 'failure'
        ]
        
        # Planning errors (coordination, workflow)
        planning_keywords = [
            'coordination', 'workflow', 'dependency', 'ordering',
            'sequence', 'planning', 'deadlock', 'race'
        ]
        
        # Score each category
        reasoning_score = sum(1 for kw in reasoning_keywords if kw in combined)
        execution_score = sum(1 for kw in execution_keywords if kw in combined)
        planning_score = sum(1 for kw in planning_keywords if kw in combined)
        
        # Return highest scoring category
        if reasoning_score >= execution_score and reasoning_score >= planning_score:
            return "reasoning"
        elif execution_score >= planning_score:
            return "execution"
        else:
            return "planning"
    
    def _extract_domain(self, item: Dict[str, Any]) -> str:
        """Extract domain from TRAIL item."""
        # Check for domain indicators
        item_str = str(item).lower()
        
        domains = {
            'finance': ['finance', 'trading', 'currency', 'stock', 'investment'],
            'software': ['code', 'programming', 'software', 'development'],
            'web': ['web', 'browser', 'html', 'scraping', 'navigation'],
            'data': ['data', 'database', 'query', 'analysis'],
            'api': ['api', 'rest', 'http', 'endpoint']
        }
        
        for domain, keywords in domains.items():
            if any(kw in item_str for kw in keywords):
                return domain
        
        return "general"
    
    def _assess_complexity(self, item: Dict[str, Any]) -> str:
        """Assess complexity of TRAIL pattern."""
        # Simple heuristic based on item structure and content
        item_str = str(item)
        
        if len(item_str) > 2000:
            return "high"
        elif len(item_str) > 800:
            return "medium"
        else:
            return "low"
    
    def _assess_recovery_possibility(self, error_info: Dict) -> bool:
        """Assess if the error allows for recovery."""
        error_str = str(error_info).lower()
        
        # Unrecoverable error indicators
        unrecoverable = [
            'fatal', 'critical', 'permanent', 'corruption',
            'security', 'unauthorized', 'forbidden'
        ]
        
        # Recoverable error indicators
        recoverable = [
            'timeout', 'temporary', 'retry', 'fallback',
            'network', 'service unavailable'
        ]
        
        if any(kw in error_str for kw in unrecoverable):
            return False
        elif any(kw in error_str for kw in recoverable):
            return True
        else:
            return True  # Default to recoverable
    
    def _create_synthetic_patterns(self) -> None:
        """Create synthetic patterns as fallback when TRAIL is unavailable."""
        synthetic_patterns = [
            # Reasoning errors
            FailurePattern(
                id="synth_reasoning_001",
                error_type="reasoning",
                description="Agent misinterprets currency symbols in financial data",
                context="Processing multi-currency financial report",
                failure_mode="symbol_confusion",
                domain="finance",
                complexity="medium",
                recovery_possible=True,
                metadata={"source": "synthetic"}
            ),
            FailurePattern(
                id="synth_reasoning_002", 
                error_type="reasoning",
                description="Agent assumes USD when currency not specified",
                context="Calculating transaction amounts without currency context",
                failure_mode="default_assumption",
                domain="finance",
                complexity="low",
                recovery_possible=True,
                metadata={"source": "synthetic"}
            ),
            
            # Execution errors
            FailurePattern(
                id="synth_execution_001",
                error_type="execution",
                description="Currency conversion API timeout during peak hours",
                context="Real-time trading system requiring immediate rates",
                failure_mode="api_timeout",
                domain="finance",
                complexity="high",
                recovery_possible=True,
                metadata={"source": "synthetic"}
            ),
            FailurePattern(
                id="synth_execution_002",
                error_type="execution", 
                description="External service returns incomplete rate data",
                context="Batch processing financial transactions",
                failure_mode="incomplete_data",
                domain="finance",
                complexity="medium",
                recovery_possible=True,
                metadata={"source": "synthetic"}
            ),
            
            # Planning errors
            FailurePattern(
                id="synth_planning_001",
                error_type="planning",
                description="Agent attempts currency conversion before rate lookup",
                context="Multi-step financial analysis workflow",
                failure_mode="dependency_violation",
                domain="finance",
                complexity="medium",
                recovery_possible=True,
                metadata={"source": "synthetic"}
            ),
        ]
        
        # Extend to ~40 patterns to simulate meaningful dataset
        base_patterns = synthetic_patterns.copy()
        for i in range(8):  # Multiply base patterns
            for pattern in base_patterns:
                new_pattern = FailurePattern(
                    id=f"{pattern.id}_var_{i}",
                    error_type=pattern.error_type,
                    description=f"Variant {i+1}: {pattern.description}",
                    context=pattern.context,
                    failure_mode=pattern.failure_mode,
                    domain=pattern.domain,
                    complexity=pattern.complexity,
                    recovery_possible=pattern.recovery_possible,
                    metadata={"source": "synthetic", "variant": i+1}
                )
                synthetic_patterns.append(new_pattern)
        
        self._patterns = synthetic_patterns[:40]  # Limit to reasonable number
    
    async def _save_to_cache(self) -> None:
        """Save patterns to cache."""
        try:
            # Save patterns
            patterns_data = [pattern.to_dict() for pattern in self._patterns]
            with open(self.cache_file, 'w') as f:
                json.dump(patterns_data, f, indent=2)
            
            # Save metadata
            metadata = {
                'cache_time': datetime.now().isoformat(),
                'pattern_count': len(self._patterns),
                'source': 'TRAIL'
            }
            with open(self.metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
                
        except Exception as e:
            logger.warning(f"Failed to save cache: {e}")
    
    def _build_indices(self) -> None:
        """Build indices for fast pattern lookup."""
        self._patterns_by_type = {}
        self._patterns_by_domain = {}
        
        for pattern in self._patterns:
            # Index by error type
            if pattern.error_type not in self._patterns_by_type:
                self._patterns_by_type[pattern.error_type] = []
            self._patterns_by_type[pattern.error_type].append(pattern)
            
            # Index by domain
            if pattern.domain not in self._patterns_by_domain:
                self._patterns_by_domain[pattern.domain] = []
            self._patterns_by_domain[pattern.domain].append(pattern)
    
    def _compute_stats(self) -> None:
        """Compute dataset statistics."""
        self._stats.total_patterns = len(self._patterns)
        
        # Count by type
        for error_type, patterns in self._patterns_by_type.items():
            self._stats.patterns_by_type[error_type] = len(patterns)
        
        # Count by domain
        for domain, patterns in self._patterns_by_domain.items():
            self._stats.patterns_by_domain[domain] = len(patterns)
        
        # Count by complexity
        complexity_counts = {}
        for pattern in self._patterns:
            complexity = pattern.complexity
            complexity_counts[complexity] = complexity_counts.get(complexity, 0) + 1
        self._stats.patterns_by_complexity = complexity_counts
    
    def get_patterns_by_type(self, error_type: str) -> List[FailurePattern]:
        """Get patterns by error type."""
        return self._patterns_by_type.get(error_type, [])
    
    def get_patterns_by_domain(self, domain: str) -> List[FailurePattern]:
        """Get patterns by domain."""
        return self._patterns_by_domain.get(domain, [])
    
    def get_all_patterns(self) -> List[FailurePattern]:
        """Get all loaded patterns."""
        return self._patterns.copy()
    
    def get_stats(self) -> TrailDatasetStats:
        """Get dataset statistics."""
        return self._stats
    
    def is_loaded(self) -> bool:
        """Check if patterns are loaded."""
        return self._loaded
    
    def get_error_types(self) -> List[str]:
        """Get available error types."""
        return list(self._patterns_by_type.keys())
    
    def get_domains(self) -> List[str]:
        """Get available domains."""
        return list(self._patterns_by_domain.keys())