"""
Quality Scoring for Generated Scenarios
Production version adapted from experiments/generation/scenario_quality_scorer.py
Scores scenarios based on: specific errors (2pts), edge cases (2pts), multi-tool (1pt), novelty (1pt)
"""

import json
import hashlib
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
from collections import Counter
try:
    import Levenshtein  # For novelty scoring
    HAS_LEVENSHTEIN = True
except ImportError:
    HAS_LEVENSHTEIN = False
    print("Warning: python-Levenshtein not installed. Using simple string comparison for novelty scoring.")


@dataclass
class QualityMetrics:
    """Metrics for scenario quality"""
    specific_error_score: float
    edge_case_score: float
    multi_tool_score: float
    novelty_score: float
    total_score: float
    passed_threshold: bool
    rejection_reason: Optional[str] = None


class ScenarioQualityScorer:
    """Scores scenario quality based on multiple dimensions"""
    
    def __init__(self, min_threshold: float = 3.0):
        self.min_threshold = min_threshold
        self.seen_scenarios: List[str] = []  # For novelty scoring
        
        # Edge case keywords that indicate good test scenarios
        self.edge_case_keywords = [
            "empty", "null", "none", "missing", "invalid", "malformed", "corrupted",
            "timeout", "expired", "unauthorized", "forbidden", "limit", "boundary",
            "overflow", "underflow", "negative", "zero", "extreme", "edge",
            "ambiguous", "unclear", "vague", "conflict", "circular", "recursive",
            "simultaneous", "concurrent", "race", "deadlock", "infinite", "loop",
            "special character", "unicode", "encoding", "escape", "injection",
            "non-existent", "fake", "mock", "test", "dummy", "placeholder"
        ]
        
        # Specific error indicators
        self.specific_error_indicators = [
            "will fail", "will timeout", "will return", "causes", "triggers",
            "results in", "leads to", "produces", "generates", "throws",
            "raises", "encounters", "experiences", "suffers from"
        ]
    
    def score_scenario(self, scenario: Dict[str, Any]) -> QualityMetrics:
        """Score a single scenario"""
        
        # Extract scenario components
        task_prompt = scenario.get('task_prompt', '')
        expected_tools = scenario.get('expected_tools', [])
        expected_error = scenario.get('expected_error', '') or scenario.get('potential_failure_mode', '')
        metadata = scenario.get('metadata', {})
        
        # Calculate individual scores
        specific_error_score = self._score_specific_error(expected_error)
        edge_case_score = self._score_edge_cases(task_prompt, expected_error)
        multi_tool_score = self._score_multi_tool(expected_tools)
        novelty_score = self._score_novelty(task_prompt)
        
        # Calculate total score
        total_score = (
            specific_error_score + 
            edge_case_score + 
            multi_tool_score + 
            novelty_score
        )
        
        # Check threshold
        passed_threshold = total_score >= self.min_threshold
        rejection_reason = None
        
        if not passed_threshold:
            if specific_error_score < 1:
                rejection_reason = "No specific error described"
            elif edge_case_score < 1:
                rejection_reason = "Not testing edge cases"
            elif total_score < self.min_threshold:
                rejection_reason = f"Total score {total_score} below threshold {self.min_threshold}"
        
        # Additional validation
        if not expected_error:
            passed_threshold = False
            rejection_reason = "Missing expected_error field"
        
        # Track scenario for novelty scoring
        self.seen_scenarios.append(task_prompt)
        
        return QualityMetrics(
            specific_error_score=specific_error_score,
            edge_case_score=edge_case_score,
            multi_tool_score=multi_tool_score,
            novelty_score=novelty_score,
            total_score=total_score,
            passed_threshold=passed_threshold,
            rejection_reason=rejection_reason
        )
    
    def _score_specific_error(self, expected_error: str) -> float:
        """Score how specific the error description is (0-2 points)"""
        if not expected_error:
            return 0.0
        
        error_lower = expected_error.lower()
        score = 0.0
        
        # Check for specific error indicators
        indicator_count = sum(1 for indicator in self.specific_error_indicators 
                            if indicator in error_lower)
        if indicator_count > 0:
            score += 1.0
        
        # Check for technical details
        technical_details = [
            "404", "500", "503", "timeout", "rate limit", "api",
            "json", "xml", "schema", "validation", "constraint",
            "permission", "authentication", "authorization"
        ]
        
        technical_count = sum(1 for detail in technical_details 
                            if detail in error_lower)
        if technical_count >= 2:
            score += 1.0
        elif technical_count == 1:
            score += 0.5
        
        # Length check - very short errors are not specific
        if len(expected_error) < 30:
            score = min(score, 1.0)
        
        return min(score, 2.0)
    
    def _score_edge_cases(self, task_prompt: str, expected_error: str) -> float:
        """Score if scenario tests edge cases (0-2 points)"""
        combined_text = f"{task_prompt} {expected_error}".lower()
        
        # Count edge case keywords
        keyword_count = sum(1 for keyword in self.edge_case_keywords 
                          if keyword in combined_text)
        
        if keyword_count >= 3:
            return 2.0
        elif keyword_count == 2:
            return 1.5
        elif keyword_count == 1:
            return 1.0
        else:
            return 0.0
    
    def _score_multi_tool(self, expected_tools: List[str]) -> float:
        """Score if scenario requires multiple tools (0-1 point)"""
        tool_count = len(expected_tools) if expected_tools else 0
        
        if tool_count >= 3:
            return 1.0
        elif tool_count == 2:
            return 0.5
        else:
            return 0.0
    
    def _score_novelty(self, task_prompt: str) -> float:
        """Score how novel/unique the scenario is (0-1 point)"""
        if not self.seen_scenarios:
            return 1.0  # First scenario is always novel
        
        # Calculate similarity with existing scenarios
        if HAS_LEVENSHTEIN:
            # Use Levenshtein distance for better comparison
            similarities = []
            for seen in self.seen_scenarios[:-1]:  # Exclude current
                distance = Levenshtein.distance(task_prompt, seen)
                similarity = 1 - (distance / max(len(task_prompt), len(seen)))
                similarities.append(similarity)
            
            max_similarity = max(similarities) if similarities else 0
        else:
            # Simple exact match check
            if task_prompt in self.seen_scenarios[:-1]:
                max_similarity = 1.0
            else:
                max_similarity = 0.0
        
        # Convert similarity to novelty score
        if max_similarity > 0.9:
            return 0.0  # Very similar
        elif max_similarity > 0.7:
            return 0.5  # Somewhat similar
        else:
            return 1.0  # Novel
    
    def score_batch(self, scenarios: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Score a batch of scenarios and return passed/failed lists"""
        passed_scenarios = []
        failed_scenarios = []
        
        for scenario in scenarios:
            metrics = self.score_scenario(scenario)
            
            # Add quality metrics to scenario
            scenario['quality_metrics'] = {
                'specific_error_score': metrics.specific_error_score,
                'edge_case_score': metrics.edge_case_score,
                'multi_tool_score': metrics.multi_tool_score,
                'novelty_score': metrics.novelty_score,
                'total_score': metrics.total_score,
                'passed_threshold': metrics.passed_threshold,
                'rejection_reason': metrics.rejection_reason
            }
            
            if metrics.passed_threshold:
                passed_scenarios.append(scenario)
            else:
                failed_scenarios.append(scenario)
        
        return passed_scenarios, failed_scenarios
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about scored scenarios"""
        return {
            'total_scenarios_seen': len(self.seen_scenarios),
            'unique_scenarios': len(set(self.seen_scenarios)),
            'min_threshold': self.min_threshold
        }


class ScenarioDeduplicator:
    """Removes duplicate or highly similar scenarios"""
    
    def __init__(self, similarity_threshold: float = 0.85):
        self.similarity_threshold = similarity_threshold
    
    def _calculate_hash(self, scenario: Dict[str, Any]) -> str:
        """Calculate hash for scenario deduplication"""
        # Use task prompt and expected error for hash
        content = f"{scenario.get('task_prompt', '')}-{scenario.get('expected_error', '')}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def deduplicate_batch(self, scenarios: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate scenarios from batch"""
        unique_scenarios = []
        seen_hashes = set()
        seen_prompts = []
        
        for scenario in scenarios:
            # Check exact hash match
            scenario_hash = self._calculate_hash(scenario)
            if scenario_hash in seen_hashes:
                continue
            
            # Check similarity if Levenshtein available
            task_prompt = scenario.get('task_prompt', '')
            is_duplicate = False
            
            if HAS_LEVENSHTEIN and seen_prompts:
                for seen_prompt in seen_prompts:
                    distance = Levenshtein.distance(task_prompt, seen_prompt)
                    similarity = 1 - (distance / max(len(task_prompt), len(seen_prompt)))
                    if similarity > self.similarity_threshold:
                        is_duplicate = True
                        break
            
            if not is_duplicate:
                unique_scenarios.append(scenario)
                seen_hashes.add(scenario_hash)
                seen_prompts.append(task_prompt)
        
        return unique_scenarios


# Utility functions
def score_scenarios_from_file(
    input_file: str,
    output_passed: str = "passed_scenarios.json",
    output_failed: str = "failed_scenarios.json",
    min_threshold: float = 3.0
) -> Dict[str, Any]:
    """Score scenarios from a JSON file"""
    
    # Load scenarios
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    scenarios = data.get('scenarios', [])
    
    # Initialize scorer
    scorer = ScenarioQualityScorer(min_threshold=min_threshold)
    
    # Score scenarios
    passed, failed = scorer.score_batch(scenarios)
    
    # Save results
    with open(output_passed, 'w') as f:
        json.dump({'scenarios': passed}, f, indent=2)
    
    with open(output_failed, 'w') as f:
        json.dump({'scenarios': failed}, f, indent=2)
    
    # Return statistics
    return {
        'total_scenarios': len(scenarios),
        'passed': len(passed),
        'failed': len(failed),
        'pass_rate': len(passed) / len(scenarios) if scenarios else 0,
        'statistics': scorer.get_statistics()
    }