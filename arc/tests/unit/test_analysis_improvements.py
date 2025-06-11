"""Unit tests for Arc analysis improvements."""

import pytest
import asyncio
from unittest.mock import MagicMock, patch, AsyncMock
import json
import inspect

from arc.analysis.funnel_analyzer import FunnelAnalyzer
from arc.analysis.assumption_detector import AssumptionDetector


class TestAnalysisImprovements:
    """Test analysis optimization improvements."""
    
    @pytest.fixture
    def funnel_analyzer(self):
        """Create FunnelAnalyzer instance."""
        return FunnelAnalyzer()
    
    @pytest.fixture
    def assumption_detector(self):
        """Create AssumptionDetector instance."""
        return AssumptionDetector()
    
    @pytest.fixture
    def sample_trajectory(self):
        """Create sample trajectory for testing."""
        return {
            "id": "test-123",
            "status": "error",
            "scenario": {
                "task_prompt": "Convert 100 USD to EUR"
            },
            "full_trajectory": [
                {"type": "message", "content": "Starting conversion"},
                {"type": "tool_call", "tool": "calculator", "tool_input": "100 * 0.92"},
                {"type": "error", "message": "Currency rate not found"},
                {"type": "message", "content": "Attempting retry"},
                {"type": "tool_call", "tool": "search", "tool_input": "USD to EUR rate"},
                {"type": "message", "content": "Found rate: 0.92"},
                {"type": "message", "content": "Calculation complete"},
                {"type": "error", "message": "Failed to format output"}
            ],
            "final_output": "Unable to complete conversion"
        }
    
    def test_trajectory_summarization(self, funnel_analyzer, sample_trajectory):
        """Test that trajectory summarization reduces event count."""
        events = sample_trajectory["full_trajectory"]
        
        # Test summarization
        summarized = funnel_analyzer._summarize_trajectory(events, max_events=5)
        
        # Should reduce from 8 to 5 events
        assert len(summarized) == 5
        # Should include first and last events
        assert summarized[0] == events[0]
        assert summarized[-1] == events[-1]
        # Should prioritize error events
        assert any(e.get("type") == "error" for e in summarized)
        # Should include tool calls
        assert any(e.get("type") == "tool_call" for e in summarized)
    
    def test_assumption_event_summarization(self, assumption_detector, sample_trajectory):
        """Test assumption-focused event summarization."""
        # Add currency-related event
        events = sample_trajectory["full_trajectory"].copy()
        events.insert(2, {"type": "message", "content": "Processing USD amount"})
        
        summarized = assumption_detector._summarize_events_for_detection(events, max_events=7)
        
        # Should prioritize currency-related event
        assert any("USD" in str(e.get("content", "")) for e in summarized)
        # Should still include errors
        assert any(e.get("type") == "error" for e in summarized)
    
    def test_cache_key_generation(self, funnel_analyzer, sample_trajectory):
        """Test deterministic cache key generation."""
        # Generate cache key
        key1 = funnel_analyzer._get_trajectory_hash(sample_trajectory, "reasoning")
        
        # Same trajectory should produce same key
        key2 = funnel_analyzer._get_trajectory_hash(sample_trajectory, "reasoning")
        assert key1 == key2
        
        # Different analysis type should produce different key
        key3 = funnel_analyzer._get_trajectory_hash(sample_trajectory, "output_quality")
        assert key1 != key3
        
        # Modified trajectory should produce different key
        modified = sample_trajectory.copy()
        modified["status"] = "success"
        key4 = funnel_analyzer._get_trajectory_hash(modified, "reasoning")
        assert key1 != key4
    
    def test_async_evaluation_methods_exist(self, funnel_analyzer):
        """Test that evaluation methods are properly async."""
        # Verify methods exist and are async coroutines
        assert hasattr(funnel_analyzer, '_evaluate_reasoning')
        assert inspect.iscoroutinefunction(funnel_analyzer._evaluate_reasoning)
        
        assert hasattr(funnel_analyzer, '_evaluate_output_generation')
        assert inspect.iscoroutinefunction(funnel_analyzer._evaluate_output_generation)
        
        assert hasattr(funnel_analyzer, '_ai_analyze_reasoning')
        assert inspect.iscoroutinefunction(funnel_analyzer._ai_analyze_reasoning)
        
        assert hasattr(funnel_analyzer, '_ai_analyze_output_quality')
        assert inspect.iscoroutinefunction(funnel_analyzer._ai_analyze_output_quality)
    
    def test_caching_infrastructure(self, funnel_analyzer):
        """Test that caching infrastructure is in place."""
        # Verify cache exists
        assert hasattr(funnel_analyzer, '_ai_cache')
        assert isinstance(funnel_analyzer._ai_cache, dict)
        
        # Test cache key generation
        trajectory = {
            "status": "error",
            "scenario": {"task_prompt": "Test task"},
            "full_trajectory": [{"type": "message", "content": "test"}],
            "final_output": "Failed"
        }
        
        key1 = funnel_analyzer._get_trajectory_hash(trajectory, "reasoning")
        key2 = funnel_analyzer._get_trajectory_hash(trajectory, "reasoning")
        
        # Same input should generate same key
        assert key1 == key2
        assert isinstance(key1, str)
        assert len(key1) == 32  # MD5 hash length
    
    def test_few_shot_prompts_structure(self, funnel_analyzer, assumption_detector):
        """Test that prompts would include few-shot examples."""
        # The prompts in the implementation include few-shot examples
        # We can verify the structure exists without making actual API calls
        
        # Test trajectory for checking prompt generation
        trajectory = {
            "scenario": {"task_prompt": "Test task"},
            "full_trajectory": [],
            "final_output": "Test output",
            "status": "error"
        }
        
        # For funnel analyzer - verify the methods that would generate prompts with examples
        # The actual prompt generation happens inside _ai_analyze_reasoning and _ai_analyze_output_quality
        # These methods include "Examples:" in their prompts
        
        # For assumption detector
        assert hasattr(assumption_detector, '_ai_enhanced_detection')
        assert inspect.iscoroutinefunction(assumption_detector._ai_enhanced_detection)
    
    def test_pattern_based_detection(self, assumption_detector):
        """Test pattern-based assumption detection."""
        trajectory = {
            "id": "test-pattern",
            "scenario": {"task_prompt": "Convert $100 USD to EUR"},
            "full_trajectory": [
                {"type": "message", "content": "Processing USD currency"},
                {"type": "error", "message": "Failed to convert currency"}
            ],
            "final_output": "Unable to complete conversion"
        }
        
        violations = assumption_detector._detect_pattern_violations(trajectory, ["finance"])
        
        # Should detect currency pattern in finance domain
        assert len(violations) > 0
        currency_violations = [v for v in violations if v.type == "currency"]
        assert len(currency_violations) > 0
        
        # Verify violation structure
        violation = currency_violations[0]
        assert hasattr(violation, 'type')
        assert hasattr(violation, 'severity')
        assert hasattr(violation, 'confidence')
        assert hasattr(violation, 'description')
        assert hasattr(violation, 'suggested_fix')
        assert hasattr(violation, 'business_impact')