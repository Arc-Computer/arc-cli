"""
Pure LLM-Based Recommendation System for Arc

This package provides intelligent, data-driven recommendations based on actual
simulation data using LLM analysis. No hardcoded patterns or rules.

The system works with ANY agent YAML configuration by extracting actual execution
patterns and using LLM to generate specific, actionable YAML improvements.
"""

from .engine import RecommendationEngine, RecommendationRequest, RecommendationResponse
from .generic_analyzer import GenericAgentAnalyzer, SimulationContext, GenericRecommendation
from .data_extractor import SimulationDataExtractor
from .llm_client import LLMClient, OpenRouterLLMClient, OpenAILLMClient, AnthropicLLMClient, create_llm_client

__all__ = [
    'RecommendationEngine',
    'RecommendationRequest',
    'RecommendationResponse',
    'GenericAgentAnalyzer',
    'SimulationContext',
    'GenericRecommendation',
    'SimulationDataExtractor',
    'LLMClient',
    'OpenRouterLLMClient',
    'OpenAILLMClient',
    'AnthropicLLMClient',
    'create_llm_client'
]
