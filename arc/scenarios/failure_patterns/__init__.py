"""
Failure patterns library for scenario generation.
Based on TRAIL taxonomy for comprehensive error coverage.
"""

from typing import Dict, List, Any
from pathlib import Path
import json

# Pattern categories based on TRAIL taxonomy
PATTERN_CATEGORIES = {
    "reasoning": ["data", "logic", "calculation"],
    "execution": ["infrastructure", "auth", "security"],
    "planning": ["navigation", "temporal", "multi_agent"]
}

# Export pattern utilities
from .library import PatternLibrary, FailurePattern

__all__ = ["PatternLibrary", "FailurePattern", "PATTERN_CATEGORIES"]