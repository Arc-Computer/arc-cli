"""
Database utilities for the Arc CLI project.
"""

import uuid
from typing import Any, Dict, Optional
import logging

logger = logging.getLogger(__name__)


def convert_row_to_dict(row) -> dict[str, Any]:
    """
    Convert a database row to a dictionary, handling UUID conversion.
    
    AsyncPG returns UUID fields as special UUID objects that need to be
    converted to strings for JSON serialization.
    
    Args:
        row: Database row object with _mapping attribute
        
    Returns:
        Dictionary with UUID fields converted to strings
    """
    data = dict(row._mapping)
    
    # Convert any UUID fields to strings
    for key, value in data.items():
        if isinstance(value, uuid.UUID):
            data[key] = str(value)
    
    return data


def normalize_reliability_score(score_data: Any) -> float:
    """
    Normalize reliability scores to 0-1 scale for consistent database storage.
    
    Handles various input formats:
    - Dict with overall_score (0-100 scale) -> 0-1 scale  
    - Float/int (0-100 scale) -> 0-1 scale
    - Float/int (0-1 scale) -> unchanged
    - None/invalid -> 0.0
    
    Args:
        score_data: Reliability score in various formats
        
    Returns:
        Float in 0-1 scale for database storage
    """
    if score_data is None:
        return 0.0
    
    try:
        if isinstance(score_data, dict):
            # Extract overall_score from reliability score dict
            overall_score = score_data.get("overall_score", 0.0)
            if isinstance(overall_score, (int, float)):
                # Convert from 0-100 to 0-1 if needed
                return overall_score / 100.0 if overall_score > 1.0 else overall_score
            else:
                logger.warning(f"Invalid overall_score type: {type(overall_score)}")
                return 0.0
                
        elif isinstance(score_data, (int, float)):
            # Convert from 0-100 to 0-1 if needed
            return float(score_data) / 100.0 if score_data > 1.0 else float(score_data)
            
        else:
            logger.warning(f"Unknown reliability score format: {type(score_data)}")
            return 0.0
            
    except Exception as e:
        logger.error(f"Error normalizing reliability score {score_data}: {e}")
        return 0.0


def normalize_modal_result(modal_result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize a Modal result for consistent database storage.
    
    This function ensures reliability scores are in 0-1 scale and handles
    various Modal response formats consistently.
    
    Args:
        modal_result: Raw result from Modal execution
        
    Returns:
        Normalized result with 0-1 scale reliability scores
    """
    if not isinstance(modal_result, dict):
        return modal_result
    
    # Create a copy to avoid modifying the original
    normalized_result = modal_result.copy()
    
    # Normalize reliability_score if present
    if "reliability_score" in normalized_result:
        original_score = normalized_result["reliability_score"]
        normalized_score = normalize_reliability_score(original_score)
        
        # Preserve the original structure but with normalized overall_score
        if isinstance(original_score, dict):
            normalized_result["reliability_score"] = original_score.copy()
            normalized_result["reliability_score"]["overall_score"] = normalized_score
        else:
            normalized_result["reliability_score"] = normalized_score
    
    return normalized_result