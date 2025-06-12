"""Database utility functions."""

import uuid
from typing import Any


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