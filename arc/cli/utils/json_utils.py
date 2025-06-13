"""JSON utilities for Arc CLI."""

from datetime import datetime
from typing import Any


def json_serializer(obj: Any) -> str:
    """JSON serializer for objects not serializable by default json code.
    
    Args:
        obj: Object to serialize
        
    Returns:
        String representation of the object
        
    Raises:
        TypeError: If object type is not supported
    """
    if isinstance(obj, datetime):
        return obj.isoformat()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable") 