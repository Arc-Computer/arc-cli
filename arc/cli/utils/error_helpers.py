"""Error categorization and handling utilities."""

from typing import Optional


def categorize_error(failure_reason: Optional[str]) -> Optional[str]:
    """Categorize error based on failure reason.
    
    Args:
        failure_reason: The failure reason text
        
    Returns:
        Error category or None
    """
    if not failure_reason:
        return None
        
    failure_lower = failure_reason.lower()
    if "currency" in failure_lower:
        return "currency_assumption"
    elif "timeout" in failure_lower:
        return "timeout"
    elif "tool" in failure_lower:
        return "tool_error"
    elif "api" in failure_lower:
        return "api_error"
    else:
        return "other"