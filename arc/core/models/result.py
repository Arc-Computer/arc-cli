"""Simulation outcome model scaffold."""

from pydantic import BaseModel

__all__: list[str] = ["Outcome"]


class Outcome(BaseModel):
    """Placeholder for a single scenario outcome."""

    success: bool | None = None
    error: str | None = None
