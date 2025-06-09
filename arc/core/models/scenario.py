"""Scenario definition model scaffold."""

from pydantic import BaseModel

__all__: list[str] = ["Scenario"]


class Scenario(BaseModel):
    """Placeholder for a test scenario."""

    id: str
    prompt: str
