"""Shared request/response models for the Arc API (scaffold)."""

from pydantic import BaseModel, Field

__all__: list[str] = ["HealthResponse"]


class HealthResponse(BaseModel):
    """Schema for the /health endpoint response."""

    status: str = Field(..., example="ok")
