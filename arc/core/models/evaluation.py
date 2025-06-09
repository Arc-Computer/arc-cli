"""Evaluation-related data models (scaffold)."""

from __future__ import annotations

from pydantic import BaseModel

__all__: list[str] = ["EvaluationResult"]


class EvaluationResult(BaseModel):
    """Placeholder model for evaluation results."""

    score: float | None = None
    details: dict[str, str] | None = None
