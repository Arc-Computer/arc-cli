"""Configuration package scaffold."""

from __future__ import annotations

import os
from functools import lru_cache

__all__: list[str] = ["get_settings"]


class Settings:  # noqa: D101
    app_env: str = os.getenv("ARC_ENV", "development")

    # Future: load secrets, DB URL, etc.


@lru_cache(maxsize=1)
def get_settings() -> Settings:  # noqa: D401
    """Return cached Settings instance."""

    return Settings()
