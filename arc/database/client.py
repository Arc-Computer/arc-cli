"""Async database client scaffold.

Replace with real SQLAlchemy AsyncEngine + session helpers.
"""

from __future__ import annotations

from typing import AsyncIterator

from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

__all__: list[str] = ["get_engine", "get_session"]

_engine: AsyncEngine | None = None


def get_engine(dsn: str | None = None) -> AsyncEngine:  # noqa: D401
    """Return a cached AsyncEngine (scaffold)."""

    global _engine
    if _engine is None:
        dsn = dsn or "postgresql+asyncpg://user:pass@localhost:5432/arc"
        _engine = create_async_engine(dsn, echo=False, future=True)
    return _engine


async def get_session() -> AsyncIterator[AsyncSession]:  # pragma: no cover
    """Yield an AsyncSession bound to the global engine (scaffold)."""

    engine = get_engine()
    async_session = sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)
    async with async_session() as session:
        yield session
