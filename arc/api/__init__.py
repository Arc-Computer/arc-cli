"""API server scaffold for Arc.

This minimal implementation allows the `arc-server` console script to start
an ad-hoc development server. Replace with a full ASGI application once the
business logic is implemented.
"""

from __future__ import annotations

import uvicorn
from fastapi import FastAPI

__all__: list[str] = ["app", "run"]

app = FastAPI(title="Arc API", description="Scaffold API", version="0.1.0")


@app.get("/health", tags=["system"])
def health() -> dict[str, str]:  # noqa: D401
    """Simple health-check endpoint."""

    return {"status": "ok"}


def run() -> None:  # pragma: no cover
    """Launch an in-process Uvicorn development server."""

    uvicorn.run("arc.api:app", host="0.0.0.0", port=8000, reload=True)  # type: ignore[arg-type]
