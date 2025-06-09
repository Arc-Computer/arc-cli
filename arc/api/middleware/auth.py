"""Authentication middleware scaffold.

Replace with real authentication/authorization logic.
"""

from __future__ import annotations

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response

__all__: list[str] = ["AuthMiddleware"]


class AuthMiddleware(BaseHTTPMiddleware):  # noqa: D101
    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:  # noqa: D401
        # TODO: Implement real authentication & context propagation
        return await call_next(request)
