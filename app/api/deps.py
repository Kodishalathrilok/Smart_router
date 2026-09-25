"""FastAPI dependencies. The runtime lives on app.state (per-app, not module-global)."""

from fastapi import Request

from app.core.bootstrap import Runtime


def get_runtime(request: Request) -> Runtime:
    return request.app.state.runtime
