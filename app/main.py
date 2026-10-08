"""FastAPI app factory and entry point: python -m app.main"""

import asyncio
import json
import logging
import sys
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.trustedhost import TrustedHostMiddleware

from app.api import routes_models, routes_query, routes_stats
from app.core.bootstrap import build_runtime
from app.core.config import PROJECT_ROOT, ConfigError, Settings
from app.providers.base import LLMProvider

log = logging.getLogger("smartrouter")


class BodySizeLimit:
    """Reject request bodies above `max_bytes` while they stream in (Content-Length or chunked),
    instead of reading and JSON-parsing an arbitrarily large body before the query-length check."""

    def __init__(self, app, max_bytes: int) -> None:
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        declared = dict(scope.get("headers") or []).get(b"content-length")
        if declared is not None and (not declared.isdigit() or int(declared) > self.max_bytes):
            return await _too_large(send, self.max_bytes)
        seen, rejected = 0, False

        async def limited_receive():
            nonlocal seen, rejected
            if rejected:
                return {"type": "http.disconnect"}
            message = await receive()
            if message["type"] == "http.request":
                seen += len(message.get("body", b""))
                if seen > self.max_bytes:  # chunked body: answer 413 now, the app sees a disconnect
                    rejected = True
                    await _too_large(send, self.max_bytes)
                    return {"type": "http.disconnect"}
            return message

        async def guarded_send(message):
            if not rejected:  # after our 413, drop whatever the app tries to send
                await send(message)

        await self.app(scope, limited_receive, guarded_send)


async def _too_large(send, limit: int) -> None:
    body = json.dumps({"detail": f"request body exceeds {limit} bytes"}).encode()
    await send({"type": "http.response.start", "status": 413, "headers": [(b"content-type", b"application/json")]})
    await send({"type": "http.response.body", "body": body})


def create_app(settings: Settings | None = None, *, providers: dict[str, LLMProvider] | None = None) -> FastAPI:
    settings = settings or Settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        try:
            app.state.runtime = await build_runtime(settings, providers=providers)
        except ConfigError as exc:
            log.error("Configuration error: %s", exc)
            raise
        log.info("SmartRouter ready (mock_mode=%s)", settings.mock_mode)
        yield
        await app.state.runtime.aclose()

    app = FastAPI(title="SmartRouter", version="1.0.0", lifespan=lifespan)

    @app.exception_handler(RequestValidationError)
    async def invalid(request: Request, exc: RequestValidationError) -> JSONResponse:
        # Default 422 body echoes the offending input, which can be huge or un-encodable (lone surrogates).
        return JSONResponse(status_code=422, content={"detail": [{"loc": e["loc"], "msg": e["msg"], "type": e["type"]} for e in exc.errors()]})

    @app.exception_handler(Exception)
    async def unhandled(request: Request, exc: Exception) -> JSONResponse:
        log.exception("unhandled error on %s", request.url.path)
        # Detail stays in the server log; exception text can carry provider responses or file paths.
        return JSONResponse(status_code=500, content={"error": "internal_error", "detail": "internal server error; see server log"})

    @app.middleware("http")
    async def revalidate_dashboard(request: Request, call_next):
        # Dashboard files change with the code; make browsers revalidate (cheap 304 via ETag) instead of
        # serving a stale stylesheet or script after an update.
        response = await call_next(request)
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("X-Frame-Options", "DENY")
        response.headers.setdefault("Referrer-Policy", "no-referrer")
        if not request.url.path.startswith("/api"):
            response.headers.setdefault("Cache-Control", "no-cache")
        return response

    @app.get("/health")
    async def health(request: Request) -> dict:
        runtime = request.app.state.runtime
        failures = runtime.events.write_failures + runtime.queries.write_failures
        return {"status": "degraded" if failures else "ok", "mock_mode": settings.mock_mode, "log_write_failures": failures}

    for module in (routes_query, routes_stats, routes_models):
        app.include_router(module.router)
    app.mount("/", StaticFiles(directory=PROJECT_ROOT / "frontend", html=True), name="frontend")
    # JSON-escaped text can take up to 6 bytes per character (\uXXXX), plus room for the envelope.
    app.add_middleware(BodySizeLimit, max_bytes=settings.max_query_chars * 6 + 4096)
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=settings.allowed_host_list)
    return app


async def _preflight(settings: Settings) -> None:
    runtime = await build_runtime(settings)
    await runtime.aclose()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    settings = Settings()
    try:  # validate config (keys, model IDs vs provider lists) before uvicorn, so errors print as one clear line
        asyncio.run(_preflight(settings))
    except ConfigError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2
    if not settings.is_loopback_host:
        log.warning("HOST=%s exposes an API with NO authentication, rate limiting or TLS; every visitor can spend your provider quota. Add the hostname clients use to ALLOWED_HOSTS", settings.host)
    uvicorn.run(create_app(settings), host=settings.host, port=settings.port)
    return 0


if __name__ == "__main__":
    sys.exit(main())
