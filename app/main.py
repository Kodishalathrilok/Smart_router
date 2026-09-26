"""FastAPI app factory and entry point: python -m app.main"""

import asyncio
import logging
import sys
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from app.api import routes_models, routes_query, routes_stats
from app.core.bootstrap import build_runtime
from app.core.config import PROJECT_ROOT, ConfigError, Settings
from app.providers.base import LLMProvider

log = logging.getLogger("smartrouter")


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

    @app.exception_handler(Exception)
    async def unhandled(request: Request, exc: Exception) -> JSONResponse:
        log.exception("unhandled error on %s", request.url.path)
        return JSONResponse(status_code=500, content={"error": type(exc).__name__, "detail": settings.redact(str(exc))})

    @app.get("/health")
    async def health() -> dict:
        return {"status": "ok", "mock_mode": settings.mock_mode}

    for module in (routes_query, routes_stats, routes_models):
        app.include_router(module.router)
    app.mount("/", StaticFiles(directory=PROJECT_ROOT / "frontend", html=True), name="frontend")
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
    uvicorn.run(create_app(settings), host=settings.host, port=settings.port)
    return 0


if __name__ == "__main__":
    sys.exit(main())
