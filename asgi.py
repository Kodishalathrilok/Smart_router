"""Vercel entrypoint: a public, mock-mode demo of SmartRouter (no API keys, no code execution).

Forced, not defaulted: the API has no authentication (README: Security model), so a public deployment must never
reach real providers or run model-written code. HOST=0.0.0.0 makes Settings treat the app as non-loopback, so
VERIFY_EXECUTE_CODE is refused on top of being forced off. Vercel's filesystem is read-only except /tmp.
"""

import os

os.environ.update(MOCK_MODE="true", HOST="0.0.0.0", VERIFY_EXECUTE_CODE="false", VALIDATE_MODELS="false")
os.environ.setdefault("ALLOWED_HOSTS", "*")  # public hostname; the loopback-only default would reject every request
os.environ.setdefault("LOG_DIR", "/tmp/smartrouter/logs")
os.environ.setdefault("BENCHMARK_DIR", "/tmp/smartrouter/benchmarks")

from app.main import create_app  # Settings are read when create_app() runs, so the environment above must come first

app = create_app()
