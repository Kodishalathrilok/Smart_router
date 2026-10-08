"""The Vercel entrypoint must come up as a mock-mode demo whatever the environment says."""

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

PROBE = textwrap.dedent(
    """
    import json
    from fastapi.testclient import TestClient
    import asgi

    with TestClient(asgi.app, base_url="https://demo.vercel.app") as client:
        print(json.dumps(client.get("/health").json()))
    """
)


def test_vercel_entrypoint_is_forced_to_mock_mode(tmp_path):
    # A subprocess, because importing asgi rewrites os.environ for the whole process.
    hostile = {**os.environ, "MOCK_MODE": "false", "VERIFY_EXECUTE_CODE": "true", "GROQ_API_KEY": "not-a-real-key", "LOG_DIR": str(tmp_path)}
    done = subprocess.run([sys.executable, "-c", PROBE], cwd=ROOT, env=hostile, capture_output=True, text=True, timeout=60)
    assert done.returncode == 0, done.stderr
    assert json.loads(done.stdout.strip().splitlines()[-1]) == {"status": "ok", "mock_mode": True, "log_write_failures": 0}
