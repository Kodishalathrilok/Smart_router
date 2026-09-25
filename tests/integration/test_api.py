import json

import pytest
from fastapi.testclient import TestClient

from app.main import create_app
from app.providers.base import RateLimitError
from tests.conftest import FakeProvider, make_settings


@pytest.fixture
def client(tmp_path):
    with TestClient(create_app(make_settings(tmp_path, max_query_chars=500))) as c:
        yield c


def test_health(client):
    assert client.get("/health").json() == {"status": "ok", "mock_mode": True}


def test_query_roundtrip(client):
    r = client.post("/api/query", json={"query": "Explain binary search simply"})
    assert r.status_code == 200
    body = r.json()
    for key in ("query_id", "answer", "initial_model", "final_model", "quality_score", "escalated", "trace", "final_status"):
        assert key in body
    assert body["trace"][0]["label"] == "Query received"

    full = client.get(f"/api/query/{body['query_id']}").json()
    assert full["query_id"] == body["query_id"] and full["attempts"][0]["dimensions"]
    assert client.get("/api/query/q_missing").status_code == 404


def test_query_validation(client):
    assert client.post("/api/query", json={"query": ""}).status_code == 422
    assert client.post("/api/query", json={"query": "   "}).status_code == 422
    assert client.post("/api/query", json={"query": "x" * 501}).status_code == 413
    assert client.post("/api/query", json={}).status_code == 422


def test_stats_routes_failures_models(client):
    empty = client.get("/api/stats").json()
    assert empty["n"] == 0 and empty["cost_reduction_vs_all_strong_pct"] is None

    for q in ("Who wrote Hamlet?", "Prove step by step why sqrt(2) is irrational and justify each step rigorously."):
        client.post("/api/query", json={"query": q})
    stats = client.get("/api/stats").json()
    assert stats["n"] == 2 and "utilization" in stats and "routing_flow" in stats
    routes = client.get("/api/routes?limit=1").json()
    assert len(routes) == 1 and routes[0]["query"].startswith("Prove")  # newest first
    assert set(client.get("/api/failures").json()) == {"groups", "recent"}
    models = client.get("/api/models").json()
    assert [m["role"] for m in models["models"]] == ["cheap", "mid", "strong", "judge"]
    assert "api_key" not in json.dumps(models).lower()


def test_benchmark_not_evaluated_then_latest(client, tmp_path):
    assert client.get("/api/benchmarks/latest").json()["status"] == "not_evaluated"
    (tmp_path / "results").mkdir(exist_ok=True)
    (tmp_path / "results" / "latest.json").write_text('{"status": "complete", "run_id": "r1"}', encoding="utf-8")
    assert client.get("/api/benchmarks/latest").json()["run_id"] == "r1"


def test_provider_failure_does_not_crash_server(tmp_path):
    err = RateLimitError("429", provider="mock")
    provider = FakeProvider({"mock-cheap": err, "mock-mid": err, "mock-strong": err, "mock-judge": ""})
    with TestClient(create_app(make_settings(tmp_path), providers={"mock": provider})) as c:
        body = c.post("/api/query", json={"query": "Who wrote Hamlet?"}).json()
        assert body["final_status"] == "failed" and body["answer"] is None
        assert c.get("/api/failures").json()["groups"] == {"Rate limit": 3}
        assert c.get("/api/stats").json()["provider_errors"] == {"mock:RateLimitError": 6}  # 1 retry each
        assert c.get("/health").status_code == 200


def test_frontend_served(client):
    assert client.get("/").status_code == 200
