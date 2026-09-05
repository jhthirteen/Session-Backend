"""API contract tests. Health + validation are offline; POST is live-gated.

Run offline:  .venv/bin/python -m pytest tests/data_tooling/test_api_contract.py -q
Run live POST: RUN_LIVE_AGENT=1 (also needs GROQ_API_KEY for the query to succeed)
"""
import os

import pytest

RUN_AGENT = os.environ.get("RUN_LIVE_AGENT") == "1" and bool(
    os.environ.get("GROQ_API_KEY")
)
needs_agent = pytest.mark.skipif(
    not RUN_AGENT, reason="live POST needs RUN_LIVE_AGENT=1 + GROQ_API_KEY"
)


@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient

    from src.data_tooling.api import app

    return TestClient(app)


def test_health_offline(client):
    assert client.get("/api/nlp/health").json() == {"status": "ok"}


def test_empty_query_rejected_offline(client):
    assert client.post("/api/nlp/query", json={"query": ""}).status_code == 422
    assert client.post("/api/nlp/query", json={}).status_code == 422


@needs_agent
def test_live_query_contract(client):
    body = client.post(
        "/api/nlp/query",
        json={"query": "how many points did jalen brunson average last year"},
    ).json()
    assert set(body) >= {"answer_text", "spec", "data", "viz_hint", "debug"}
    assert body["spec"]["intent"] == "player_season_avg"
    assert body["viz_hint"]["type"] == "single_stat"
    assert len(body["data"]) == 1
