"""LLM metrics fallback — stubbed client, no network/key needed.

Contract: deterministic match first (zero LLM cost), model interprets only
blank phrasing, garbage/exceptions degrade to the PTS default, results cached.

Run: .venv/bin/python -m pytest tests/data_tooling/test_metric_fallback.py -q
"""
import json

import pytest

from src.data_tooling import agent


class _Msg:
    def __init__(self, content):
        self.content = content
        self.tool_calls = []


class _Choice:
    def __init__(self, content):
        self.message = _Msg(content)


class _Resp:
    def __init__(self, content):
        self.choices = [_Choice(content)]


class _StubClient:
    """Mimics Groq client's chat.completions.create shape."""

    def __init__(self, payload=None, explode=False):
        self.payload = payload if payload is not None else {"metrics": ["BLK"]}
        self.explode = explode
        self.calls = 0
        self.chat = self

    @property
    def completions(self):
        return self

    def create(self, **kwargs):
        self.calls += 1
        if self.explode:
            raise RuntimeError("network down")
        return _Resp(json.dumps(self.payload))


@pytest.fixture(autouse=True)
def clear_cache():
    from src.data_tooling import resolver
    resolver.LLM_RESOLVED_METRICS.clear()
    yield
    resolver.LLM_RESOLVED_METRICS.clear()


def test_explicit_match_never_calls_llm():
    client = _StubClient()
    assert agent.resolve_metrics_with_fallback(
        "how many blocks did he average", client, "m") == ["BLK"]
    assert client.calls == 0


def test_blank_phrasing_uses_llm():
    client = _StubClient({"metrics": ["BLK"]})
    assert agent.resolve_metrics_with_fallback(
        "who protects the rim best", client, "m") == ["BLK"]
    assert client.calls == 1


def test_llm_result_cached():
    client = _StubClient({"metrics": ["STL"]})
    q = "who is the most disruptive defender"
    assert agent.resolve_metrics_with_fallback(q, client, "m") == ["STL"]
    assert agent.resolve_metrics_with_fallback(q, client, "m") == ["STL"]
    assert client.calls == 1


def test_invalid_key_falls_back_to_default():
    client = _StubClient({"metrics": ["EFF"]})
    assert agent.resolve_metrics_with_fallback(
        "who is the most efficient", client, "m") == ["PTS"]


def test_empty_reply_falls_back_to_default():
    client = _StubClient({"metrics": []})
    assert agent.resolve_metrics_with_fallback(
        "tell me about the lakers", client, "m") == ["PTS"]


def test_client_error_falls_back_to_default():
    client = _StubClient(explode=True)
    assert agent.resolve_metrics_with_fallback(
        "some novel stat phrasing here", client, "m") == ["PTS"]


def test_no_client_returns_default():
    assert agent.resolve_metrics_with_fallback(
        "some novel stat phrasing here") == ["PTS"]


def test_hints_carry_override():
    import datetime
    content = agent._build_user_content(
        "tell me about the lakers", datetime.date(2026, 9, 7),
        metrics_hint=["STL"],
    )
    assert "metrics=['STL']" in content


# --- fallback propagates to viz + synthesis --------------------------------
def test_llm_resolved_stat_drives_team_viz():
    from src.data_tooling import resolver
    q = "which warriors protect the rim best in 2024-25"  # no synonym hit
    assert resolver.explicit_metrics(q) == []
    resolver.LLM_RESOLVED_METRICS[q] = ["BLK"]
    row = {"TEAM_NAME": "Golden State Warriors", "SEASON": "2024-25",
           "GP": 82, "W": 48, "L": 34, "W_PCT": 0.585, "BLK": 391}
    data = [dict(row)]
    spec = agent._infer_spec(
        q, ["get_team_stats"],
        [{"team_name": "Golden State Warriors", "season": "2024-25"}],
        data, ["BLK"], "2024-25", 10, "PerGame",
    )
    viz = resolver.choose_viz_hint(spec)
    assert viz.type == "single_stat" and viz.y_keys == ["BLK"]
    text = agent._synthesize_answer("team_stats", [dict(row)], q, ["BLK"])
    assert "391 BLK" in text


def test_unmappable_still_records_card():
    from src.data_tooling import resolver
    q = "how did the celtics do last season"
    resolver.LLM_RESOLVED_METRICS[q] = []  # model could not map it
    row = {"TEAM_NAME": "Boston Celtics", "SEASON": "2025-26",
           "W": 56, "L": 26, "W_PCT": 0.683}
    data = [dict(row)]
    spec = agent._infer_spec(
        q, ["get_team_stats"],
        [{"team_name": "Boston Celtics", "season": "2025-26"}],
        data, ["PTS"], "2025-26", 10, "PerGame",
    )
    assert resolver.choose_viz_hint(spec).type == "team_stat_card"
