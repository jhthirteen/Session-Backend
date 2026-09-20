"""Live agent tests — real Groq calls (cost + latency). Heavily gated.

Only runs with RUN_LIVE_AGENT=1 AND GROQ_API_KEY set.
Assertions target data/spec (deterministic), never exact LLM wording.
Run:  RUN_LIVE_AGENT=1 .venv/bin/python -m pytest tests/data_tooling/test_agent_live.py -q
"""
import os

import pytest

RUN_AGENT = os.environ.get("RUN_LIVE_AGENT") == "1" and bool(
    os.environ.get("GROQ_API_KEY")
)
needs_agent = pytest.mark.skipif(
    not RUN_AGENT, reason="needs RUN_LIVE_AGENT=1 + GROQ_API_KEY"
)


@pytest.fixture(scope="module")
def runner():
    from src.data_tooling.agent import run_query

    return run_query


@needs_agent
def test_career_peak_single_bulk_call(runner):
    r = runner("what is jalen brunson's highest points per game season of his career?")
    assert r.spec.intent == "player_career_trend"
    assert len(r.data) == 8
    assert [t.tool for t in r.debug] == ["get_player_career_trend"]  # no per-season loop
    peak = max(r.data, key=lambda d: d["PTS"])
    assert (peak["SEASON"], peak["PTS"]) == ("2023-24", pytest.approx(28.7))
    assert r.viz_hint.type == "trend_line"


@needs_agent
def test_improvement_marks_highlight_only(runner):
    r = runner("what was jalen brunson most improved season scoring")
    assert r.spec.intent == "player_career_trend"
    assert r.spec.highlight_season == "2022-23"
    assert "+7.7" in (r.spec.highlight_note or "")


@needs_agent
def test_single_season_regression(runner):
    r = runner("how many points did jalen brunson average last year")
    assert r.spec.intent == "player_season_avg"
    assert len(r.data) == 1
    assert r.data[0]["PTS"] == pytest.approx(26.0)


@needs_agent
def test_unknown_player_no_hallucination(runner):
    r = runner("how many ppg did mickey mouse average last year")
    assert r.spec.intent == "needs_clarification"
    assert r.data == []
