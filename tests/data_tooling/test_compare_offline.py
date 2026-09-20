"""Offline tests for multi-thing comparisons — no network, no Groq key.

Covers: entity cap, compare fan-out tools (via monkeypatched single-entity
fetchers), _infer_spec routing incl. the legacy per-player-trend path,
choose_viz_hint series_key, and detect_comparison_intent.

Run:  .venv/bin/python -m pytest tests/data_tooling/test_compare_offline.py -q
"""

import pytest

from src.data_tooling import nba_tools as tools
from src.data_tooling import resolver as R
from src.data_tooling.agent import _infer_spec, _synthesize_answer
from src.data_tooling.models import QuerySpec


def _trend(player, seasons):
    return [
        {"PLAYER_NAME": player, "SEASON": s, "PTS": 20.0 + i, "PER_MODE": "PerGame"}
        for i, s in enumerate(seasons)
    ]


# --- entity cap -----------------------------------------------------------
def test_cap_rejects_five_players():
    with pytest.raises(Exception, match="Too many players"):
        tools.compare_player_career_trends(["a", "b", "c", "d", "e"])


def test_cap_rejects_five_teams():
    with pytest.raises(Exception, match="Too many teams"):
        tools.compare_teams(["a", "b", "c", "d", "e"], season="2024-25")


# --- fan-out --------------------------------------------------------------
def test_compare_player_career_trends_tags_rows(monkeypatch):
    monkeypatch.setattr(
        tools,
        "get_player_career_trend",
        lambda name, **kw: _trend(name, ["2023-24", "2024-25"]),
    )
    rows = tools.compare_player_career_trends(["Jayson Tatum", "Jaylen Brown"])
    assert len(rows) == 4
    assert {r["PLAYER_NAME"] for r in rows} == {"Jayson Tatum", "Jaylen Brown"}


def test_compare_trends_partial_failure_kept(monkeypatch):
    def fake(name, **kw):
        if name == "Bad Name":
            raise tools.ToolError("Unknown player 'Bad Name'.")
        return _trend(name, ["2024-25"])

    monkeypatch.setattr(tools, "get_player_career_trend", fake)
    rows = tools.compare_player_career_trends(["Jayson Tatum", "Bad Name"])
    assert len(rows) == 2
    assert rows[1] == {"PLAYER_NAME": "Bad Name", "error": "Unknown player 'Bad Name'."}


def test_compare_teams_trims_metrics(monkeypatch):
    monkeypatch.setattr(
        tools,
        "get_team_stats",
        lambda name, season: {
            "TEAM_NAME": name,
            "SEASON": season,
            "W": 56,
            "L": 26,
            "W_PCT": 0.683,
            "PTS": 120.0,
        },
    )
    rows = tools.compare_teams(
        ["Boston Celtics", "Los Angeles Lakers"], season="2024-25"
    )
    assert [r["TEAM_NAME"] for r in rows] == [
        "Boston Celtics",
        "Los Angeles Lakers",
    ]
    assert rows[0]["W"] == 56
    assert "PTS" not in rows[0]  # default metrics are W/L/W_PCT only


def test_compare_team_histories_tags_rows(monkeypatch):
    monkeypatch.setattr(
        tools,
        "get_team_history_trend",
        lambda name, **kw: [
            {"TEAM_NAME": name, "SEASON": "2024-25", "W": 56, "L": 26}
        ],
    )
    rows = tools.compare_team_histories(["Boston Celtics", "Los Angeles Lakers"])
    assert len(rows) == 2
    assert {r["TEAM_NAME"] for r in rows} == {
        "Boston Celtics",
        "Los Angeles Lakers",
    }


# --- spec inference -------------------------------------------------------
def test_infer_compare_trends_from_bulk_tool():
    data = _trend("Jayson Tatum", ["2023-24", "2024-25"]) + _trend(
        "Jaylen Brown", ["2023-24", "2024-25"]
    )
    spec = _infer_spec(
        "tatum vs brown careers",
        ["compare_player_career_trends"],
        [{"players": ["Jayson Tatum", "Jaylen Brown"], "per_mode": "PerGame"}],
        data,
        ["PTS"],
        "2024-25",
        10,
        "PerGame",
    )
    assert spec.intent == "compare_trends"
    assert spec.players == ["Jayson Tatum", "Jaylen Brown"]
    assert spec.seasons == ["2023-24", "2024-25"]
    assert spec.season == "2024-25"


def test_infer_compare_trends_from_legacy_per_player_calls():
    # Model called single-entity trend twice — must NOT collapse to one line.
    data = _trend("Jayson Tatum", ["2023-24", "2024-25"]) + _trend(
        "Jaylen Brown", ["2023-24", "2024-25"]
    )
    spec = _infer_spec(
        "tatum vs brown careers",
        ["get_player_career_trend", "get_player_career_trend"],
        [{"player_name": "Jayson Tatum"}, {"player_name": "Jaylen Brown"}],
        data,
        ["PTS"],
        "2024-25",
        10,
        "PerGame",
    )
    assert spec.intent == "compare_trends"


def test_infer_compare_teams():
    spec = _infer_spec(
        "celtics vs lakers record last season",
        ["compare_teams"],
        [{"teams": ["Boston Celtics", "Los Angeles Lakers"], "season": "2024-25"}],
        [{"TEAM_NAME": "Boston Celtics", "W": 56}],
        ["W"],
        "2024-25",
        10,
        "PerGame",
    )
    assert spec.intent == "compare_teams"
    assert spec.teams == ["Boston Celtics", "Los Angeles Lakers"]


def test_single_trend_still_singular():
    spec = _infer_spec(
        "brunson career",
        ["get_player_career_trend"],
        [{"player_name": "Jalen Brunson"}],
        _trend("Jalen Brunson", ["2023-24"]),
        ["PTS"],
        "2024-25",
        10,
        "PerGame",
    )
    assert spec.intent == "player_career_trend"


# --- viz hints ------------------------------------------------------------
def test_multi_trend_hint_has_series_key():
    viz = R.choose_viz_hint(
        QuerySpec(
            intent="compare_trends",
            players=["Jayson Tatum", "Jaylen Brown"],
            metrics=["PTS"],
            seasons=["2023-24", "2024-25"],
        )
    )
    assert viz.type == "multi_trend"
    assert viz.x_key == "SEASON"
    assert viz.series_key == "PLAYER_NAME"


def test_team_compare_hint():
    viz = R.choose_viz_hint(
        QuerySpec(
            intent="compare_teams",
            teams=["Boston Celtics", "Los Angeles Lakers"],
            season="2024-25",
            metrics=["W"],
        )
    )
    assert viz.type == "comparison_bars"
    assert viz.x_key == "TEAM_NAME"
    assert viz.series_key == "TEAM_NAME"


def test_player_compare_hint_now_has_keys():
    viz = R.choose_viz_hint(
        QuerySpec(
            intent="compare_players",
            players=["A", "B"],
            season="2024-25",
            metrics=["PTS"],
        )
    )
    assert viz.x_key == "PLAYER_NAME"
    assert viz.series_key == "PLAYER_NAME"


# --- comparison detection --------------------------------------------------
def test_detect_comparison_intent():
    assert R.detect_comparison_intent("compare tatum vs brown careers")
    assert R.detect_comparison_intent("celtics versus lakers record")
    assert R.detect_comparison_intent("who is better, jokic or embiid?")
    assert not R.detect_comparison_intent("how many points did brunson average")


# --- answer synthesis ------------------------------------------------------
def test_synthesize_compare_trends():
    text = _synthesize_answer(
        "compare_trends",
        _trend("Jayson Tatum", ["2023-24", "2024-25"])
        + _trend("Jaylen Brown", ["2023-24", "2024-25"]),
        "q",
        ["PTS"],
    )
    assert "Jayson Tatum" in text and "Jaylen Brown" in text
    assert "2023-24 to 2024-25" in text
