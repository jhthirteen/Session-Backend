"""Offline tests for league-leader queries — no network/Groq key needed.

Run: .venv/bin/python -m pytest tests/data_tooling/test_leaders_offline.py -q
"""
import datetime

import pytest

from src.data_tooling import agent
from src.data_tooling import resolver as R
from src.data_tooling.models import QueryResponse, QuerySpec
from src.data_tooling.nba_tools import TOOL_FUNCS

TODAY = datetime.date(2026, 9, 7)


def _leader_rows(n=5, metric="AST", season="2024-25"):
    names = ["Trae Young", "Nikola Jokic", "Tyrese Haliburton", "Cade Cunningham", "Luka Doncic",
             "Shai Gilgeous-Alexander", "Jalen Brunson", "Anthony Edwards", "Kevin Durant", "LeBron James"]
    vals = [11.6, 10.2, 9.2, 9.1, 8.5, 8.0, 7.5, 7.0, 6.5, 6.0]
    return [
        {"RANK": i + 1, "PLAYER_NAME": names[i], "TEAM_ABBREVIATION": "ATL",
         "SEASON": season, "GP": 70, metric: vals[i], "PER_MODE": "PerGame"}
        for i in range(n)
    ]


# --- detector --------------------------------------------------------------
def test_detect_led_assists():
    assert R.detect_leaders_intent("Who led the NBA in assists in the 2024-25 season?")


def test_detect_top_scorers():
    assert R.detect_leaders_intent("Who were the top 10 scorers in the NBA in 2024-25?")


def test_no_leaders_for_single_player():
    assert not R.detect_leaders_intent("how many points did brunson average")
    assert not R.detect_leaders_intent("compare tatum vs brown")


def test_no_leaders_for_trend():
    assert not R.detect_leaders_intent("brunson year by year")


# --- top_n -----------------------------------------------------------------
def test_top_10_extracted():
    assert R.extract_top_n("top 10 scorers") == 10


def test_bare_led_defaults_five():
    assert R.extract_top_n("who led the NBA in assists") == 5


def test_top_n_clamped():
    assert R.extract_top_n("top 99 scorers") == 25


# --- metrics ---------------------------------------------------------------
def test_scorers_is_pts():
    assert R.resolve_metrics("Who were the top 10 scorers in the NBA in 2024-25?") == ["PTS"]


def test_assists_is_ast():
    assert R.resolve_metrics("Who led the NBA in assists in the 2024-25 season?") == ["AST"]


# --- spec routing ----------------------------------------------------------
def test_infer_league_leaders_top10():
    data = _leader_rows(10, "PTS")
    q = "Who were the top 10 scorers in the NBA in 2024-25?"
    spec = agent._infer_spec(
        q, ["get_league_leaders"],
        [{"stat_category": "PTS", "season": "2024-25", "top_n": 10}],
        data, ["PTS"], "2024-25", 10, "PerGame",
    )
    assert spec.intent == "league_leaders"
    assert spec.top_n == 10
    assert spec.season == "2024-25"
    assert spec.players[0] == "Trae Young"
    assert len(spec.players) == 10
    viz = R.choose_viz_hint(spec)
    assert viz.type == "leaderboard"
    assert viz.x_key == "PLAYER_NAME"
    assert viz.y_keys == ["PTS"]


def test_infer_league_leaders_led_defaults_top5():
    data = _leader_rows(5, "AST")
    q = "Who led the NBA in assists in the 2024-25 season?"
    spec = agent._infer_spec(
        q, ["get_league_leaders"],
        [{"stat_category": "AST", "season": "2024-25"}],
        data, ["AST"], "2024-25", 10, "PerGame",
    )
    assert spec.intent == "league_leaders"
    assert spec.top_n == 5  # falls back to extract_top_n default
    assert R.choose_viz_hint(spec).type == "leaderboard"


def test_guardrail_ignores_league_leaders():
    # Single-season scope is True for leaders queries, but the trend
    # guardrail must not touch non-trend intents or their data.
    data = _leader_rows(5, "AST")
    q = "Who led the NBA in assists in the 2024-25 season?"
    assert R.is_single_season_scope(q, TODAY)
    spec = agent._infer_spec(
        q, ["get_league_leaders"],
        [{"stat_category": "AST", "season": "2024-25"}],
        data, ["AST"], "2024-25", 10, "PerGame",
    )
    assert spec.intent == "league_leaders"
    assert len(data) == 5


# --- synthesis -------------------------------------------------------------
def test_synthesize_leaders():
    data = _leader_rows(5, "AST")
    text = agent._synthesize_answer("league_leaders", data, "who led in assists?", ["AST"])
    assert "Trae Young" in text and "11.6" in text


# --- contract --------------------------------------------------------------
def test_response_serializes_with_top_n():
    spec = QuerySpec(intent="league_leaders", players=["Trae Young"], season="2024-25",
                     metrics=["AST"], top_n=5)
    viz = R.choose_viz_hint(spec)
    resp = QueryResponse(answer_text="Trae Young led ...", spec=spec,
                         data=_leader_rows(5), viz_hint=viz, debug=[])
    assert resp.model_dump()["spec"]["top_n"] == 5
    assert resp.viz_hint.type == "leaderboard"


def test_tool_registered():
    assert "get_league_leaders" in TOOL_FUNCS
    schema = next(s for s in
                  __import__("src.data_tooling.nba_tools", fromlist=["GROQ_TOOL_SCHEMAS"]).GROQ_TOOL_SCHEMAS
                  if s["function"]["name"] == "get_league_leaders")
    assert set(schema["function"]["parameters"]["required"]) == {"stat_category", "season"}


# --- leaders guardrails (player mirror of the Sept 7 team screenshot) --------
def _pts_board():
    names = ["Shai Gilgeous-Alexander", "Giannis Antetokounmpo", "Nikola Jokic",
             "Jalen Brunson", "Anthony Edwards"]
    vals = [32.7, 30.4, 29.6, 26.0, 27.6]
    return [
        {"RANK": i + 1, "PLAYER_NAME": n, "TEAM_ABBREVIATION": "OKC",
         "SEASON": "2024-25", "GP": 70, "PTS": vals[i], "PER_MODE": "PerGame"}
        for i, n in enumerate(names)
    ]


def test_named_player_downgrades_to_season_avg():
    data = _pts_board()
    q = "how many points did jalen brunson average in 2024-25?"
    spec = agent._infer_spec(
        q, ["get_league_leaders"],
        [{"stat_category": "PTS", "season": "2024-25", "top_n": 10}],
        data, ["PTS"], "2024-25", 10, "PerGame",
    )
    assert spec.intent == "player_season_avg"
    assert spec.players == ["Jalen Brunson"]
    assert len(data) == 1 and data[0]["PTS"] == 26.0
    assert R.choose_viz_hint(spec).type == "single_stat"


def test_two_named_players_leave_board_alone():
    data = _pts_board()
    q = "compare jalen brunson vs anthony edwards in 2024-25"
    spec = agent._infer_spec(
        q, ["get_league_leaders"],
        [{"stat_category": "PTS", "season": "2024-25"}],
        data, ["PTS"], "2024-25", 10, "PerGame",
    )
    assert spec.intent == "league_leaders"
    assert len(data) == 5


def test_best_flavored_single_player_keeps_board():
    data = _pts_board()
    q = "Is Brunson the best scorer in the league?"
    spec = agent._infer_spec(
        q, ["get_league_leaders"],
        [{"stat_category": "PTS", "season": "2024-25"}],
        data, ["PTS"], "2024-25", 10, "PerGame",
    )
    assert spec.intent == "league_leaders"
    assert len(data) == 5


def test_true_player_leaders_untouched():
    data = _pts_board()
    q = "Who were the top 10 scorers in the NBA in 2024-25?"
    spec = agent._infer_spec(
        q, ["get_league_leaders"],
        [{"stat_category": "PTS", "season": "2024-25", "top_n": 10}],
        data, ["PTS"], "2024-25", 10, "PerGame",
    )
    assert spec.intent == "league_leaders"
    assert len(data) == 5
