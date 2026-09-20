"""Offline tests for TEAM leader queries — no network/Groq key needed.

Run: .venv/bin/python -m pytest tests/data_tooling/test_team_leaders_offline.py -q
"""
import pytest

from src.data_tooling import agent
from src.data_tooling import resolver as R
from src.data_tooling.models import QuerySpec
from src.data_tooling.nba_tools import TEAM_LEADER_SPECS, TOOL_FUNCS


def _team_rows(n=5, stat="W", season="2024-25"):
    names = ["Oklahoma City Thunder", "Cleveland Cavaliers", "Boston Celtics",
             "New York Knicks", "Denver Nuggets", "Minnesota Timberwolves",
             "Los Angeles Lakers", "Golden State Warriors"]
    vals = [68, 64, 61, 51, 50, 49, 47, 46]
    return [
        {"RANK": i + 1, "TEAM_NAME": names[i], "SEASON": season,
         "GP": 82, "W": vals[i], "L": 82 - vals[i], stat: vals[i],
         "PER_MODE": "PerGame"}
        for i in range(n)
    ]


# --- vocab ---------------------------------------------------------------
def test_best_record_is_w():
    assert R.resolve_team_leader_stat("which team had the best record in 2024-25?") == "W"


def test_most_wins_is_w():
    assert R.resolve_team_leader_stat("which team won the most games") == "W"


def test_best_offense_is_pts():
    assert R.resolve_team_leader_stat("best offense in the NBA") == "PTS"


def test_best_defense_is_opp_pts():
    assert R.resolve_team_leader_stat("best defense last season") == "OPP_PTS"


def test_fewest_allowed_beats_points():
    # Longer phrase wins: not plain PTS.
    assert R.resolve_team_leader_stat("fewest points allowed") == "OPP_PTS"


def test_net_rating():
    assert R.resolve_team_leader_stat("best net rating") == "NET_RATING"


# --- detector ------------------------------------------------------------
def test_detect_most_wins():
    assert R.detect_team_leaders_intent("which team won the most games in 2024-25?")


def test_detect_best_offense():
    assert R.detect_team_leaders_intent("best offense in the NBA last season")


def test_detect_top_offenses():
    assert R.detect_team_leaders_intent("top 10 offenses in 2024-25")


def test_single_team_vetoes():
    assert not R.detect_team_leaders_intent("what was the celtics record last season")
    assert not R.detect_team_leaders_intent("celtics vs lakers record last season")


def test_single_player_not_team_leaders():
    assert not R.detect_team_leaders_intent("how many points did brunson average")


def test_player_leaders_not_team_leaders():
    assert not R.detect_team_leaders_intent("who led the NBA in assists in 2024-25?")


# --- routing -------------------------------------------------------------
def test_infer_team_leaders_wins():
    data = _team_rows(5, "W")
    q = "which team won the most games in the 2024-25 season?"
    spec = agent._infer_spec(
        q, ["get_team_leaders"],
        [{"stat": "W", "season": "2024-25", "top_n": 5}],
        data, ["PTS"], "2024-25", 10, "PerGame",
    )
    assert spec.intent == "team_leaders"
    assert spec.top_n == 5
    assert spec.metrics == ["W"]  # from tool stat arg, not player metric map
    assert spec.teams[0] == "Oklahoma City Thunder"
    viz = R.choose_viz_hint(spec)
    assert viz.type == "leaderboard"
    assert viz.x_key == "TEAM_NAME"
    assert viz.y_keys == ["W"]


def test_infer_team_leaders_defense_top10():
    data = [
        {"RANK": i + 1, "TEAM_NAME": f"Team {i}", "SEASON": "2024-25",
         "GP": 82, "W": 50, "L": 32, "OPP_PTS": 100.0 + i, "PER_MODE": "PerGame"}
        for i in range(10)
    ]
    q = "top 10 defenses by fewest points allowed in 2024-25"
    spec = agent._infer_spec(
        q, ["get_team_leaders"],
        [{"stat": "OPP_PTS", "season": "2024-25", "top_n": 10}],
        data, ["PTS"], "2024-25", 10, "PerGame",
    )
    assert spec.intent == "team_leaders"
    assert spec.top_n == 10
    assert R.choose_viz_hint(spec).x_key == "TEAM_NAME"


def test_guardrail_ignores_team_leaders():
    data = _team_rows(5, "W")
    q = "which team won the most games in the 2024-25 season?"
    spec = agent._infer_spec(
        q, ["get_team_leaders"],
        [{"stat": "W", "season": "2024-25"}],
        data, ["W"], "2024-25", 10, "PerGame",
    )
    assert spec.intent == "team_leaders"
    assert len(data) == 5


# --- synthesis / registry ------------------------------------------------
def test_synthesize_team_leaders():
    text = agent._synthesize_answer("team_leaders", _team_rows(5), "most wins?", ["W"])
    assert "Oklahoma City Thunder" in text and "68" in text


def test_specs_cover_expected_stats():
    for stat in ("W", "PTS", "OPP_PTS", "NET_RATING", "OFF_RATING", "DEF_RATING", "FG3M"):
        assert stat in TEAM_LEADER_SPECS


def test_tool_registered_with_schema():
    assert "get_team_leaders" in TOOL_FUNCS
    from src.data_tooling import nba_tools as NT
    schema = next(s for s in NT.GROQ_TOOL_SCHEMAS if s["function"]["name"] == "get_team_leaders")
    assert set(schema["function"]["parameters"]["required"]) == {"stat", "season"}


def test_bad_stat_rejected():
    with pytest.raises(Exception, match="Can't rank teams"):
        __import__("src.data_tooling.nba_tools", fromlist=["get_team_leaders"]).get_team_leaders(
            "DUNKS", "2024-25")


# --- leaders guardrails (Sept 7 GSW screenshot) ------------------------------
def _fg3m_board():
    teams = ["Boston Celtics", "Cleveland Cavaliers", "Chicago Bulls",
             "Golden State Warriors", "Minnesota Timberwolves"]
    totals = [1457, 1303, 1266, 1264, 1250]
    return [
        {"RANK": i + 1, "TEAM_NAME": t, "SEASON": "2024-25", "GP": 82,
         "W": 50, "L": 32, "FG3M": totals[i], "PER_MODE": "Totals"}
        for i, t in enumerate(teams)
    ]


def test_named_team_downgrades_to_team_stats():
    data = _fg3m_board()
    q = "How many three-pointers did the Golden State Warriors make in 2024-25?"
    spec = agent._infer_spec(
        q, ["get_team_leaders"],
        [{"stat": "FG3M", "season": "2024-25", "top_n": 10}],
        data, ["FG3M"], "2024-25", 10, "Totals",
    )
    assert spec.intent == "team_stats"
    assert spec.teams == ["Golden State Warriors"]
    assert spec.top_n is None
    assert len(data) == 1 and data[0]["FG3M"] == 1264
    viz = R.choose_viz_hint(spec)
    assert viz.type == "single_stat" and viz.y_keys == ["FG3M"]


def test_true_team_leaders_untouched():
    data = _fg3m_board()
    q = "which team won the most games in the 2024-25 season?"
    spec = agent._infer_spec(
        q, ["get_team_leaders"],
        [{"stat": "W", "season": "2024-25", "top_n": 5}],
        data, ["W"], "2024-25", 10, "PerGame",
    )
    assert spec.intent == "team_leaders"
    assert len(data) == 5


def test_mixed_per_modes_unified():
    board = _fg3m_board()
    per_game = [dict(r, FG3M=round(r["FG3M"] / 82, 1), PER_MODE="PerGame") for r in board]
    data = per_game + [dict(r) for r in board]  # LLM double-call: both modes
    q = "which team won the most games in the 2024-25 season?"
    spec = agent._infer_spec(
        q, ["get_team_leaders"],
        [{"stat": "W", "season": "2024-25", "per_mode": "Totals"}],
        data, ["W"], "2024-25", 10, "PerGame",
    )
    assert spec.intent == "team_leaders"
    assert len(data) == 5
    assert {r["PER_MODE"] for r in data} == {"Totals"}


def test_find_named_entities():
    names = ["Boston Celtics", "Golden State Warriors", "Chicago Bulls"]
    assert R.find_named_entities("how many threes did the Warriors make", names) == [
        "Golden State Warriors"]
    assert R.find_named_entities("celtics vs lakers", names + ["Los Angeles Lakers"]) == [
        "Boston Celtics", "Los Angeles Lakers"]
    assert R.find_named_entities("which team won the most games", names) == []


def test_ranking_flavored_single_team_keeps_board():
    # "Most" needs the ranking context — downgrading would lose the answer.
    data = _fg3m_board()
    q = "Did the Warriors make the most threes in 2024-25?"
    spec = agent._infer_spec(
        q, ["get_team_leaders"],
        [{"stat": "FG3M", "season": "2024-25"}],
        data, ["FG3M"], "2024-25", 10, "Totals",
    )
    assert spec.intent == "team_leaders"
    assert len(data) == 5


def test_has_ranking_language():
    assert R.has_ranking_language("Did the Warriors make the most threes")
    assert R.has_ranking_language("Is Brunson the best scorer")
    assert not R.has_ranking_language("How many threes did the Warriors make")
    assert not R.has_ranking_language("how many points did brunson average")
