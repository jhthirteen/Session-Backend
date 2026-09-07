"""Offline tests for the visualization-robustness fixes — no network/Groq key.

Covers:
  A. metric word-boundary matching (no PTS false-positive in "three pointers")
  B. verb-aware per_mode (made/make -> Totals, average/per game -> PerGame)
  C. single-season scope guardrail in _infer_spec (trend tool + explicit
     season, no trend words -> sliced snapshot + single_stat/team_stat_card)

Run: .venv/bin/python -m pytest tests/data_tooling/test_scope_guardrail.py -q
"""
import datetime

from src.data_tooling import agent
from src.data_tooling import resolver as R
from src.data_tooling.resolver import choose_viz_hint

TODAY = datetime.date(2026, 9, 7)


def _trend_rows(player="Stephen Curry", seasons=("2021-22", "2022-23", "2023-24", "2024-25")):
    return [
        {"PLAYER_NAME": player, "SEASON": s, "FG3M": 357 if s == "2023-24" else 300}
        for s in seasons
    ]


# --- A. metrics ------------------------------------------------------------
def test_three_pointers_is_fg3m_only():
    assert R.resolve_metrics("How many three pointers did Stephen Curry make in 2023-24?") == ["FG3M"]


def test_pointer_substring_does_not_fire_pts():
    assert "PTS" not in R.resolve_metrics("three pointers")


def test_points_still_pts():
    assert R.resolve_metrics("how many points did brunson average") == ["PTS"]


def test_three_point_percentage_no_double_match():
    assert R.resolve_metrics("what is his three point percentage") == ["FG3_PCT"]


# --- B. per_mode -----------------------------------------------------------
def test_how_many_made_is_totals():
    assert R.resolve_per_mode("How many three pointers did Curry make in 2023-24?") == "Totals"


def test_average_stays_per_game():
    assert R.resolve_per_mode("how many points did brunson average last year") == "PerGame"


def test_ppg_stays_per_game():
    assert R.resolve_per_mode("brunson ppg career") == "PerGame"


# --- scope helpers ---------------------------------------------------------
def test_has_explicit_season():
    assert R.has_explicit_season("in 2023-24", TODAY)
    assert R.has_explicit_season("last season", TODAY)
    assert not R.has_explicit_season("best season ever", TODAY)


def test_single_season_scope():
    assert R.is_single_season_scope("How many threes did Curry make in 2023-24?", TODAY)
    assert not R.is_single_season_scope("Curry year by year", TODAY)
    assert not R.is_single_season_scope("compare Curry vs Brunson in 2023-24", TODAY)
    assert not R.is_single_season_scope("Curry last 5 seasons", TODAY)


# --- C. guardrail ----------------------------------------------------------
def test_guardrail_downgrades_curry_trend_to_single_stat():
    data = _trend_rows()
    q = "How many three pointers did Stephen Curry make in 2023-24?"
    spec = agent._infer_spec(
        q, ["get_player_career_trend"], [{"player_name": "Stephen Curry"}],
        data, ["FG3M"], "2023-24", 10, "Totals",
    )
    assert spec.intent == "player_season_avg"
    assert spec.season == "2023-24"
    assert spec.seasons == ["2023-24"]
    assert len(data) == 1 and data[0]["SEASON"] == "2023-24"
    assert choose_viz_hint(spec).type == "single_stat"


def test_guardrail_leaves_true_trend_alone():
    data = _trend_rows(player="Jalen Brunson")
    q = "what is brunson's highest points per game season of his career?"
    spec = agent._infer_spec(
        q, ["get_player_career_trend"], [{"player_name": "Jalen Brunson"}],
        data, ["PTS"], "2025-26", 10, "PerGame",
    )
    assert spec.intent == "player_career_trend"
    assert len(data) == 4
    assert choose_viz_hint(spec).type == "trend_line"


def test_guardrail_team_history_to_team_stats():
    data = [{"TEAM_NAME": "Boston Celtics", "SEASON": s, "W": 50} for s in ("2023-24", "2024-25")]
    q = "what was the celtics record in 2024-25?"
    spec = agent._infer_spec(
        q, ["get_team_history_trend"], [{"team_name": "Boston Celtics"}],
        data, ["W"], "2024-25", 10, "PerGame",
    )
    assert spec.intent == "team_stats"
    assert choose_viz_hint(spec).type == "team_stat_card"


def test_guardrail_passthrough_when_season_row_missing():
    data = _trend_rows()  # no 2025-26 row
    q = "how many threes did curry make last season?"
    season = R.resolve_season(q, TODAY)  # 2025-26
    spec = agent._infer_spec(
        q, ["get_player_career_trend"], [{"player_name": "Stephen Curry"}],
        data, ["FG3M"], season, 10, "Totals",
    )
    # Never fabricate: keep the trend rather than slicing to nothing.
    assert spec.intent == "player_career_trend"
    assert len(data) == 4
