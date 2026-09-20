"""Live tests for nba_tools.py — hit the real NBA stats API.

Gated: skipped unless RUN_LIVE=1 (network + latency, ~1-2 min total).
Run:  RUN_LIVE=1 .venv/bin/python -m pytest tests/data_tooling/test_nba_tools_live.py -q
"""
import os

import pytest

RUN_LIVE = os.environ.get("RUN_LIVE") == "1"
needs_live = pytest.mark.skipif(not RUN_LIVE, reason="needs RUN_LIVE=1 (NBA API)")


@pytest.fixture(scope="module")
def tools():
    from src.data_tooling import nba_tools as T

    T._CACHE.clear()
    return T


@needs_live
def test_player_season_averages_brunson(tools):
    row = tools.get_player_season_averages("Jalen Brunson", "2024-25")
    assert row["PTS"] == pytest.approx(26.0)
    assert row["GP"] == 65
    assert row["TEAM_ABBREVIATION"] == "NYK"


@needs_live
def test_career_trend_chronological_and_complete(tools):
    rows = tools.get_player_career_trend("Jalen Brunson")
    seasons = [r["SEASON"] for r in rows]
    assert seasons == sorted(seasons)  # chronological
    assert seasons[0] == "2018-19"
    assert "2025-26" in seasons
    assert all(r["PER_MODE"] == "PerGame" for r in rows)


@needs_live
def test_career_trend_window_and_totals(tools):
    windowed = tools.get_player_career_trend("Jalen Brunson", last_n_seasons=3)
    assert [r["SEASON"] for r in windowed][-3:] == ["2023-24", "2024-25", "2025-26"]
    totals = tools.get_player_career_trend("Jalen Brunson", per_mode="Totals")
    row = next(r for r in totals if r["SEASON"] == "2023-24")
    assert row["PTS"] == 2212  # season total, not average


@needs_live
def test_traded_year_collapses_to_tot(tools):
    rows = tools.get_player_career_trend("Kevin Durant")
    stints_2223 = [r for r in rows if r["SEASON"] == "2022-23"]
    assert len(stints_2223) == 1  # BKN + PHX collapse to one row
    assert stints_2223[0]["TEAM_ABBREVIATION"] == "TOT"


@needs_live
def test_team_history_aliases_and_window(tools):
    rows = tools.get_team_history_trend("Boston Celtics", last_n_seasons=3)
    assert [(r["SEASON"], r["W"], r["L"]) for r in rows] == [
        ("2023-24", 64, 18),
        ("2024-25", 61, 21),
        ("2025-26", 56, 26),
    ]
    assert all(r["W_PCT"] is not None for r in rows)


@needs_live
def test_game_logs_return_latest_first_regression(tools):
    """Regression: API is reverse-chron; logs must be the latest games."""
    logs = tools.get_player_game_logs("Jalen Brunson", "2024-25", last_n=3)
    assert len(logs) == 3
    assert all("2025" in str(g["GAME_DATE"]) for g in logs)


@needs_live
def test_ambiguous_name_raises_with_candidates(tools):
    with pytest.raises(tools.ToolError) as exc:
        tools.get_player_season_averages("Brunson", "2024-25")
    assert "Jalen Brunson" in str(exc.value)


@needs_live
def test_compare_partial_failure_keeps_rows(tools):
    rows = tools.compare_players(
        ["Jalen Brunson", "Tyrese Haliburton"], "2025-26", ["AST"]
    )
    assert len(rows) == 2
    assert rows[0]["AST"] is not None
    assert "error" in rows[1]  # missed season surfaces, doesn't nuke the table
