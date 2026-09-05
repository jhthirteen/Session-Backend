"""Offline tests for resolver.py — no network, no Groq key needed.

Run:  .venv/bin/python -m pytest tests/data_tooling/test_resolver.py -q
"""
import datetime

import pytest

from src.data_tooling import resolver as R
from src.data_tooling.models import QuerySpec

TODAY = datetime.date(2026, 9, 5)

FAKE_PLAYERS = [
    {"id": 1, "full_name": "Jalen Brunson"},
    {"id": 2, "full_name": "Rick Brunson"},
    {"id": 3, "full_name": "Tyrese Haliburton"},
    {"id": 4, "full_name": "Luka Dončić"},
]

FAKE_TEAMS = [
    {"id": 1, "full_name": "Boston Celtics"},
    {"id": 2, "full_name": "New York Knicks"},
]


# --- seasons ---------------------------------------------------------------
def test_most_recent_completed_season_offseason():
    assert R.most_recent_completed_season(TODAY) == "2025-26"


def test_resolve_season_last_year():
    assert (
        R.resolve_season("how many points did brunson average last year", TODAY)
        == "2025-26"
    )


def test_resolve_season_explicit():
    assert R.resolve_season("celtics record in 2024-25", TODAY) == "2024-25"


def test_resolve_season_bare_year():
    assert R.resolve_season("lakers 2024", TODAY) == "2024-25"


def test_resolve_season_rookie_year_needs_context():
    assert R.resolve_season("wemby rookie year PPG", TODAY) is None


# --- metrics / game windows -------------------------------------------------
def test_resolve_metrics_multi():
    assert R.resolve_metrics("how many assists and rebounds did he average") == [
        "REB",
        "AST",
    ]


def test_resolve_metrics_default():
    assert R.resolve_metrics("how good is he") == ["PTS"]


def test_extract_last_n():
    assert R.extract_last_n("brunson last 5 games points") == 5


def test_extract_last_n_default():
    assert R.extract_last_n("brunson game log") == 10


# --- names ------------------------------------------------------------------
def test_resolve_player_exact_injected():
    name, cands = R.resolve_player("Jalen Brunson", FAKE_PLAYERS)
    assert name == "Jalen Brunson"
    assert cands == []


def test_resolve_player_ambiguous_never_guesses():
    name, cands = R.resolve_player("Brunson", FAKE_PLAYERS)
    assert name is None
    assert cands == ["Jalen Brunson", "Rick Brunson"]


def test_resolve_player_unknown():
    assert R.resolve_player("Mickey Mouse", FAKE_PLAYERS) == (None, [])


def test_resolve_player_accent_insensitive():
    assert R.resolve_player("Luka Doncic", FAKE_PLAYERS) == ("Luka Dončić", [])
    assert R.resolve_player("Luka Dončić", FAKE_PLAYERS) == ("Luka Dončić", [])
    assert R.resolve_player("luka doncic", FAKE_PLAYERS) == ("Luka Dončić", [])


def test_fold_accents():
    assert R.fold_accents("Dončić") == "doncic"
    assert R.fold_accents("Nikola Jokić") == "nikola jokic"


def test_resolve_team_alias():
    name, _ = R.resolve_team("Pels", FAKE_TEAMS)
    assert name == "New Orleans Pelicans"


def test_resolve_team_static_list():
    name, _ = R.resolve_team("Knicks", FAKE_TEAMS)
    assert name == "New York Knicks"


# --- trends -----------------------------------------------------------------
def test_detect_trend_intent():
    assert R.detect_trend_intent("highest PPG season of his career") is True
    assert R.detect_trend_intent("how many points last year") is False


def test_detect_improvement_scoped():
    assert R.detect_improvement_intent("most improved season") is True
    assert R.detect_improvement_intent("best season ever") is False


def test_resolve_per_mode():
    assert R.resolve_per_mode("most total points in a season") == "Totals"
    assert R.resolve_per_mode("ppg by season") == "PerGame"


def test_resolve_season_window_last_n():
    assert R.resolve_season_window("brunson last 5 seasons") == {
        "last_n_seasons": 5,
        "since_season": None,
    }


def test_resolve_season_window_since():
    assert R.resolve_season_window("lakers wins since 2020") == {
        "last_n_seasons": None,
        "since_season": "2020-21",
    }


def test_resolve_season_window_full():
    assert R.resolve_season_window("career ppg by season") == {
        "last_n_seasons": None,
        "since_season": None,
    }


def test_apply_season_window_last_n():
    seasons = ["2021-22", "2022-23", "2023-24", "2024-25"]
    assert R.apply_season_window(seasons, last_n_seasons=2) == [
        "2023-24",
        "2024-25",
    ]


def test_largest_yoy_jump():
    rows = [
        {"SEASON": "2021-22", "PTS": 16.3},
        {"SEASON": "2022-23", "PTS": 24.0},
        {"SEASON": "2023-24", "PTS": 28.7},
    ]
    season, delta, note = R.largest_yoy_jump(rows, "PTS")
    assert season == "2022-23"
    assert delta == pytest.approx(7.7)
    assert "2022-23" in note


def test_largest_yoy_jump_no_increase():
    rows = [
        {"SEASON": "2023-24", "PTS": 28.7},
        {"SEASON": "2024-25", "PTS": 26.0},
    ]
    assert R.largest_yoy_jump(rows, "PTS") == (None, None, None)


# --- viz --------------------------------------------------------------------
def test_viz_trend_line():
    spec = QuerySpec(
        intent="player_career_trend",
        players=["Jalen Brunson"],
        metrics=["PTS"],
        seasons=["2021-22", "2025-26"],
    )
    viz = R.choose_viz_hint(spec)
    assert viz.type == "trend_line"
    assert viz.x_key == "SEASON"
    assert viz.y_keys == ["PTS"]


def test_viz_single_stat_and_comparison():
    single = QuerySpec(
        intent="player_season_avg",
        players=["Jalen Brunson"],
        season="2025-26",
        metrics=["PTS"],
    )
    assert R.choose_viz_hint(single).type == "single_stat"
    comp = QuerySpec(
        intent="compare_players",
        players=["Jalen Brunson", "Tyrese Haliburton"],
        season="2024-25",
        metrics=["PTS"],
    )
    assert R.choose_viz_hint(comp).type == "comparison_bars"
