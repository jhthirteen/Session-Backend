"""Offline tests for agent arg sanitizing — no network, no Groq key.

Regression test for the Luka Doncic 400 bug: the model sent
{"last_n_seasons": null, "since_season": null} and Groq's server-side
validation rejected the call. Belt and suspenders now: nullable schema
unions (nba_tools.py) + omit-don't-send-null prompt rule + this sanitizer.

Run:  .venv/bin/python -m pytest tests/data_tooling/test_agent_offline.py -q
"""

from src.data_tooling.agent import _clean_args


def test_null_optionals_dropped():
    assert _clean_args(
        "get_player_career_trend",
        {
            "player_name": "Luka Doncic",
            "per_mode": "PerGame",
            "last_n_seasons": None,
            "since_season": None,
        },
    ) == {"player_name": "Luka Doncic", "per_mode": "PerGame"}


def test_unknown_keys_dropped():
    assert _clean_args(
        "compare_players",
        {"players": ["A", "B"], "season": "2024-25", "metrics": None, "bogus": 1},
    ) == {"players": ["A", "B"], "season": "2024-25"}


def test_unknown_tool_passthrough():
    assert _clean_args("nope", {"a": None}) == {"a": None}


def test_required_args_kept():
    args = {"player_name": "Jalen Brunson", "season": "2024-25", "last_n": 5}
    assert _clean_args("get_player_game_logs", args) == args
