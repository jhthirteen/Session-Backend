"""Offline tests for agent arg sanitizing — no network, no Groq key.

Regression test for the Luka Doncic 400 bug: the model sent
{"last_n_seasons": null, "since_season": null} and Groq's server-side
validation rejected the call. Belt and suspenders now: nullable schema
unions (nba_tools.py) + omit-don't-send-null prompt rule + this sanitizer.

Run:  .venv/bin/python -m pytest tests/data_tooling/test_agent_offline.py -q
"""

from src.data_tooling.agent import _clean_args, _slim_tool_content


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


def _wide_trend_row(player, season, pts):
    row = {
        "PLAYER_NAME": player,
        "SEASON": season,
        "PTS": pts,
        "GP": 70,
        "TEAM_ABBREVIATION": "BOS",
        "PER_MODE": "PerGame",
    }
    # Simulate the ~30 nba_api columns that used to blow up TPM.
    for i in range(25):
        row[f"EXTRA_{i}"] = 1.5
    return row


def test_slim_tool_content_strips_extra_columns():
    import json

    rows = [_wide_trend_row("A", "2023-24", 25.0)]
    content = _slim_tool_content({"players": ["A"]}, rows, ["PTS"])
    slim = json.loads(content)
    assert set(slim[0]) == {
        "PLAYER_NAME",
        "SEASON",
        "PTS",
        "GP",
        "TEAM_ABBREVIATION",
    }
    assert len(content) < 500


def test_slim_tool_content_caps_trend_tail():
    import json

    rows = [_wide_trend_row("A", f"{2000 + i}-{(2001 + i) % 100:02d}", 20.0) for i in range(50)]
    content = _slim_tool_content({}, rows, ["PTS"])
    slim = json.loads(content.split(" [")[0])
    assert len(slim) == 40
    assert slim[-1]["SEASON"] == rows[-1]["SEASON"]  # most recent kept
    assert "omitted" in content


def test_slim_tool_content_caps_game_logs_head():
    import json

    rows = [
        {"PLAYER_NAME": "A", "GAME_DATE": f"Apr {i}, 2025", "PTS": 20.0, "MIN": 30.0}
        for i in range(1, 51)
    ]
    content = _slim_tool_content({}, rows, ["PTS"])
    slim = json.loads(content.split(" [")[0])
    assert len(slim) == 40
    assert slim[0]["GAME_DATE"] == "Apr 1, 2025"  # most recent kept
    assert "older games omitted" in content


def test_slim_tool_content_keeps_error_rows():
    import json

    rows = [{"PLAYER_NAME": "Bad", "error": "Unknown player."}]
    assert json.loads(_slim_tool_content({}, rows, ["PTS"])) == rows


def test_slim_tool_content_dict_echoed_whole():
    import json

    row = {"PLAYER_NAME": "A", "SEASON": "2024-25", "PTS": 26.0}
    assert json.loads(_slim_tool_content({}, row, ["PTS"])) == row
