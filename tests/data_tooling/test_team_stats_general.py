"""Team single-stat generality — any stat, not just the record.

Covers: explicit_metrics(), team_stats viz routing (stat card vs record
card), and team_stats answer synthesis.

Run: .venv/bin/python -m pytest tests/data_tooling/test_team_stats_general.py -q
"""
from src.data_tooling import agent
from src.data_tooling import resolver as R

GSW = {"TEAM_NAME": "Golden State Warriors", "SEASON": "2024-25", "GP": 82,
       "W": 48, "L": 34, "W_PCT": 0.585, "FG3M": 1264, "PTS": 9331, "AST": 2386}
CELTICS = {"TEAM_NAME": "Boston Celtics", "SEASON": "2025-26", "GP": 82,
           "W": 56, "L": 26, "W_PCT": 0.683, "FG3M": 1100, "PTS": 9200}


def _spec(q, metrics, row):
    data = [dict(row)]
    spec = agent._infer_spec(
        q, ["get_team_stats"],
        [{"team_name": row["TEAM_NAME"], "season": row["SEASON"]}],
        data, metrics, row["SEASON"], 10, "Totals",
    )
    return spec, data


# --- explicit_metrics ------------------------------------------------------
def test_threes_is_explicit_fg3m():
    assert R.explicit_metrics(
        "How many three-pointers did the Golden State Warriors make in 2024-25?"
    ) == ["FG3M"]


def test_record_is_explicit_w():
    assert R.explicit_metrics("what was the celtics record last season") == ["W"]


def test_vague_query_has_no_explicit_metrics():
    assert R.explicit_metrics("how did the celtics do last season") == []


# --- viz routing -----------------------------------------------------------
def test_team_threes_routes_to_single_stat():
    q = "How many three-pointers did the Golden State Warriors make in 2024-25?"
    spec, _ = _spec(q, ["FG3M"], GSW)
    viz = R.choose_viz_hint(spec)
    assert viz.type == "single_stat"
    assert viz.y_keys == ["FG3M"]
    assert "Golden State Warriors" in viz.title


def test_team_points_routes_to_single_stat():
    q = "how many points did the celtics score in 2024-25?"
    spec, _ = _spec(q, ["PTS"], CELTICS)
    assert R.choose_viz_hint(spec).type == "single_stat"


def test_team_record_stays_record_card():
    q = "what was the celtics record last season"
    spec, _ = _spec(q, ["W"], CELTICS)
    assert R.choose_viz_hint(spec).type == "team_stat_card"


def test_vague_team_query_stays_record_card():
    q = "how did the celtics do last season"
    spec, _ = _spec(q, ["PTS"], CELTICS)
    assert R.choose_viz_hint(spec).type == "team_stat_card"


def test_player_single_stat_unchanged():
    data = [{"PLAYER_NAME": "Jalen Brunson", "SEASON": "2025-26", "PTS": 26.0}]
    spec = agent._infer_spec(
        "how many points did jalen brunson average last year",
        ["get_player_season_averages"],
        [{"player_name": "Jalen Brunson", "season": "2025-26"}],
        data, ["PTS"], "2025-26", 10, "PerGame",
    )
    assert R.choose_viz_hint(spec).type == "single_stat"


# --- synthesis -------------------------------------------------------------
def test_synthesize_team_threes():
    text = agent._synthesize_answer(
        "team_stats", [dict(GSW)],
        "How many three-pointers did the Golden State Warriors make in 2024-25?",
        ["FG3M"],
    )
    assert "1,264" in text and "FG3M" in text and "Golden State Warriors" in text


def test_synthesize_team_record_preserved():
    text = agent._synthesize_answer(
        "team_stats", [dict(CELTICS)],
        "what was the celtics record last season", ["W"],
    )
    assert "56-26" in text


def test_synthesize_team_multi_stat():
    text = agent._synthesize_answer(
        "team_stats", [dict(GSW)],
        "warriors points and assists in 2024-25?", ["PTS", "AST"],
    )
    assert "9,331 PTS" in text and "2,386 AST" in text
