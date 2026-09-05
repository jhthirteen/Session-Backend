"""Offline tests for models.py — pure Pydantic contracts, no I/O.

Run:  .venv/bin/python -m pytest tests/data_tooling/test_models.py -q
"""
import json

import pytest
from pydantic import ValidationError

from src.data_tooling.models import QueryResponse, QuerySpec, VizHint


def test_query_spec_defaults():
    spec = QuerySpec(intent="player_season_avg")
    assert spec.players == []
    assert spec.metrics == ["PTS"]
    assert spec.per_mode == "PerGame"
    assert spec.seasons == []
    assert spec.highlight_season is None
    assert spec.highlight_note is None


def test_query_spec_trend_fields():
    spec = QuerySpec(
        intent="player_career_trend",
        players=["Jalen Brunson"],
        per_mode="Totals",
        seasons=["2022-23", "2023-24"],
        highlight_season="2022-23",
        highlight_note="largest PTS jump: +7.7 (2022-23)",
    )
    assert spec.per_mode == "Totals"
    assert spec.highlight_season == "2022-23"


def test_invalid_intent_rejected():
    with pytest.raises(ValidationError):
        QuerySpec(intent="mvp_prediction")  # type: ignore[arg-type]


def test_query_response_json_round_trip():
    resp = QueryResponse(
        answer_text="Jalen Brunson averaged 26.0 PPG in 2025-26.",
        spec=QuerySpec(
            intent="player_season_avg",
            players=["Jalen Brunson"],
            season="2025-26",
            metrics=["PTS"],
        ),
        data=[{"PLAYER_NAME": "Jalen Brunson", "SEASON": "2025-26", "PTS": 26.0}],
        viz_hint=VizHint(type="single_stat", title="x", y_keys=["PTS"]),
        debug=[],
    )
    assert json.loads(resp.model_dump_json())["spec"]["intent"] == "player_season_avg"


def test_trend_viz_hint_valid():
    viz = VizHint(
        type="trend_line",
        title="Jalen Brunson PTS per game by season",
        x_key="SEASON",
        y_keys=["PTS"],
    )
    assert viz.x_key == "SEASON"
