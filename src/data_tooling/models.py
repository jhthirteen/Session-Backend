"""Pydantic contracts for the NLP visualizer.

Frontend contract (React):
    QueryResponse { answer_text, spec, data, viz_hint, debug }

No nba_api / Groq imports here — pure schemas so both
agent.py and api.py can share them.
"""

from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Intents (V1 scope: players + teams only — no leaders/standings/play-by-play)
# ---------------------------------------------------------------------------
Intent = Literal[
    "player_season_avg",
    "player_game_logs",
    "team_stats",
    "team_game_logs",
    "compare_players",
    "compare_teams",
    "compare_trends",
    "player_career_trend",
    "team_history_trend",
    "needs_clarification",
]

VizType = Literal[
    "single_stat",  # one big number, e.g. Brunson PPG 2024-25
    "comparison_bars",  # 2+ entities side-by-side (x_key says which axis)
    "time_series",  # game-by-game line chart
    "game_log_table",  # last-N games table
    "team_stat_card",  # team record + ratings card
    "trend_line",  # season-by-season line chart (x=SEASON, chartable per-game or totals)
    "multi_trend",  # 2+ entities over time — one line per series_key value
]

# Canonical metric keys (match nba_api column names where possible).
MetricKey = Literal[
    "PTS",
    "AST",
    "REB",
    "STL",
    "BLK",
    "MIN",
    "FG_PCT",
    "FG3M",
    "FG3_PCT",
    "FT_PCT",
    "W",
    "L",
    "W_PCT",
]


class QuerySpec(BaseModel):
    """Structured interpretation of the user's natural-language query."""

    intent: Intent = Field(description="What kind of data the user wants.")
    players: List[str] = Field(
        default_factory=list, description="Canonical 'First Last' player names."
    )
    teams: List[str] = Field(
        default_factory=list,
        description="Canonical team names, e.g. 'Boston Celtics'.",
    )
    season: Optional[str] = Field(
        default=None,
        description="Season in 'YYYY-YY' nba_api format, e.g. '2024-25'.",
    )
    metrics: List[MetricKey] = Field(
        default_factory=lambda: ["PTS"],
        description="Metrics to fetch / plot.",
    )
    last_n: Optional[int] = Field(
        default=None,
        description="For game-log queries: how many recent games (default 10).",
    )
    per_mode: Literal["PerGame", "Totals"] = Field(
        default="PerGame",
        description="Trend stat flavor: per-game averages or season totals.",
    )
    seasons: List[str] = Field(
        default_factory=list,
        description="Resolved seasons covered by a trend (chronological).",
    )
    highlight_season: Optional[str] = Field(
        default=None,
        description="Improvement queries only: season with largest YoY jump.",
    )
    highlight_note: Optional[str] = Field(
        default=None,
        description="Human-readable note for highlight_season, e.g. '+7.7 PTS'.",
    )
    raw_query: Optional[str] = Field(
        default=None, description="Original user string for debugging."
    )


class VizHint(BaseModel):
    """Tells React which component to render and which keys to use."""

    type: VizType
    title: str
    x_key: Optional[str] = Field(
        default=None,
        description="X-axis key: 'GAME_DATE' (time_series), 'SEASON' (trends), "
        "'PLAYER_NAME'/'TEAM_NAME' (comparison_bars snapshots).",
    )
    y_keys: List[str] = Field(
        default_factory=list, description="E.g. ['PTS'] or ['PTS', 'AST']."
    )
    series_key: Optional[str] = Field(
        default=None,
        description="Compare discriminator: 'PLAYER_NAME' / 'TEAM_NAME' / "
        "'SEASON'. Null = single series (legacy behavior).",
    )


class ToolCallTrace(BaseModel):
    """One executed tool call in the Groq agentic loop (for debugging)."""

    tool: str
    args: Dict = Field(default_factory=dict)
    # Keep result summary small — full rows live in QueryResponse.data.
    result_summary: Optional[str] = None


class QueryResponse(BaseModel):
    """Full payload returned to the frontend."""

    answer_text: str = Field(description="One-sentence natural-language answer.")
    spec: QuerySpec
    data: List[Dict] = Field(
        default_factory=list,
        description="Row-oriented, JSON-serializable, chart-ready records.",
    )
    viz_hint: VizHint
    debug: List[ToolCallTrace] = Field(default_factory=list)


class ClarificationNeeded(BaseModel):
    """Returned when a name/season is ambiguous — never guess."""

    message: str
    candidates: List[str] = Field(default_factory=list)
