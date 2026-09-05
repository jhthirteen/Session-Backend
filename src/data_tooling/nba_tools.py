"""MCP-style tool module wrapping swar/nba_api (https://github.com/swar/nba_api).

Each public function is one Groq tool. nba_api is imported lazily so this
module imports cleanly even before `pip install nba_api` — calls will raise
a clear error telling you what to install (see ToolError).

Endpoints used (all from swar/nba_api docs):
  - PlayerCareerStats(player_id, per_mode36='PerGame') -> season_totals_regular_season
  - PlayerGameLog(player_id, season, season_type_all_star='Regular Season')
  - TeamGameLog(team_id, season, season_type_all_star='Regular Season')
  - TeamYearByYearStats(team_id) -> team_stats rows incl. YEAR/W/L/W_PCT
  - CommonPlayerInfo(player_id) -> common_player_info (for disambiguation/debug)

Conventions:
  - season format is always 'YYYY-YY', e.g. '2025-26' (see resolver.py).
  - timeout=30s on every request; one retry on failure.
  - 1-hour in-memory TTL cache keyed on (tool, args) to avoid NBA rate limits.
"""

import time
from typing import Any, Callable, Dict, List, Optional

from . import resolver
from .resolver import resolve_player, resolve_team


class ToolError(RuntimeError):
    """Raised when a tool can't fulfill the request (bad name, no season, API down).

    The agent catches this and either asks for clarification or reports
    the failure instead of hallucinating stats.
    """

    def __init__(self, message: str, candidates: Optional[List[str]] = None):
        super().__init__(message)
        self.candidates = candidates or []


# ---------------------------------------------------------------------------
# Tiny TTL cache (no extra deps)
# ---------------------------------------------------------------------------
_CACHE: Dict[str, Any] = {}
_TTL_SECONDS = 3600


def _cache_key(tool: str, args: Dict[str, Any]) -> str:
    parts = [tool] + [f"{k}={v}" for k, v in sorted(args.items())]
    return "|".join(parts)


def _cache_get(key: str) -> Optional[Any]:
    hit = _CACHE.get(key)
    if not hit:
        return None
    ts, value = hit
    if time.time() - ts > _TTL_SECONDS:
        _CACHE.pop(key, None)
        return None
    return value


def _cache_set(key: str, value: Any) -> None:
    _CACHE[key] = (time.time(), value)


def _require_nba_api():
    try:
        import nba_api  # type: ignore  # noqa: F401
    except ImportError as e:
        raise ToolError(
            "nba_api is not installed. Run `.venv/bin/pip install nba-api pandas` "
            "(note: nba_api>=1.11 requires Python>=3.10; this repo's .venv is "
            "Python 3.9 — see checkpoint-2 notes) then retry."
        ) from e


def _call_with_retry(fn: Callable[[], Any], timeout_note: str = "") -> Any:
    """Call an nba_api endpoint constructor with one retry on failure."""
    last_err: Optional[Exception] = None
    for _ in range(2):
        try:
            return fn()
        except Exception as e:  # network / rate-limit / bad param
            last_err = e
            time.sleep(1.0)
    raise ToolError(f"NBA stats API request failed {timeout_note}: {last_err}")


# ---------------------------------------------------------------------------
# ID resolution
# ---------------------------------------------------------------------------
def get_player_id(player_name: str) -> int:
    _require_nba_api()
    from nba_api.stats.static import players  # type: ignore

    canonical, candidates = resolve_player(player_name, players.get_players())
    if canonical is None:
        if candidates:
            raise ToolError(
                f"Ambiguous player '{player_name}'. Candidates: {', '.join(candidates)}.",
                candidates=candidates,
            )
        raise ToolError(f"Unknown player '{player_name}'. Check spelling.")
    matches = players.find_players_by_full_name(canonical)
    if not matches:
        raise ToolError(f"Unknown player '{player_name}'. Check spelling.")
    return matches[0]["id"]


def get_team_id(team_name: str) -> int:
    _require_nba_api()
    from nba_api.stats.static import teams  # type: ignore

    canonical, candidates = resolve_team(team_name, teams.get_teams())
    if canonical is None:
        if candidates:
            raise ToolError(
                f"Ambiguous team '{team_name}'. Candidates: {', '.join(candidates)}.",
                candidates=candidates,
            )
        raise ToolError(f"Unknown team '{team_name}'. Try e.g. 'Celtics' or 'BOS'.")
    found = teams.find_teams_by_full_name(canonical)
    if not found:
        # Alias resolved to a canonical name but static lookup missed (e.g.
        # 'LA Clippers' vs 'Los Angeles Clippers') — match case-insensitively.
        all_teams = teams.get_teams()
        for t in all_teams:
            if t["full_name"].lower() == canonical.lower():
                return t["id"]
        raise ToolError(f"Unknown team '{team_name}'.")
    return found[0]["id"]


# ---------------------------------------------------------------------------
# Tools (1:1 with Groq function schemas below)
# ---------------------------------------------------------------------------
def get_player_season_averages(player_name: str, season: str) -> Dict[str, Any]:
    """Per-game averages for one player + season, e.g. Brunson 2025-26 PTS/AST/REB."""
    key = _cache_key("get_player_season_averages", {"p": player_name, "s": season})
    hit = _cache_get(key)
    if hit is not None:
        return hit

    pid = get_player_id(player_name)
    from nba_api.stats.endpoints import playercareerstats  # type: ignore

    obj = _call_with_retry(
        lambda: playercareerstats.PlayerCareerStats(
            player_id=pid, per_mode36="PerGame", timeout=30
        )
    )
    frames = obj.get_data_frames()
    # season_totals_regular_season is the first/primary frame for this endpoint.
    df = frames[0]
    # SEASON_ID looks like '2025-26'.
    row = df[df["SEASON_ID"] == season]
    if row.empty:
        # Player may have been inactive / did not play that season.
        raise ToolError(f"No stats for {player_name} in season {season}.")
    record = row.iloc[0].to_dict()
    # Normalize numpy types -> plain Python for JSON.
    record = {k: (v.item() if hasattr(v, "item") else v) for k, v in record.items()}
    record["PLAYER_NAME"] = player_name
    record["SEASON"] = season
    _cache_set(key, record)
    return record


def get_player_game_logs(
    player_name: str, season: str, last_n: int = 10
) -> List[Dict[str, Any]]:
    """Most recent N game logs for a player + season (default 10)."""
    last_n = max(1, min(int(last_n or 10), 82))
    key = _cache_key(
        "get_player_game_logs", {"p": player_name, "s": season, "n": last_n}
    )
    hit = _cache_get(key)
    if hit is not None:
        return hit

    pid = get_player_id(player_name)
    from nba_api.stats.endpoints import playergamelog  # type: ignore

    obj = _call_with_retry(
        lambda: playergamelog.PlayerGameLog(
            player_id=pid, season=season, season_type_all_star="Regular Season",
            timeout=30,
        )
    )
    df = obj.get_data_frames()[0]
    if df.empty:
        raise ToolError(f"No game logs for {player_name} in season {season}.")
    # API returns reverse-chronological (most recent first) — head = latest.
    head = df.head(last_n)
    records = head.to_dict(orient="records")
    clean = [
        {k: (v.item() if hasattr(v, "item") else v) for k, v in r.items()}
        for r in records
    ]
    for r in clean:
        r["PLAYER_NAME"] = player_name
    _cache_set(key, clean)
    return clean


def get_team_stats(team_name: str, season: str) -> Dict[str, Any]:
    """Season row for a team (W/L/W_PCT + points etc.) via TeamYearByYearStats."""
    key = _cache_key("get_team_stats", {"t": team_name, "s": season})
    hit = _cache_get(key)
    if hit is not None:
        return hit

    tid = get_team_id(team_name)
    from nba_api.stats.endpoints import teamyearbyyearstats  # type: ignore

    obj = _call_with_retry(
        lambda: teamyearbyyearstats.TeamYearByYearStats(team_id=tid, timeout=30)
    )
    df = obj.get_data_frames()[0]
    # Column is YEAR like '2025-26'.
    row = df[df["YEAR"] == season]
    if row.empty:
        raise ToolError(f"No stats for {team_name} in season {season}.")
    record = row.iloc[0].to_dict()
    record = {k: (v.item() if hasattr(v, "item") else v) for k, v in record.items()}
    # Normalize: TeamYearByYearStats uses WINS/LOSSES/WIN_PCT — alias to W/L/W_PCT
    # so the frontend + agent can use one key set for players and teams.
    if "WINS" in record and "W" not in record:
        record["W"] = record["WINS"]
    if "LOSSES" in record and "L" not in record:
        record["L"] = record["LOSSES"]
    if "WIN_PCT" in record and "W_PCT" not in record:
        record["W_PCT"] = record["WIN_PCT"]
    record["TEAM_NAME"] = team_name
    record["SEASON"] = season
    _cache_set(key, record)
    return record


def get_team_game_logs(
    team_name: str, season: str, last_n: int = 10
) -> List[Dict[str, Any]]:
    """Most recent N game logs for a team + season (default 10)."""
    last_n = max(1, min(int(last_n or 10), 82))
    key = _cache_key(
        "get_team_game_logs", {"t": team_name, "s": season, "n": last_n}
    )
    hit = _cache_get(key)
    if hit is not None:
        return hit

    tid = get_team_id(team_name)
    from nba_api.stats.endpoints import teamgamelog  # type: ignore

    obj = _call_with_retry(
        lambda: teamgamelog.TeamGameLog(
            team_id=tid, season=season, season_type_all_star="Regular Season",
            timeout=30,
        )
    )
    df = obj.get_data_frames()[0]
    if df.empty:
        raise ToolError(f"No game logs for {team_name} in season {season}.")
    # Same as player logs: reverse-chronological, head = latest.
    head = df.head(last_n)
    records = head.to_dict(orient="records")
    clean = [
        {k: (v.item() if hasattr(v, "item") else v) for k, v in r.items()}
        for r in records
    ]
    for r in clean:
        r["TEAM_NAME"] = team_name
    _cache_set(key, clean)
    return clean


def compare_players(
    players: List[str], season: str, metrics: Optional[List[str]] = None
) -> List[Dict[str, Any]]:
    """Fan-out: season averages for 2+ players, trimmed to requested metrics."""
    metrics = metrics or ["PTS"]
    rows: List[Dict[str, Any]] = []
    for name in players:
        try:
            full = get_player_season_averages(name, season)
        except ToolError as e:
            # Partial failure (e.g. missed season to injury) — keep the row
            # so the frontend can render available bars + a missing note.
            rows.append(
                {"PLAYER_NAME": name, "SEASON": season, "error": str(e)}
            )
            continue
        trimmed: Dict[str, Any] = {
            "PLAYER_NAME": full.get("PLAYER_NAME", name),
            "SEASON": season,
            "GP": full.get("GP"),
            "TEAM_ABBREVIATION": full.get("TEAM_ABBREVIATION"),
        }
        for m in metrics:
            trimmed[m] = full.get(m)
        rows.append(trimmed)
    return rows


def _normalize_row(record: Dict[str, Any]) -> Dict[str, Any]:
    return {k: (v.item() if hasattr(v, "item") else v) for k, v in record.items()}


def _season_start(season: str) -> int:
    try:
        return int(str(season).split("-")[0])
    except (ValueError, IndexError, AttributeError):
        return 0


def get_player_career_trend(
    player_name: str,
    per_mode: str = "PerGame",
    last_n_seasons: Optional[int] = None,
    since_season: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Season-by-season history for one player in ONE API call.

    per_mode: 'PerGame' (default) or 'Totals'. Windowing: last_n_seasons and/or
    since_season ('YYYY-YY') filter server-side. Traded mid-season years collapse
    to the TOT aggregate row (see V1_SCOPE_AND_GAPS.txt). Rows chronological.
    """
    if per_mode not in ("PerGame", "Totals"):
        raise ToolError(f"per_mode must be 'PerGame' or 'Totals', got '{per_mode}'.")
    key = _cache_key("get_player_career_trend", {
        "p": player_name, "m": per_mode, "n": last_n_seasons, "s": since_season,
    })
    hit = _cache_get(key)
    if hit is not None:
        return hit

    pid = get_player_id(player_name)
    from nba_api.stats.endpoints import playercareerstats  # type: ignore

    obj = _call_with_retry(
        lambda: playercareerstats.PlayerCareerStats(
            player_id=pid, per_mode36=per_mode, timeout=30
        )
    )
    df = obj.get_data_frames()[0]  # season_totals_regular_season: ALL seasons
    if df.empty:
        raise ToolError(f"No career stats for {player_name}.")

    # TOT-wins dedup: group by SEASON_ID, prefer TEAM_ABBREVIATION == 'TOT'.
    seasons: Dict[str, Any] = {}
    order: List[str] = []
    for _, series in df.iterrows():
        sid = str(series["SEASON_ID"])
        if sid not in seasons:
            seasons[sid] = series
            order.append(sid)
        elif str(series.get("TEAM_ABBREVIATION")) == "TOT":
            seasons[sid] = series
    rows = [_normalize_row(seasons[s].to_dict()) for s in order]
    for r in rows:
        r["PLAYER_NAME"] = player_name
        r["SEASON"] = str(r.get("SEASON_ID"))
        r["PER_MODE"] = per_mode

    window = resolver.apply_season_window(
        [r["SEASON"] for r in rows], last_n_seasons, since_season
    )
    keep = set(window)
    rows = [r for r in rows if r["SEASON"] in keep]
    rows.sort(key=lambda r: _season_start(r["SEASON"]))
    if not rows:
        raise ToolError(f"No seasons for {player_name} in that window.")
    _cache_set(key, rows)
    return rows


def get_team_history_trend(
    team_name: str,
    last_n_seasons: Optional[int] = None,
    since_season: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Year-by-year history for one team in ONE API call.

    Windowing works like the player trend. Rows carry W/L/W_PCT aliases plus a
    unified SEASON key so React uses one trend_line path for players and teams.
    """
    key = _cache_key("get_team_history_trend", {
        "t": team_name, "n": last_n_seasons, "s": since_season,
    })
    hit = _cache_get(key)
    if hit is not None:
        return hit

    tid = get_team_id(team_name)
    from nba_api.stats.endpoints import teamyearbyyearstats  # type: ignore

    obj = _call_with_retry(
        lambda: teamyearbyyearstats.TeamYearByYearStats(team_id=tid, timeout=30)
    )
    df = obj.get_data_frames()[0]  # one row per YEAR, full history
    if df.empty:
        raise ToolError(f"No history for {team_name}.")

    rows = [_normalize_row(r) for r in df.to_dict(orient="records")]
    for r in rows:
        if "WINS" in r and "W" not in r:
            r["W"] = r["WINS"]
        if "LOSSES" in r and "L" not in r:
            r["L"] = r["LOSSES"]
        if "WIN_PCT" in r and "W_PCT" not in r:
            r["W_PCT"] = r["WIN_PCT"]
        r["TEAM_NAME"] = team_name
        r["SEASON"] = str(r.get("YEAR"))

    window = resolver.apply_season_window(
        [r["SEASON"] for r in rows], last_n_seasons, since_season
    )
    keep = set(window)
    rows = [r for r in rows if r["SEASON"] in keep]
    rows.sort(key=lambda r: _season_start(r["SEASON"]))
    if not rows:
        raise ToolError(f"No seasons for {team_name} in that window.")
    _cache_set(key, rows)
    return rows


# ---------------------------------------------------------------------------
# Registry + Groq tool schemas (agentic loop in agent.py consumes these)
# ---------------------------------------------------------------------------
TOOL_FUNCS: Dict[str, Callable[..., Any]] = {
    "get_player_season_averages": get_player_season_averages,
    "get_player_game_logs": get_player_game_logs,
    "get_team_stats": get_team_stats,
    "get_team_game_logs": get_team_game_logs,
    "compare_players": compare_players,
    "get_player_career_trend": get_player_career_trend,
    "get_team_history_trend": get_team_history_trend,
}

GROQ_TOOL_SCHEMAS: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "get_player_season_averages",
            "description": "Get per-game season averages for one NBA player and season.",
            "parameters": {
                "type": "object",
                "properties": {
                    "player_name": {"type": "string", "description": "E.g. 'Jalen Brunson'"},
                    "season": {"type": "string", "description": "Season 'YYYY-YY', e.g. '2025-26'"},
                },
                "required": ["player_name", "season"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_player_game_logs",
            "description": "Get most recent N game logs for a player and season.",
            "parameters": {
                "type": "object",
                "properties": {
                    "player_name": {"type": "string"},
                    "season": {"type": "string", "description": "'YYYY-YY'"},
                    "last_n": {"type": ["integer", "null"], "description": "Games to return, 1-82 (default 10). Omit if unsure."},
                },
                "required": ["player_name", "season"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_team_stats",
            "description": "Get season stats (W/L/W_PCT) for one NBA team and season.",
            "parameters": {
                "type": "object",
                "properties": {
                    "team_name": {"type": "string", "description": "E.g. 'Boston Celtics' or 'BOS'"},
                    "season": {"type": "string", "description": "'YYYY-YY'"},
                },
                "required": ["team_name", "season"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_team_game_logs",
            "description": "Get most recent N game logs for a team and season.",
            "parameters": {
                "type": "object",
                "properties": {
                    "team_name": {"type": "string"},
                    "season": {"type": "string", "description": "'YYYY-YY'"},
                    "last_n": {"type": ["integer", "null"], "description": "1-82 (default 10). Omit if unsure."},
                },
                "required": ["team_name", "season"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "compare_players",
            "description": "Compare per-game season averages for 2+ players on given metrics.",
            "parameters": {
                "type": "object",
                "properties": {
                    "players": {"type": "array", "items": {"type": "string"}},
                    "season": {"type": "string", "description": "'YYYY-YY'"},
                    "metrics": {
                        "type": ["array", "null"],
                        "items": {"type": "string"},
                        "description": "E.g. ['PTS','AST','REB']. Omit for default.",
                    },
                },
                "required": ["players", "season"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_player_career_trend",
            "description": (
                "Get season-by-season history for one player in a SINGLE call. "
                "USE THIS for any career/multi-season question "
                "(best season ever, year by year, over his career, progression, "
                "most improved, totals across seasons) — NEVER loop "
                "get_player_season_averages per season."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "player_name": {"type": "string"},
                    "per_mode": {
                        "type": ["string", "null"],
                        "enum": ["PerGame", "Totals", None],
                        "description": "Per-game averages (default) or season totals. Omit if unsure.",
                    },
                    "last_n_seasons": {
                        "type": ["integer", "null"],
                        "description": "Optional window, e.g. 5 for 'last 5 seasons'. Omit for full history.",
                    },
                    "since_season": {
                        "type": ["string", "null"],
                        "description": "Optional cutoff 'YYYY-YY', e.g. '2020-21' for 'since 2020'. Omit for full history.",
                    },
                },
                "required": ["player_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_team_history_trend",
            "description": (
                "Get year-by-year history for one team in a SINGLE call. USE THIS "
                "for any multi-season team question (wins by year, record over "
                "time, best season in franchise stretch) — NEVER loop "
                "get_team_stats per season."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "team_name": {"type": "string"},
                    "last_n_seasons": {
                        "type": ["integer", "null"],
                        "description": "Optional window, e.g. 10 for 'last 10 years'. Omit for full history.",
                    },
                    "since_season": {
                        "type": ["string", "null"],
                        "description": "Optional cutoff 'YYYY-YY'. Omit for full history.",
                    },
                },
                "required": ["team_name"],
            },
        },
    },
]

# Re-export resolver helpers for the agent (keeps imports in one place).
resolve_season = resolver.resolve_season
resolve_metrics = resolver.resolve_metrics
extract_last_n = resolver.extract_last_n
most_recent_completed_season = resolver.most_recent_completed_season
detect_trend_intent = resolver.detect_trend_intent
detect_improvement_intent = resolver.detect_improvement_intent
resolve_per_mode = resolver.resolve_per_mode
resolve_season_window = resolver.resolve_season_window
apply_season_window = resolver.apply_season_window
largest_yoy_jump = resolver.largest_yoy_jump
