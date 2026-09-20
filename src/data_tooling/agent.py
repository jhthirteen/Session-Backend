"""Groq agentic loop for the NLP visualizer (Checkpoint 3).

Pattern (per your vote): agentic tool loop.
  user query -> Groq (with tools=GROQ_TOOL_SCHEMAS) -> execute nba_tools ->
  feed results back -> repeat until final answer (max 6 iters).

Same Groq conventions as news.py: Groq(api_key=os.environ.get("GROQ_API_KEY")),
default model "openai/gpt-oss-120b". Override with GROQ_MODEL env var.
"""

import datetime
import inspect
import json
import os
from typing import Any, Dict, List, Optional, get_args

from groq import Groq

from . import resolver
from .models import MetricKey, QueryResponse, QuerySpec, ToolCallTrace
from .nba_tools import GROQ_TOOL_SCHEMAS, TOOL_FUNCS, ToolError
from .resolver import choose_viz_hint

DEFAULT_MODEL = os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b")
MAX_ITERS = 6

#: Canonical stat keys the model may choose in the metrics fallback.
VALID_METRIC_KEYS = sorted(set(get_args(MetricKey)))

METRIC_FALLBACK_SYSTEM = (
    "You map a basketball fan's stat phrasing to canonical keys. Reply with "
    "JSON {\"metrics\": [...]} using ONLY these keys: "
    + ", ".join(sorted(set(get_args(MetricKey))))
    + ". Pick 1-2 keys, most important first (e.g. 'boards' -> [\"REB\"], "
    + "'swats' -> [\"BLK\"]). "
    "If the query names no stat or you cannot map it, reply {\"metrics\": []}. "
    "No other text."
)


def _llm_resolve_metrics(
    query: str, client: Any, model: str
) -> Optional[List[str]]:
    """Ask the model which canonical stat key(s) the query wants.

    Returns None when unmappable or on any failure — the caller keeps the
    PTS default. Never raises: a helper must not break the main loop.
    """
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": METRIC_FALLBACK_SYSTEM},
                {"role": "user", "content": query},
            ],
            response_format={"type": "json_object"},
            temperature=0.0,
        )
        parsed = json.loads(resp.choices[0].message.content or "{}")
        raw = parsed.get("metrics", [])
        if isinstance(raw, str):
            raw = [raw]
        if not isinstance(raw, list):
            return None
        valid = set(VALID_METRIC_KEYS)
        out = [str(m).upper() for m in raw if str(m).upper() in valid]
        return out or None
    except Exception:
        return None


def resolve_metrics_with_fallback(
    query: str, client: Optional[Any] = None, model: Optional[str] = None
) -> List[str]:
    """Metrics for a query: deterministic match first, LLM fallback on blank.

    The resolver owns spelling variants (hyphens, digits, synonyms) — pure
    Python, no cost. Only when it matches NOTHING does the model interpret
    the phrasing (slang, novel stats), once per unique query (cached in the
    shared resolver store so viz + synthesis see the same answer). No client
    (offline/tests) or unmappable phrasing → the PTS default. The shared
    store keeps [] for unmappable (meaning "none named") rather than the
    default, so downstream gates can still tell vague from explicit.
    """
    explicit = resolver.explicit_metrics(query)
    if explicit:
        return explicit
    if query in resolver.LLM_RESOLVED_METRICS:
        return resolver.LLM_RESOLVED_METRICS[query] or ["PTS"]
    resolved: List[str] = []
    if client is not None:
        resolved = _llm_resolve_metrics(query, client, model or DEFAULT_MODEL) or []
    resolver.LLM_RESOLVED_METRICS[query] = resolved
    return resolved or ["PTS"]

SYSTEM_PROMPT = """You are an NBA stats analyst powering a data-visualization hub. \
Answer ONLY using data returned from the provided tools — never invent stats.

Rules:
1. Seasons are 'YYYY-YY' (e.g. '2025-26'). The hints tell you the resolved season — use it unless the user names another explicitly.
2. For 'compare X vs Y' call compare_players once (or one averages call per player if metrics differ).
3. For 'last N games' call the game-log tool with last_n=N.
4. For ANY career / multi-season / history question (best season ever, year by year, \
over his career, progression, most improved, totals across seasons, wins by year), \
call get_player_career_trend or get_team_history_trend ONCE — NEVER loop single-season \
tools per season. One bulk call returns the whole history.
5. per_mode: use 'Totals' only when the user says totals/combined/altogether; default 'PerGame'.
6. Fuzzy names are YOUR job, not the resolver's: it only matches exact/substring \
against the official roster. If a tool reports AMBIGUOUS (with candidates), STOP and \
ask the user to pick — do not guess. If it reports UNKNOWN (no candidates) and the \
mention looks like a nickname, short form, or misspelling, retry ONCE with the \
player's full formal name from your own knowledge (Steph -> Stephen Curry) and only \
ask the user if that retry also fails.
7. Keep the final answer to 1-2 sentences with the key numbers; the frontend renders charts from the data.
8. Regular season only unless the user says playoffs.
9. When calling tools, OMIT optional parameters you don't need — never send null. \
E.g. for full-career questions call get_player_career_trend with just {"player_name": ...}.
10. For comparisons of MULTIPLE things, use ONE bulk compare call: 'compare X vs Y \
in <season>' -> compare_players / compare_teams once; 'X vs Y throughout their \
careers / over time / by year' -> compare_player_career_trends / \
compare_team_histories once. NEVER call a single-entity trend tool once per \
entity, and NEVER loop single-season tools per season.
11. Compare at most 4 entities — if the user names more, STOP and ask which 4 matter most.
12. Scope rule (most important): an explicit single season ('in 2023-24', \
'last season', '2024-25') with NO multi-season words (career/history/over time/\
year by year/every season/best season/most improved/last N seasons/since/compare) \
means ONE season only — call get_player_season_averages / get_team_stats / \
compare_players / compare_teams, NEVER a career/history trend tool. Trend tools \
are ONLY for questions that ask about multiple seasons, careers, or history.
13. League-leader questions ('who led the NBA in assists', 'top 10 scorers', \
'most threes', 'scoring title' — ranked players, one stat, one season, NO \
named player) -> call get_league_leaders ONCE with stat_category + season + \
top_n ('who led' -> 5, 'top N' -> N). NEVER fan out per-player calls for a \
ranking, NEVER use a trend/history tool for it, and NEVER use it when the \
query names ONE specific player ('how many points did Brunson average' -> \
get_player_season_averages, even though it says 'how many').
14. Team-leader questions ('which team won the most games', 'best offense / \
defense / net rating', 'top 10 offenses' — ranked TEAMS, one stat, one season, \
NO named team) -> call get_team_leaders ONCE with stat + season + top_n. \
Stat map: wins/record -> 'W', offense -> 'PTS', defense/fewest allowed -> \
'OPP_PTS', net rating -> 'NET_RATING', offensive/defensive rating -> \
'OFF_RATING'/'DEF_RATING'. NEVER fan out per-team get_team_stats calls for a \
ranking, NEVER use the player-leaders tool for team questions, and NEVER use \
it when the query names ONE specific team ('how many threes did GSW make' -> \
get_team_stats, even though it says 'how many').
"""


def _build_user_content(
    query: str,
    today: datetime.date,
    metrics_hint: Optional[List[str]] = None,
) -> str:
    season_hint = resolver.resolve_season(query, today)
    if metrics_hint is None:
        metrics_hint = resolver.resolve_metrics(query)
    last_n_hint = resolver.extract_last_n(query)
    per_mode_hint = resolver.resolve_per_mode(query)
    window = resolver.resolve_season_window(query, today)
    trend_hint = resolver.detect_trend_intent(query)
    improvement_hint = resolver.detect_improvement_intent(query)
    comparison_hint = resolver.detect_comparison_intent(query)
    leaders_hint = resolver.detect_leaders_intent(query)
    top_n_hint = resolver.extract_top_n(query)
    team_stat_hint = resolver.resolve_team_leader_stat(query)
    team_leaders_hint = resolver.detect_team_leaders_intent(query)
    return (
        f"User query: {query}\n"
        f"[hints] today={today.isoformat()} resolved_season={season_hint} "
        f"metrics={metrics_hint} last_n={last_n_hint} per_mode={per_mode_hint} "
        f"trend={trend_hint} improvement={improvement_hint} window={window} "
        f"comparison={comparison_hint} leaders={leaders_hint} top_n={top_n_hint} "
        f"team_leaders={team_leaders_hint} team_stat={team_stat_hint}"
    )


def _clean_args(name: str, args: Dict[str, Any]) -> Dict[str, Any]:
    """Drop nulls and unknown keys before calling a tool.

    Models sometimes emit {"last_n_seasons": null} for unset optionals even
    when told to omit them; our functions treat missing as default, so strip
    Nones. Unknown keys are dropped too — never crash on a stray param.
    """
    fn = TOOL_FUNCS.get(name)
    if fn is None:
        return args
    try:
        valid = set(inspect.signature(fn).parameters)
    except (TypeError, ValueError):
        return {k: v for k, v in args.items() if v is not None}
    return {k: v for k, v in args.items() if k in valid and v is not None}


# Keys echoed back to the model per tool-result row. The model only needs
# enough to write its 1-2 sentence answer — full rows live in QueryResponse.data
# for the frontend. Career/game-log rows carry ~30 nba_api columns each, so
# echoing them whole blows past small TPM limits (e.g. Groq on-demand 8k).
_ECHO_IDENTITY_KEYS = (
    "PLAYER_NAME",
    "TEAM_NAME",
    "SEASON",
    "SEASON_ID",
    "GAME_DATE",
    "MATCHUP",
    "WL",
)
_ECHO_CONTEXT_KEYS = ("GP", "TEAM_ABBREVIATION", "W", "L", "W_PCT", "MIN")
# Max echoed rows per tool call + hard char cap on the echoed JSON.
MAX_TOOL_ECHO_ROWS = 40
MAX_TOOL_ECHO_CHARS = 4000


def _slim_tool_content(
    args: Dict[str, Any],
    result: Any,
    metrics_hint: List[str],
) -> str:
    """Serialize a tool result for the model loop, slimmed to fit TPM limits.

    List rows are projected to identity + requested-metric + context keys and
    capped at MAX_TOOL_ECHO_ROWS (most recent rows kept). Dict results (single
    rows) are small and echoed whole. Full data is always preserved in
    QueryResponse.data — this only affects model context.
    """
    if isinstance(result, list):
        wanted_metrics = args.get("metrics") or metrics_hint or ["PTS"]
        slimmed: List[Dict[str, Any]] = []
        for row in result:
            if not isinstance(row, dict) or row.get("error"):
                slimmed.append(row)
                continue
            slim: Dict[str, Any] = {}
            for key in _ECHO_IDENTITY_KEYS:
                if key in row:
                    slim[key] = row[key]
            for key in wanted_metrics:
                if key in row:
                    slim[key] = row[key]
            for key in _ECHO_CONTEXT_KEYS:
                if key in row and key not in slim:
                    slim[key] = row[key]
            slimmed.append(slim)
        note = ""
        if len(slimmed) > MAX_TOOL_ECHO_ROWS:
            dropped = len(slimmed) - MAX_TOOL_ECHO_ROWS
            if any(isinstance(r, dict) and "GAME_DATE" in r for r in slimmed):
                # Game logs arrive most-recent-first: keep the head.
                slimmed = slimmed[:MAX_TOOL_ECHO_ROWS]
                note = f" [{dropped} older games omitted; full data kept for charts]"
            else:
                # Trends arrive chronological: keep the most recent tail.
                slimmed = slimmed[-MAX_TOOL_ECHO_ROWS:]
                note = f" [{dropped} older rows omitted; full data kept for charts]"
        return json.dumps(slimmed, default=str)[:MAX_TOOL_ECHO_CHARS] + note
    return json.dumps(result, default=str)[:MAX_TOOL_ECHO_CHARS]


def _peak_row(
    data: List[Dict[str, Any]], metric: str
) -> Optional[Dict[str, Any]]:
    """Row with the max numeric value of metric (for best-season answers)."""
    best: Optional[Dict[str, Any]] = None
    best_val: Optional[float] = None
    for r in data:
        try:
            v = float(r[metric]) if r.get(metric) is not None else None
        except (TypeError, ValueError):
            v = None
        if v is None:
            continue
        if best_val is None or v > best_val:
            best_val = v
            best = r
    return best


def _synthesize_answer(
    intent: str,
    data: List[Dict[str, Any]],
    query: str,
    metrics_hint: Optional[List[str]] = None,
    highlight: Optional[str] = None,
) -> str:
    """Fallback one-liner when the LLM returns no text (should be rare)."""
    if not data:
        return "I couldn't find stats for that query — try a different player, team, or season."
    metrics_hint = metrics_hint or ["PTS"]
    try:
        if intent == "league_leaders":
            metric = metrics_hint[0] if metrics_hint else "PTS"
            ranked = sorted(data, key=lambda r: r.get("RANK") or 999)
            top = ranked[0]
            name = top.get("PLAYER_NAME", "Unknown")
            val = top.get(metric)
            season = top.get("SEASON", "")
            unit = "per game" if top.get("PER_MODE", "PerGame") == "PerGame" else "total"
            return (
                f"{name} led the NBA in {metric} ({val} {unit}) in {season} "
                f"— top {len(ranked)} shown."
            )
        if intent == "team_leaders":
            metric = metrics_hint[0] if metrics_hint else "W"
            ranked = sorted(data, key=lambda r: r.get("RANK") or 999)
            top = ranked[0]
            name = top.get("TEAM_NAME", "Unknown")
            val = top.get(metric)
            season = top.get("SEASON", "")
            return (
                f"{name} led the NBA in team {metric} ({val}) in {season} "
                f"— top {len(ranked)} shown."
            )
        if intent in ("player_career_trend", "team_history_trend"):
            metric = metrics_hint[0] if metrics_hint else "PTS"
            name = data[0].get("PLAYER_NAME") or data[0].get("TEAM_NAME") or "Team"
            peak = _peak_row(data, metric)
            seasons = [str(r.get("SEASON")) for r in data]
            span = f"{seasons[0]} to {seasons[-1]}" if len(seasons) > 1 else seasons[0]
            if highlight:
                return (
                    f"{name}'s breakout was {highlight}. "
                    f"Full {metric} trend covers {span} ({len(data)} seasons)."
                )
            if peak is not None:
                return (
                    f"{name}'s best {metric} season was {peak.get('SEASON')} "
                    f"({peak.get(metric)}). Trend covers {span} ({len(data)} seasons)."
                )
            return f"{metric} trend for {name} covers {span} ({len(data)} seasons)."
        if intent in ("player_season_avg", "compare_players", "compare_teams"):
            label = "TEAM_NAME" if intent == "compare_teams" else "PLAYER_NAME"
            bits = []
            for r in data:
                if r.get("error"):
                    bits.append(
                        f"{r.get(label, 'Unknown')} unavailable ({r['error']})"
                    )
                    continue
                name = r.get(label, "Unknown")
                if intent == "compare_teams":
                    bits.append(
                        f"{name} finished {r.get('W')}-{r.get('L')} "
                        f"({r.get('SEASON')})"
                    )
                else:
                    pts = r.get("PTS")
                    ast = r.get("AST")
                    extra = f", {ast} AST" if ast is not None else ""
                    bits.append(f"{name} averaged {pts} PPG{extra} ({r.get('SEASON')})")
            return "; ".join(bits) + "."
        if intent == "compare_trends":
            series = sorted(
                {
                    str(r.get("PLAYER_NAME") or r.get("TEAM_NAME") or "?")
                    for r in data
                    if not r.get("error")
                }
            )
            metric = metrics_hint[0] if metrics_hint else "PTS"
            seasons = sorted({str(r.get("SEASON")) for r in data if r.get("SEASON")})
            span = (
                f"{seasons[0]} to {seasons[-1]}"
                if len(seasons) > 1
                else (seasons[0] if seasons else "")
            )
            who = " vs ".join(series) if series else "entities"
            return (
                f"{metric} trends for {who} cover {span} "
                f"({len(seasons)} seasons, {len(series)} compared)."
            )
        if intent in ("player_game_logs", "team_game_logs"):
            first = data[0]
            name = first.get("PLAYER_NAME") or first.get("TEAM_NAME")
            return f"Showing the latest {len(data)} games for {name} ({first.get('SEASON_ID', '')})."
        if intent == "team_stats":
            r = data[0]
            name = r.get("TEAM_NAME", "Unknown")
            season = r.get("SEASON", "")
            # Any named non-record stat leads ("... made 1,264 FG3M ...");
            # record questions keep the W-L phrasing. Named covers
            # LLM-resolved stats too, via the shared fallback store.
            named = resolver.named_stat_metrics(query)
            stats = [m for m in named if m not in resolver.RECORD_METRICS]
            if stats:
                bits = []
                for m in stats:
                    v = r.get(m)
                    if v is None:
                        continue
                    try:
                        shown = f"{int(v):,}"
                    except (TypeError, ValueError):
                        shown = str(v)
                    bits.append(f"{shown} {m}")
                if bits:
                    return f"{name} recorded {' and '.join(bits)} in {season}."
            return (
                f"{name} finished {r.get('W')}-{r.get('L')} "
                f"({r.get('W_PCT')}) in {season}."
            )
    except Exception:
        pass
    return f"Here are the results for: {query}"


def _infer_spec(
    query: str,
    tool_names: List[str],
    tool_args: List[Dict[str, Any]],
    data: List[Dict[str, Any]],
    metrics_hint: List[str],
    season_hint: Optional[str],
    last_n_hint: int,
    per_mode_hint: str = "PerGame",
) -> QuerySpec:
    players: List[str] = []
    teams: List[str] = []
    season: Optional[str] = season_hint
    last_n: Optional[int] = None
    top_n: Optional[int] = None
    per_mode: str = per_mode_hint
    seasons: List[str] = []
    highlight_season: Optional[str] = None
    highlight_note: Optional[str] = None

    for args in tool_args:
        if not season and args.get("season"):
            season = args["season"]
        if args.get("per_mode") in ("PerGame", "Totals"):
            per_mode = args["per_mode"]
        if isinstance(args.get("players"), list):
            for p in args["players"]:
                if p not in players:
                    players.append(p)
        if isinstance(args.get("teams"), list):
            for t in args["teams"]:
                if t not in teams:
                    teams.append(t)
        if args.get("player_name") and args["player_name"] not in players:
            players.append(args["player_name"])
        if args.get("team_name") and args["team_name"] not in teams:
            teams.append(args["team_name"])
        if args.get("last_n"):
            last_n = int(args["last_n"])
        if args.get("top_n"):
            top_n = max(1, min(int(args["top_n"]), 25))

    if "compare_player_career_trends" in tool_names or "compare_team_histories" in tool_names:
        intent = "compare_trends"
        seasons = sorted({str(r.get("SEASON")) for r in data if r.get("SEASON")})
        season = seasons[-1] if seasons else season
    elif "get_player_career_trend" in tool_names and len(players) >= 2:
        # Legacy path: model called the single-entity trend tool per player.
        # Same shape as compare_player_career_trends output — route it to the
        # multi-series viz instead of collapsing to one line.
        intent = "compare_trends"
        seasons = sorted({str(r.get("SEASON")) for r in data if r.get("SEASON")})
        season = seasons[-1] if seasons else season
    elif "compare_teams" in tool_names:
        intent = "compare_teams"
    elif "get_league_leaders" in tool_names:
        intent = "league_leaders"
        # Leaderboard rows carry canonical names — collect them for the spec
        # (the query itself names no players).
        for r in data:
            name = r.get("PLAYER_NAME")
            if name and name not in players:
                players.append(str(name))
        top_n = top_n or resolver.extract_top_n(query)
        if args_season := next(
            (a.get("season") for a in tool_args if a.get("season")), None
        ):
            season = args_season
    elif "get_team_leaders" in tool_names:
        intent = "team_leaders"
        for r in data:
            name = r.get("TEAM_NAME")
            if name and name not in teams:
                teams.append(str(name))
        top_n = top_n or resolver.extract_top_n(query)
        # Metrics for team boards come from the tool's stat arg (the player
        # metric map has no team vocabulary like defense/ratings).
        stat_arg = next(
            (a.get("stat") for a in tool_args if a.get("stat")), None
        )
        if stat_arg:
            metrics_hint = [str(stat_arg).upper()]
        if args_season := next(
            (a.get("season") for a in tool_args if a.get("season")), None
        ):
            season = args_season
    elif "get_player_career_trend" in tool_names:
        intent = "player_career_trend"
        seasons = [str(r.get("SEASON")) for r in data if r.get("SEASON")]
        season = seasons[-1] if seasons else season
        # Improvement wording only: mark the largest YoY jump for annotation.
        if resolver.detect_improvement_intent(query) and data:
            metric = (metrics_hint or ["PTS"])[0]
            hs, _, note = resolver.largest_yoy_jump(data, metric)
            highlight_season, highlight_note = hs, note
    elif "get_team_history_trend" in tool_names:
        intent = "team_history_trend"
        seasons = [str(r.get("SEASON")) for r in data if r.get("SEASON")]
        season = seasons[-1] if seasons else season
        if resolver.detect_improvement_intent(query) and data:
            metric = (metrics_hint or ["W"])[0]
            key = metric if any(metric in r for r in data) else "W"
            hs, _, note = resolver.largest_yoy_jump(data, key)
            highlight_season, highlight_note = hs, note
    elif "compare_players" in tool_names or len(players) >= 2:
        intent = "compare_players"
    elif "get_player_game_logs" in tool_names:
        intent = "player_game_logs"
        last_n = last_n or last_n_hint
    elif "get_team_game_logs" in tool_names:
        intent = "team_game_logs"
        last_n = last_n or last_n_hint
    elif "get_team_stats" in tool_names:
        intent = "team_stats"
    elif not data:
        intent = "needs_clarification"
    else:
        intent = "player_season_avg"

    # Leaders guardrails (deterministic safety net behind prompt rules 13-14).
    # Two failure modes seen live, both fixed here for players AND teams:
    # (a) mixed per_modes: the model calls the tool twice (PerGame + Totals)
    # and both row sets concatenate into one chart. Keep one mode.
    # (b) single-entity misfire: "how many threes did GSW make" names ONE team
    # but got a top-N board. Slice to that entity -> snapshot intent, so the
    # viz becomes a single_stat card instead of a 10-row leaderboard.
    if intent in ("league_leaders", "team_leaders"):
        modes = {str(r.get("PER_MODE")) for r in data if r.get("PER_MODE")}
        if len(modes) > 1:
            unified = [r for r in data if str(r.get("PER_MODE")) == per_mode]
            if unified:
                data[:] = unified
        if not resolver.detect_leaders_intent(query) and not resolver.detect_team_leaders_intent(query):
            # Ranking-flavored questions ("Did GSW make the MOST threes") keep
            # the board even when they name one entity — the rank context is
            # the answer. Only plain single-entity questions downgrade.
            key = None
            if not resolver.has_ranking_language(query):
                key = "PLAYER_NAME" if intent == "league_leaders" else "TEAM_NAME"
            mentioned = resolver.find_named_entities(
                query, [str(r.get(key)) for r in data if key and r.get(key)]
            ) if key else []
            if len(mentioned) == 1:
                data[:] = [r for r in data if str(r.get(key)) == mentioned[0]]
                modes = {str(r.get("PER_MODE")) for r in data if r.get("PER_MODE")}
                if len(modes) > 1:
                    unified = [r for r in data if str(r.get("PER_MODE")) == per_mode]
                    if unified:
                        data[:] = unified
                top_n = None
                highlight_season, highlight_note = None, None
                if intent == "league_leaders":
                    intent = "player_season_avg"
                    players = mentioned
                else:
                    intent = "team_stats"
                    teams = mentioned
    # Scope guardrail (deterministic safety net behind prompt rule 12): the LLM
    # sometimes calls a trend tool for an explicit single-season question
    # ("How many threes did Curry make in 2023-24"). When the query is
    # single-season scope but a trend tool ran, slice to that season and
    # downgrade to the snapshot intent — so viz_hint becomes single_stat /
    # team_stat_card instead of a 17-season trend_line. No-op when the season
    # row is absent (never fabricate) or the query genuinely wants history.
    if intent in ("player_career_trend", "team_history_trend"):
        if resolver.is_single_season_scope(query):
            # Slice to the query's explicit season (not seasons[-1], which the
            # trend branch above overwrote to the latest data row).
            explicit = resolver.resolve_season(query)
            target = explicit or season
            scoped = [r for r in data if str(r.get("SEASON")) == target] if target else []
            if scoped:
                data[:] = scoped
                season = target
                seasons = [target] if target else []
                highlight_season, highlight_note = None, None
                if intent == "player_career_trend" and len(players) == 1:
                    intent = "player_season_avg"
                elif intent == "team_history_trend" and len(teams) == 1:
                    intent = "team_stats"

    return QuerySpec(
        intent=intent,  # type: ignore[arg-type]
        players=players,
        teams=teams,
        season=season,
        metrics=metrics_hint or ["PTS"],  # type: ignore[arg-type]
        last_n=last_n,
        top_n=top_n,
        per_mode=per_mode,  # type: ignore[arg-type]
        seasons=seasons,
        highlight_season=highlight_season,
        highlight_note=highlight_note,
        raw_query=query,
    )


def run_query(
    query: str,
    model: Optional[str] = None,
    max_iters: int = MAX_ITERS,
    today: Optional[datetime.date] = None,
    client: Optional[Groq] = None,
) -> QueryResponse:
    """Run the full agentic loop and return a frontend-ready QueryResponse."""
    today = today or datetime.date.today()
    model = model or DEFAULT_MODEL
    client = client or Groq(api_key=os.environ.get("GROQ_API_KEY"))

    # Deterministic match first; LLM fallback only when nothing matches.
    metrics_hint = resolve_metrics_with_fallback(query, client, model)
    last_n_hint = resolver.extract_last_n(query)

    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": _build_user_content(
            query, today, metrics_hint=metrics_hint)},
    ]

    traces: List[ToolCallTrace] = []
    tool_names: List[str] = []
    tool_args: List[Dict[str, Any]] = []
    data: List[Dict[str, Any]] = []
    answer_text: Optional[str] = None

    for _ in range(max_iters):
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=GROQ_TOOL_SCHEMAS,
            tool_choice="auto",
            temperature=0.0,
        )
        msg = resp.choices[0].message
        tool_calls = getattr(msg, "tool_calls", None) or []

        if not tool_calls:
            answer_text = (msg.content or "").strip() or None
            break

        # Append assistant turn (with tool calls) so the model keeps context.
        messages.append(
            {
                "role": "assistant",
                "content": msg.content or "",
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in tool_calls
                ],
            }
        )

        for tc in tool_calls:
            name = tc.function.name
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}
            args = _clean_args(name, args)
            tool_names.append(name)
            tool_args.append(args)

            fn = TOOL_FUNCS.get(name)
            if fn is None:
                content = json.dumps({"error": f"Unknown tool '{name}'."})
                traces.append(
                    ToolCallTrace(tool=name, args=args, result_summary="unknown tool")
                )
            else:
                try:
                    result = fn(**args)
                    if isinstance(result, list):
                        data.extend(result)
                        summary = f"{len(result)} rows"
                    else:
                        data.append(result)
                        summary = "1 row"
                    traces.append(
                        ToolCallTrace(tool=name, args=args, result_summary=summary)
                    )
                    content = _slim_tool_content(args, result, metrics_hint)
                except ToolError as e:
                    traces.append(
                        ToolCallTrace(
                            tool=name, args=args, result_summary=f"error: {e}"
                        )
                    )
                    content = json.dumps(
                        {"error": str(e), "candidates": e.candidates}
                    )
                except Exception as e:  # never crash the loop on bad data
                    traces.append(
                        ToolCallTrace(
                            tool=name, args=args, result_summary=f"error: {e}"
                        )
                    )
                    content = json.dumps({"error": f"Tool failed: {e}"})

            messages.append(
                {"role": "tool", "tool_call_id": tc.id, "content": content}
            )
    else:
        # Hit max_iters without a final answer — synthesize below.
        pass

    spec = _infer_spec(
        query, tool_names, tool_args, data, metrics_hint,
        resolver.resolve_season(query, today), last_n_hint,
        resolver.resolve_per_mode(query),
    )
    viz = choose_viz_hint(spec)
    if not answer_text:
        answer_text = _synthesize_answer(
            spec.intent, data, query, metrics_hint, spec.highlight_note
        )

    return QueryResponse(
        answer_text=answer_text, spec=spec, data=data, viz_hint=viz, debug=traces
    )
