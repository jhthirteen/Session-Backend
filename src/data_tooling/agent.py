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
from typing import Any, Dict, List, Optional

from groq import Groq

from . import resolver
from .models import QueryResponse, QuerySpec, ToolCallTrace
from .nba_tools import GROQ_TOOL_SCHEMAS, TOOL_FUNCS, ToolError
from .resolver import choose_viz_hint

DEFAULT_MODEL = os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b")
MAX_ITERS = 6

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
6. If a tool reports ambiguous/unknown names, STOP and ask the user to clarify — do not guess.
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
"""


def _build_user_content(query: str, today: datetime.date) -> str:
    season_hint = resolver.resolve_season(query, today)
    metrics_hint = resolver.resolve_metrics(query)
    last_n_hint = resolver.extract_last_n(query)
    per_mode_hint = resolver.resolve_per_mode(query)
    window = resolver.resolve_season_window(query, today)
    trend_hint = resolver.detect_trend_intent(query)
    improvement_hint = resolver.detect_improvement_intent(query)
    comparison_hint = resolver.detect_comparison_intent(query)
    return (
        f"User query: {query}\n"
        f"[hints] today={today.isoformat()} resolved_season={season_hint} "
        f"metrics={metrics_hint} last_n={last_n_hint} per_mode={per_mode_hint} "
        f"trend={trend_hint} improvement={improvement_hint} window={window} "
        f"comparison={comparison_hint}"
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
            return (
                f"{r.get('TEAM_NAME')} finished {r.get('W')}-{r.get('L')} "
                f"({r.get('W_PCT')}) in {r.get('SEASON')}."
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

    return QuerySpec(
        intent=intent,  # type: ignore[arg-type]
        players=players,
        teams=teams,
        season=season,
        metrics=metrics_hint or ["PTS"],  # type: ignore[arg-type]
        last_n=last_n,
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

    metrics_hint = resolver.resolve_metrics(query)
    last_n_hint = resolver.extract_last_n(query)

    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": _build_user_content(query, today)},
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
