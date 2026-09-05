# Session Notes — 2026-09-05: NLP Visualizer (Groq + nba_api)

## What we built
NLP-in → chart-ready-JSON-out backend for the NBA hub, in `src/data_tooling/` (was empty).
User flow: React `POST /api/nlp/query {query}` → Groq agentic loop → swar/nba_api
(https://github.com/swar/nba_api) → `{answer_text, spec, data, viz_hint, debug}`.

Scope voted by user: players + teams, agentic tool loop, MCP-style Python module
(not a literal MCP server), FastAPI + viz hints.

## Checkpoints (all completed, all live-tested)
1. **models.py + resolver.py** — `QuerySpec`, `VizHint` (single_stat /
   comparison_bars / time_series / game_log_table / team_stat_card),
   `QueryResponse`. Deterministic season/metric/name normalization, no LLM.
   Season rule: Sep 2026 offseason ⇒ "last year" = `2025-26`. Team aliases
   (Pels, Wolves, BOS…) included. Ambiguous names return candidates, never guess.
2. **nba_tools.py** — 5 tools + Groq schemas: `get_player_season_averages`,
   `get_player_game_logs`, `get_team_stats`, `get_team_game_logs`,
   `compare_players`. 30s timeout, 1 retry, 1-hr TTL cache, `ToolError` instead
   of hallucinations. Lazy `nba_api` imports.
3. **agent.py** — `run_query()` Groq loop (default `openai/gpt-oss-120b`, same as
   `news.py`; `GROQ_MODEL` override), max 6 iters, temp 0. Pre-injects
   resolved season/metrics/last_n hints. Tool errors feed back so Groq asks for
   clarification. Infers `QuerySpec` from actual calls; synthesizes fallback answer.
4. **api.py** — `POST /api/nlp/query` → `QueryResponse`, `GET /api/nlp/health`.
   Exposes `router` (mount via `include_router`) + standalone `app`/`create_app()`.

Also wrote `src/data_tooling/V1_SCOPE_AND_GAPS.txt` (handled intents, out-of-scope
list, hardcoded assumptions, gray zone) at user's request — no code changes with it.

## Test results (live Groq + NBA API)
- Brunson 2024-25: 65 GP, 26.0 PTS / 7.3 AST / 2.9 REB ✅ (matches known stats)
- Brunson 2025-26: 74 GP, 26.0 PTS ✅ | Celtics 2024-25: 61-21 (.744) ✅
- Celtics 2025-26: 56-26 (.683) ✅
- Compare 2024-25 Brunson vs Haliburton: 26.0/7.3 vs 18.6/9.2 ✅
- Haliburton 2025-26 correctly missing (injury-missed season) → partial-row fix ✅
- Game-log last-N, team logs, multi-metric (AST+REB), unknown player
  ("Mickey Mouse" → `needs_clarification`, zero rows, no hallucination) ✅
- `QueryResponse` JSON-serializes cleanly; repeat queries hit TTL cache ✅
- API: health 200, empty/missing query 422, live POST 200 ✅

## Learnings (the non-obvious stuff)
1. **Game logs are reverse-chronological.** `PlayerGameLog`/`TeamGameLog` return
   most-recent-first, so `.tail(n)` silently returns the *oldest* games
   (caught live: returned Oct 2024 instead of Apr 2025). Fixed to `.head(n)`.
2. **TeamYearByYearStats uses WINS/LOSSES/WIN_PCT**, not W/L/W_PCT. Aliased in
   `get_team_stats` so players and teams share one key set. Always print raw
   columns before assuming names.
3. **Python version pin matters.** Repo `.venv` is Python 3.9.13; nba_api ≥1.11
   requires ≥3.10. User installed `nba_api 1.10.2` — correct call, all endpoints
   used work on it. Upgrading Python unlocks latest nba_api later.
4. **Groq infers famous names.** Bare "brunson" → Groq passed "Jalen Brunson" to
   the tool, bypassing our Jalen-vs-Rick disambiguation. Good UX, softens
   "never guess" — open question whether to enforce strict pass-through.
   (Direct tool call with "Brunson" still correctly raises ambiguous.)
5. **Partial failure > total failure for comparisons.** First version of
   `compare_players` raised when *one* player missed a season; now keeps
   `{..., "error": ...}` rows so React renders available bars + a missing note.
6. **Unknown entities need no tool call.** Groq answered "Mickey Mouse" with
   clarification and zero tool calls — the system prompt's "don't guess" rule
   holds even without tool feedback.
7. **Hints beat guessing.** Injecting `resolved_season/metrics/last_n` into the
   user message eliminated an entire class of season-format errors; Groq used
   `2025-26` correctly every run.
8. **No installs without approval works fine.** All env changes (nba-api/pandas,
   Groq key) were user-side; code degrades to clear errors
   (`ToolError: nba_api is not installed…`) when deps are missing thanks to lazy
   imports — keep that pattern.
9. **`news.py` conventions carried over well.** Same Groq client init
   (`GROQ_API_KEY` env), same model default, Pydantic everywhere. Consistency
   made review fast.
10. **The toolbelt is the scope.** Anything outside the 5 tools gets a
    clarification or a wrong-shaped answer — no amount of prompting fixes that.
    Next cheapest expansion: `compare_teams` (mirrors existing fan-out).

## Open questions / next steps
- Strict vs lenient last-name handling (see learning #4) — user's call.
- Next milestone vote: `compare_teams` → leaders/standings → box scores →
  multi-season trends (see `V1_SCOPE_AND_GAPS.txt`).
- Frontend: React components keyed on `viz_hint.type`; `debug` array available
  for a tool-call trace view.
- Ops: persistent cache, rate limiting, `uvicorn` in `.venv` (needs approval).

---

## Continued session (same day) — what happened after Checkpoint 4

### Checkpoint 4 shipped (`api.py`)
- `POST /api/nlp/query` → `QueryResponse`, `GET /api/nlp/health`, `router` for
  mounting + standalone `app`/`create_app()`. Verified: health 200, empty query
  422, live POST 200 (Brunson 26.0 PPG).
- `uvicorn` confirmed present in `.venv`; serve with
  `.venv/bin/uvicorn src.data_tooling.api:app --reload --port 8000`.

### Interactive playground (`src/data_tooling/test.py`)
- REPL: prompts at `ask>`, prints answer + spec + viz hint + data rows + tool
  trace. Run as module: `.venv/bin/python -m src.data_tooling.test`.
- First real user query through it exposed the 6-iteration cap: Brunson
  "highest PPG season" looped one season per call (2018-19 → 2023-24), hit
  `MAX_ITERS=6`, and the fallback synthesizer listed rows with no real answer.
  Root cause was a missing capability, not a bug — led to the trend expansion.

### History-trend expansion (7 intents now)
- New: `player_career_trend` + `team_history_trend` (one bulk API call each —
  the endpoints already returned all seasons; we just stopped filtering to one
  row), `trend_line` viz (chartable per-game AND totals, per user vote),
  `per_mode` / `seasons` / `highlight_season` / `highlight_note` spec fields.
- Most-improved wording → max single-season YoY delta as an annotation
  (`highlight_*`); general trend queries return the full trend unmarked.
  Traded mid-season years collapse to the TOT row (verified: Durant 2022-23).
- The failing Brunson query now answers in 1 call: peak 28.7 PPG 2023-24.
- `V1_SCOPE_AND_GAPS.txt` updated: trends moved from gap → capability.

### Bug: Groq 400 on explicit nulls (Luka Dončić query)
- Symptom: `get_player_career_trend` with `"last_n_seasons": null` →
  `BadRequestError 400 tool_use_failed`. Cause: Groq validates tool calls
  server-side; optional params were strict `integer`/`string`. Fix, 3 layers:
  nullable unions on ALL optional params (also preempted identical flaw in
  `last_n`, `metrics`), prompt rule 9 (omit, never null), `_clean_args()`
  sanitizer in `agent.py`. Regression tests in
  `tests/data_tooling/test_agent_offline.py`. Live-verified: Luka +12.3
  (33.5 vs 21.2 rookie).
- Learning: with agentic tool loops, the schema is a contract with the
  *provider's validator*, not just your code — declare every optional as
  nullable even when your function defaults handle None.

### Accent-insensitive name matching
- `fold_accents()` (NFKD → strip marks → casefold) applied to both mention and
  candidates in `resolve_player`/`resolve_team`. "Luka Doncic" now resolves
  first try (was: fail → LLM self-correct → retry). Covers Jokić, Vučević, etc.
  Fold collisions surface as ambiguous candidates per the never-guess rule.

### Test suite (`tests/data_tooling/`, top-level as requested)
- `conftest.py` (sys.path), `test_resolver.py` (27 tests), `test_models.py` (5),
  `test_agent_offline.py` (4), `test_nba_tools_live.py` (8, `RUN_LIVE=1`),
  `test_agent_live.py` (4, `RUN_LIVE_AGENT=1` + key), `test_api_contract.py`
  (health/422 offline + gated live POST). Live tests assert stable facts
  (2024-25 numbers, TOT collapse), never LLM wording.
- Note: suite was deleted once by accident and rebuilt from scratch; all green.
- Still blocked: `pytest` not in `.venv` — needs user-approved
  `.venv/bin/pip install pytest` before the suite can execute.

### Standing rules reaffirmed this session
- No `pip install` / env changes without explicit user approval (all installs —
  nba-api/pandas, Groq key — were user-side).
- Lazy `nba_api` imports preserved so modules import cleanly without deps.

