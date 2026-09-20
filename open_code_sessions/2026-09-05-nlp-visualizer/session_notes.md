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

---

## Continued session 2026-09-06 — comparisons + TPM fix (backend half)

Frontend repo (`Session-Frontend`) was scaffolded separately (Vite + React 19 +
recharts) with a `HANDOFF.md` contract; `api.py` gained CORS (`frontend_origins()`,
`FRONTEND_ORIGINS` env) to serve it — that change predates this session's work.

### Comparison framework (10 intents now)
User's screenshot: "Tatum vs Brown PPG throughout careers" rendered ONE jumbled
line. Root cause: `get_player_career_trend` is single-entity, so the agent called
it twice and `_infer_spec` collapsed the concatenated rows to
`player_career_trend`. Fixed on both sides of the contract:
- New tools (`nba_tools.py`): `compare_player_career_trends`,
  `compare_team_histories` (entity×time matrix, rows tagged `PLAYER_NAME` /
  `TEAM_NAME`), `compare_teams` (team snapshot — closed the gap noted in
  learning #10). `MAX_COMPARE_ENTITIES = 4`; over-cap → `ToolError` so the
  agent asks to narrow down. Partial-failure `error` rows preserved per entity.
- New intents `compare_trends` / `compare_teams` + `VizHint.series_key`
  (`PLAYER_NAME` / `TEAM_NAME` / `SEASON`, null = legacy single series) and
  `multi_trend` viz type — all additive, old payloads byte-identical.
- `_infer_spec`: new tools route to the new intents; the legacy path (two
  single-entity trend calls) ALSO routes to `compare_trends` with union seasons.
  `teams` list now collected from `teams` tool args (previously only
  `team_name`). `_synthesize_answer` covers both new intents.
- System prompt rules 10–11 (one bulk compare call, never per-entity/per-season
  loops, max 4 entities) + `comparison=` hint in `_build_user_content` via new
  `resolver.detect_comparison_intent()`.
- `choose_viz_hint`: `compare_trends` → `multi_trend` (x=`SEASON`); `compare_teams`
  → `comparison_bars` (x=`TEAM_NAME`); `compare_players` hint now carries
  `x_key`/`series_key` for the frontend's generalized bars.
- `V1_SCOPE_AND_GAPS.txt` updated (10 intents; team-compare gap closed; noted
  multi-entity last-N-games overlay still has no dedicated tool — frontend
  pivots it via series detection).
- Tests: `tests/data_tooling/test_compare_offline.py` (15 tests: caps, fan-out,
  partial failure, spec routing incl. legacy path, viz hints, synthesis).
  Full offline suite: 53 passed, 1 skipped (pre-existing skip).

### Groq 413 TPM fix (LeBron vs KD careers: requested 8507 > 8000 limit)
Two stacked causes: 3 new schemas added per-request tokens, but the real killer
was the tool-result echo — ~40 career rows × ~30 raw nba_api columns ≈ 21k
chars (~5.3k tokens) fed back just so the model could write one sentence.
- New `_slim_tool_content()` in `agent.py`: list rows projected to identity +
  requested-metric + context (`GP`, team, W/L) keys, capped at 40 rows
  (`MAX_TOOL_ECHO_ROWS`) keeping the most recent (game logs keep head, trends
  keep tail), hard-capped at 4000 chars with an omission note. `error` rows pass
  through; dict (single-row) results echo whole. `QueryResponse.data` is
  untouched — charts lose nothing.
- Trimmed verbose tool-schema descriptions (sent on every loop iteration).
- Measured on a LeBron+KD-shaped payload: echo 21,040 → 4,000 chars
  (~4.2k tokens saved; request ≈8507 → ≈4200). 5 new tests in
  `test_agent_offline.py`. Suite: 58 passed, 1 skipped.
- Learning: with small TPM tiers, the model context is a budget — echo
  summaries, never raw bulk rows. Charts read `data`, the model only needs
  answer-sized evidence.
- Not live-verified (no `GROQ_API_KEY` in this shell); user to retry the query.


---

## Continued session 2026-09-07 — viz robustness + leaders + team stats (backend half)

All work on branch `feature/the_answer` (note: `vizualization_cleanup` was
merged into it via PR #6 early in the session).

### Viz-selection robustness (Curry 3PM screenshot: career trend for a single season)
- `resolve_metrics()` was substring matching: `"point" in "three pointers"` →
  `["PTS","FG3M"]`. Now word-boundary matching with overlap suppression
  (longest phrase claims its span): `three pointers → [FG3M]`,
  `three point percentage → [FG3_PCT]` only. Added hyphen/space `3pt` variants.
- `resolve_per_mode()` was Totals-only on `totals/combined/...`. Now verb-aware:
  per-game language (average/per game/ppg) wins; `how many ... made/hit/...`
  → Totals. Curry query → Totals, Brunson-average → PerGame.
- Prompt rule 12 (explicit single season + no trend words → single-season tool,
  never trend) + tightened trend-tool descriptions.
- Deterministic guardrail in `_infer_spec`: trend tool + `is_single_season_scope()`
  (new `has_explicit_season()`) → slice data to the query season, downgrade to
  `player_season_avg`/`team_stats`. Passes through when the season row is absent
  (never fabricate). Caught own off-by-one in testing (sliced to `seasons[-1]`
  instead of `resolve_season(query)`).
- New `tests/data_tooling/test_scope_guardrail.py` (13 tests).

### League leaders (new intent #11: `league_leaders`, viz `leaderboard`)
- New tool `get_league_leaders(stat_category, season, top_n, per_mode)` — one
  `LeagueLeaders` call, RANK-sorted rows (`PLAYER_NAME/TEAM_ABBREVIATION/RANK/GP`
  + stat), top_n clamped 1–25 (`MAX_LEADERBOARD_ROWS`). PerGame default per
  title convention. Live-verified: 2024-25 AST Trae 11.6 → Harden 8.7 (top 5);
  PTS SGA 32.7 → Lillard 24.9 (top 10).
- Resolver: `detect_leaders_intent()` (led/leaders/top-N/scoring-title; "most"
  deliberately left to the LLM — overlaps peak wording), `extract_top_n()`
  (default 5 for bare "led"), `scorer/scorers/scoring → PTS`.
- Contract (additive): `Intent += league_leaders`, `VizType += leaderboard`,
  `QuerySpec.top_n`. Agent rule 13 + `leaders=`/`top_n=` hints + routing +
  synthesis ("Trae Young led the NBA in AST (11.6 per game) ...").
- New `tests/data_tooling/test_leaders_offline.py` (15 tests).

### Team leaders (new intent #12: `team_leaders`, shared `leaderboard` viz)
- New tool `get_team_leaders(stat, season, top_n, per_mode)` — one
  `LeagueDashTeamStats` call for all 30 teams (fan-out of 30× get_team_stats
  rejected as slow/rate-hostile). Stat→(measure, column, direction) table:
  Base (W/PTS/FG3M/...), Opponent (`OPP_PTS` ascending — best defense),
  Advanced (`OFF/DEF/NET_RATING`, columns verified live before coding).
  Live-verified 2024-25: wins OKC 68, offense Cavs 121.9, defense ORL 105.5
  (plan guessed OKC — wrong, good thing we check), net OKC +12.7.
- Resolver: separate team vocab table (`best record/most wins → W`,
  `offense → PTS`, `defense → OPP_PTS`, ratings), `detect_team_leaders_intent()`
  requires ranking words + team nouns AND vetoes any specific-team alias mention
  ("celtics record" stays `team_stats` — precision over recall).
- Reuses `leaderboard` viz with `x_key=TEAM_NAME`; `MetricKey += PLUS_MINUS,
  OPP_PTS, OFF/DEF/NET_RATING` (first test run caught the Literal gap).
  Agent rule 14 + `team_leaders=`/`team_stat=` hints + routing (metrics from
  tool `stat` arg — player map has no defense/ratings words).
- New `tests/data_tooling/test_team_leaders_offline.py` (19 tests).
- `V1_SCOPE_AND_GAPS.txt` updated (12 intents; fixed stale "5 tools" line).

### Team single-stat generality (GSW 1264-threes query)
- Diagnosis: data + hints were correct; gaps were downstream — team_stats
  synthesis hardcoded W-L phrasing, viz always chose the record card, frontend
  StatCard assumed player rows.
- New `explicit_metrics()` (named stats vs PTS default) + `RECORD_METRICS`:
  team_stats + explicit non-record stat → `single_stat`, else record card
  (vague "how did the celtics do" still cards). Synthesis leads with named
  stats ("recorded 1,264 FG3M"), record phrasing preserved otherwise.
- New `tests/data_tooling/test_team_stats_general.py` (11 tests).

### Leaders guardrails (GSW leaderboard screenshot: wrong intent + doubled rows)
- Screenshot showed `team_leaders` top-10 for one named team AND PerGame+Totals
  row sets concatenated (Boston 17.8 + 1,457 in one chart) — LLM double-call.
- `_infer_spec` leaders guardrails (players + teams): unify mixed PER_MODE row
  sets to one mode; single-entity misfire (exactly one data name mentioned,
  no ranking intent) → slice + downgrade to `player_season_avg`/`team_stats`.
- Overfit caught by audit: "Did the Warriors make the MOST threes" was wrongly
  downgraded (ranking context IS the answer). Added `has_ranking_language()`
  gate (most/best/top/led...; excludes "last" and "win/won"). 5-scenario matrix
  verified: plain single-entity → card; ranking-flavored → board kept.
- Prompt rules 13/14 gained NEVER clauses with the exact failing examples.
- New `find_named_entities()` (full name or last token, word-boundary,
  data-driven — no hardcoded names).

### Suite status
- Offline: 126 passed, 13 skipped (skips are live/gated). No GROQ_API_KEY in
  this shell — LLM end-to-end (tool choice for leaders/team-stat queries) left
  for the user's keyed shell throughout.

---

## Continued session 2026-09-08 — messy NLP: stat normalization + LLM fallbacks

Branch `feature/the_answer`. Trigger: "Show me Steph Curry's 3-point
percentage vs Luka Doncic's over the last 5 seasons" failed twice over —
`3-point percentage` parsed as PTS ("point" substring hit), "Steph Curry"
resolved to nobody.

### Stat spelling normalization (deterministic, general)
- New `_normalize_stat_text()`: hyphens between word chars become spaces
  before synonym matching, so ONE entry covers `3-point` / `3 point` /
  `three-point` / `free-throw` / `field-goal` spellings across all families.
- Digit-form synonyms: `3 point percentage → FG3_PCT`, `3 point/3 pointers →
  FG3M`, plus `trey/treys` slang. Deliberate exception: bare `3 points` stays
  PTS-quantity ("scored 3 points") — wrong-stat chart is worse than default.
- The example now parses fully: FG3_PCT + comparison + last-5 window.

### Names: nickname map added, then REVERTED (learning)
- First attempt was a 25-entry PLAYER_NICKNAMES table. User correctly called it
  overfitting: it duplicates model knowledge and rots with trades/rookies.
- Final design — resolver owns spelling (exact/substring only), model owns
  meaning: unknown-name ToolErrors invite ONE retry with the full formal name,
  prompt rule 6 split (ambiguous+canididates → ask user; unknown → retry formal
  name, clarify only if that fails). Zero name lists, works for any nickname
  including post-cutoff players the map could never hold.

### LLM metrics fallback (+ propagation fix)
- `resolve_metrics_with_fallback()`: deterministic `explicit_metrics()` first
  (zero cost); only blanks hit the model — one strict-JSON call (temp 0,
  constrained to valid MetricKeys), cached, garbage/exceptions → PTS default.
- Audit caught incomplete wiring: viz gate + synthesis re-derived from
  synonyms, so model-resolved stats still rendered record cards. Fixed with
  single shared store `resolver.LLM_RESOLVED_METRICS` (stores [] for
  unmappable so vague stays distinguishable) + `named_stat_metrics()` reader
  used by BOTH downstream consumers. Agent-local duplicate cache removed.
- `V1_SCOPE_AND_GAPS.txt` gray-zone updated (fallback replaces silent default).

### Suite status
- Offline: 149 passed, 13 skipped. New `test_messy_nlp.py` (stat forms +
  fail-clean names) and `test_metric_fallback.py` (stubbed client: blank→LLM,
  invalid/empty/exception→default, caching, hint override, viz/synth
  propagation). No GROQ_API_KEY here — live fallback + name-retry e2e left
  for the keyed shell.
