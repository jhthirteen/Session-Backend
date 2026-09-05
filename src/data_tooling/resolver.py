"""Deterministic normalization for the NLP visualizer.

No LLM calls here on purpose — Groq decides *what* to fetch, but season
strings, metric synonyms, and name matching are resolved with pure Python
so results are reproducible and testable.

nba_api is imported lazily so this module works even before
`pip install nba-api` (falls back to injected lists — see tests).
"""

import datetime
import re
import unicodedata
from functools import lru_cache
from typing import Dict, List, Optional, Tuple

from .models import QuerySpec, VizHint


def fold_accents(text: str) -> str:
    """Strip diacritics + casefold: 'Dončić' -> 'doncic', 'José' -> 'jose'.

    Applied to BOTH the user mention and candidate names so non-ASCII names
    match regardless of which side has accents.
    """
    ascii_only = unicodedata.normalize("NFKD", text).encode("ascii", "ignore")
    return ascii_only.decode("ascii").casefold().strip()

# ---------------------------------------------------------------------------
# Season resolution
# ---------------------------------------------------------------------------
# nba_api season format: "2024-25" (season starting Oct 2024, ending Jun 2025).

_SEASON_RE = re.compile(r"^(\d{4})-(\d{2})$")
_YEAR_ONLY_RE = re.compile(r"\b(19\d{2}|20\d{2})\b")
_SHORT_SEASON_RE = re.compile(r"\b(\d{2})-(\d{2})\b")  # e.g. "24-25"
_LAST_N_RE = re.compile(r"\b(?:last|past|previous)\s+(\d{1,2})\s+(?:games?)\b", re.I)


def most_recent_completed_season(today: Optional[datetime.date] = None) -> str:
    """Return the most recently *completed or current* season in 'YYYY-YY'.

    NBA calendar: season YYYY-YY runs Oct YYYY -> Jun YYYY+1.
      - Oct-Dec YYYY       -> f"{YYYY}-{YY+1}" (season just started)
      - Jan-Jun YYYY+1     -> f"{YYYY}-{YY+1}" (season in progress / playoffs)
      - Jul-Sep YYYY+1     -> f"{YYYY}-{YY+1}" (offseason, that season completed)

    Examples:
      - 2026-09-05 (today) -> "2025-26"
      - 2025-02-10         -> "2024-25"
      - 2024-11-01         -> "2024-25"
    """
    today = today or datetime.date.today()
    y, m = today.year, today.month
    if m >= 10:
        start = y
    elif m >= 7:
        start = y - 1  # offseason: last completed season
    else:
        start = y - 1  # Jan-Jun: current season started last calendar year
    return f"{start}-{str(start + 1)[2:]}"


def resolve_season(
    text: str, today: Optional[datetime.date] = None
) -> Optional[str]:
    """Map a natural-language season reference to 'YYYY-YY'.

    Returns None only when the text references something unresolvable
    without extra context (e.g. 'rookie year' with no player career lookup).
    Callers should then ask Groq/tools to resolve or request clarification.
    """
    today = today or datetime.date.today()
    default = most_recent_completed_season(today)
    t = text.lower().strip()

    # Explicit phrases first.
    if any(p in t for p in ("last year", "last season", "past season", "previous season")):
        return default
    if any(p in t for p in ("this season", "current season", "this year")):
        return default
    if "rookie year" in t or "rookie season" in t:
        return None  # needs player career lookup — do not guess

    # Explicit "2024-25" style (tolerates spaces: "2024 - 25").
    compact = re.sub(r"\s+", "", t)
    m = re.search(r"(\d{4})-(\d{2})", compact)
    if m:
        return f"{m.group(1)}-{m.group(2)}"

    # Short "24-25" style -> "2024-25".
    m = _SHORT_SEASON_RE.search(t)
    if m:
        return f"20{m.group(1)}-{m.group(2)}"

    # Bare year "2024" -> season *starting* that year ("2024-25").
    # Documented assumption: avoids Jan-Jun ambiguity; explicit form wins.
    m = _YEAR_ONLY_RE.search(t)
    if m:
        yyyy = int(m.group(1))
        if 2000 <= yyyy <= today.year + 1:
            return f"{yyyy}-{str(yyyy + 1)[2:]}"

    # No season mentioned -> default (most recent). This matches the
    # "how many points did X average last year" UX where users omit it.
    return default


def extract_last_n(text: str, default: int = 10) -> int:
    """Extract 'last N games' window; defaults to 10 for game-log intents."""
    m = _LAST_N_RE.search(text)
    if m:
        try:
            n = int(m.group(1))
            return max(1, min(n, 82))
        except ValueError:
            return default
    return default


# ---------------------------------------------------------------------------
# Metric synonyms -> canonical keys
# ---------------------------------------------------------------------------
_METRIC_SYNONYMS: Dict[str, str] = {
    # points
    "point": "PTS",
    "points": "PTS",
    "ppg": "PTS",
    "scored": "PTS",
    # assists
    "assist": "AST",
    "assists": "AST",
    "apg": "AST",
    "dimes": "AST",
    # rebounds
    "rebound": "REB",
    "rebounds": "REB",
    "rpg": "REB",
    "boards": "REB",
    # stocks
    "steal": "STL",
    "steals": "STL",
    "spg": "STL",
    "block": "BLK",
    "blocks": "BLK",
    "bpg": "BLK",
    # shooting
    "field goal percentage": "FG_PCT",
    "fg%": "FG_PCT",
    "field goal": "FG_PCT",
    "three": "FG3M",
    "threes": "FG3M",
    "3pm": "FG3M",
    "three point percentage": "FG3_PCT",
    "3p%": "FG3_PCT",
    "free throw percentage": "FT_PCT",
    "ft%": "FT_PCT",
    "minutes": "MIN",
    "mins": "MIN",
    "mpg": "MIN",
    # team record
    "win": "W",
    "wins": "W",
    "loss": "L",
    "losses": "L",
    "record": "W",
    "win percentage": "W_PCT",
    "win pct": "W_PCT",
}


def resolve_metrics(text: str) -> List[str]:
    """Extract canonical metric keys from free text. Defaults to ['PTS']."""
    t = text.lower()
    found: List[str] = []
    # Multi-word phrases first so "field goal percentage" wins over "field goal".
    for phrase in sorted(_METRIC_SYNONYMS, key=len, reverse=True):
        if phrase in t:
            key = _METRIC_SYNONYMS[phrase]
            if key not in found:
                found.append(key)
    return found or ["PTS"]


# ---------------------------------------------------------------------------
# Name resolution (players + teams)
# ---------------------------------------------------------------------------
TEAM_ALIASES: Dict[str, str] = {
    # abbreviation / city / nickname -> canonical full name
    "atl": "Atlanta Hawks",
    "hawks": "Atlanta Hawks",
    "atlanta": "Atlanta Hawks",
    "bos": "Boston Celtics",
    "celtics": "Boston Celtics",
    "boston": "Boston Celtics",
    "bkn": "Brooklyn Nets",
    "nets": "Brooklyn Nets",
    "brooklyn": "Brooklyn Nets",
    "cha": "Charlotte Hornets",
    "hornets": "Charlotte Hornets",
    "charlotte": "Charlotte Hornets",
    "chi": "Chicago Bulls",
    "bulls": "Chicago Bulls",
    "chicago": "Chicago Bulls",
    "cle": "Cleveland Cavaliers",
    "cavs": "Cleveland Cavaliers",
    "cavaliers": "Cleveland Cavaliers",
    "cleveland": "Cleveland Cavaliers",
    "dal": "Dallas Mavericks",
    "mavs": "Dallas Mavericks",
    "mavericks": "Dallas Mavericks",
    "dallas": "Dallas Mavericks",
    "den": "Denver Nuggets",
    "nuggets": "Denver Nuggets",
    "denver": "Denver Nuggets",
    "det": "Detroit Pistons",
    "pistons": "Detroit Pistons",
    "detroit": "Detroit Pistons",
    "gsw": "Golden State Warriors",
    "warriors": "Golden State Warriors",
    "golden state": "Golden State Warriors",
    "hou": "Houston Rockets",
    "rockets": "Houston Rockets",
    "houston": "Houston Rockets",
    "ind": "Indiana Pacers",
    "pacers": "Indiana Pacers",
    "indiana": "Indiana Pacers",
    "lac": "LA Clippers",
    "clippers": "LA Clippers",
    "lal": "Los Angeles Lakers",
    "lakers": "Los Angeles Lakers",
    "los angeles lakers": "Los Angeles Lakers",
    "mem": "Memphis Grizzlies",
    "grizzlies": "Memphis Grizzlies",
    "memphis": "Memphis Grizzlies",
    "mia": "Miami Heat",
    "heat": "Miami Heat",
    "miami": "Miami Heat",
    "mil": "Milwaukee Bucks",
    "bucks": "Milwaukee Bucks",
    "milwaukee": "Milwaukee Bucks",
    "min": "Minnesota Timberwolves",
    "wolves": "Minnesota Timberwolves",
    "timberwolves": "Minnesota Timberwolves",
    "minnesota": "Minnesota Timberwolves",
    "nop": "New Orleans Pelicans",
    "pels": "New Orleans Pelicans",
    "pelicans": "New Orleans Pelicans",
    "new orleans": "New Orleans Pelicans",
    "nyk": "New York Knicks",
    "knicks": "New York Knicks",
    "new york": "New York Knicks",
    "okc": "Oklahoma City Thunder",
    "thunder": "Oklahoma City Thunder",
    "oklahoma city": "Oklahoma City Thunder",
    "orl": "Orlando Magic",
    "magic": "Orlando Magic",
    "orlando": "Orlando Magic",
    "phi": "Philadelphia 76ers",
    "sixers": "Philadelphia 76ers",
    "76ers": "Philadelphia 76ers",
    "philadelphia": "Philadelphia 76ers",
    "phx": "Phoenix Suns",
    "suns": "Phoenix Suns",
    "phoenix": "Phoenix Suns",
    "por": "Portland Trail Blazers",
    "blazers": "Portland Trail Blazers",
    "portland": "Portland Trail Blazers",
    "sac": "Sacramento Kings",
    "kings": "Sacramento Kings",
    "sacramento": "Sacramento Kings",
    "sas": "San Antonio Spurs",
    "spurs": "San Antonio Spurs",
    "san antonio": "San Antonio Spurs",
    "tor": "Toronto Raptors",
    "raptors": "Toronto Raptors",
    "toronto": "Toronto Raptors",
    "uta": "Utah Jazz",
    "jazz": "Utah Jazz",
    "utah": "Utah Jazz",
    "was": "Washington Wizards",
    "wizards": "Washington Wizards",
    "washington": "Washington Wizards",
}


@lru_cache(maxsize=1)
def _load_static_players() -> List[Dict]:
    from nba_api.stats.static import players  # type: ignore

    return players.get_players()


@lru_cache(maxsize=1)
def _load_static_teams() -> List[Dict]:
    from nba_api.stats.static import teams  # type: ignore

    return teams.get_teams()


def resolve_player(
    mention: str, all_players: Optional[List[Dict]] = None
) -> Tuple[Optional[str], List[str]]:
    """Resolve a player mention to canonical 'First Last'.

    Returns (canonical_name_or_None, candidates).
    - Exact (case-insensitive) match -> (name, [])
    - One substring match         -> (name, [])
    - Multiple matches            -> (None, [up to 5 candidates]) — caller
      must ask for clarification, never guess.
    - No match                    -> (None, [])
    """
    if all_players is None:
        try:
            all_players = _load_static_players()
        except Exception:
            return None, []  # nba_api not installed — caller injects list in tests

    q = mention.strip().lower()
    if not q:
        return None, []
    qf = fold_accents(mention)

    folded = {p["full_name"]: fold_accents(p["full_name"]) for p in all_players}
    exact = [name for name, f in folded.items() if f == qf]
    if len(exact) == 1:
        return exact[0], []

    contains = [name for name, f in folded.items() if qf in f]
    if len(contains) == 1:
        return contains[0], []
    if len(contains) > 1:
        # Prefer matches where the query equals first or last name exactly —
        # e.g. "Brunson" should win over "Jalen Brunson" vs others cleanly
        # only when unambiguous; otherwise surface candidates.
        return None, sorted(contains)[:5]
    return None, []


def resolve_team(
    mention: str, all_teams: Optional[List[Dict]] = None
) -> Tuple[Optional[str], List[str]]:
    """Resolve a team mention to canonical full name (aliases + static list)."""
    q = mention.strip().lower()
    if not q:
        return None, []
    qf = fold_accents(mention)
    if qf in TEAM_ALIASES:
        return TEAM_ALIASES[qf], []

    if all_teams is None:
        try:
            all_teams = _load_static_teams()
        except Exception:
            # Fall back to alias values only.
            cands = sorted({v for k, v in TEAM_ALIASES.items() if qf in fold_accents(k)})
            if len(cands) == 1:
                return cands[0], []
            return None, cands[:5]

    names = [t["full_name"] for t in all_teams]
    folded = {n: fold_accents(n) for n in names}
    exact = [n for n in names if folded[n] == qf]
    if len(exact) == 1:
        return exact[0], []
    contains = [n for n in names if qf in folded[n]]
    if len(contains) == 1:
        return contains[0], []
    if contains:
        return None, sorted(contains)[:5]
    return None, []


# ---------------------------------------------------------------------------
# Trend detection + season windows + per-mode (history-trend ability)
# ---------------------------------------------------------------------------
_TREND_RE = re.compile(
    r"\b(career|history|historical|over time|year\s*by\s*year|year\s*over\s*year"
    r"|\byoy\b|every season|all seasons|by season|progression|evolved?|evolution"
    r"|trend|across (his|her|their) career|over (his|her|their) career"
    r"|entire career|whole career|season\s*by\s*season)\b",
    re.I,
)
_PEAK_RE = re.compile(
    r"\b(highest|best|most|greatest|peak|career.?high)\b.{0,40}\b(season|year|ppg|average|avg)\b"
    r"|\b(season|year)\b.{0,40}\b(highest|best|most|peak)\b",
    re.I,
)
_IMPROVEMENT_RE = re.compile(
    r"\bmost improved\b|\bbiggest (leap|jump|improvement)\b"
    r"|\blargest (increase|jump|improvement)\b|\bbreakout (season|year)\b",
    re.I,
)
_LAST_N_SEASONS_RE = re.compile(
    r"\b(?:last|past|previous|final)\s+(\d{1,2})\s+(?:seasons?|years?)\b", re.I
)
_SINCE_SEASON_RE = re.compile(
    r"\bsince\s+((?:19|20)\d{2}(?:-\d{2})?)\b", re.I
)
_TOTALS_RE = re.compile(
    r"\btotals?\b|\bcombined\b|\baltogether\b|\bcumulative\b", re.I
)
_COMPARISON_RE = re.compile(
    r"\bcompar(e|ison|ing)\b|\bvs\.?\b|\bversus\b|\bbetween\b"
    r"|\bbetter\b|\bhead.to.head\b|\bhead to head\b|\bwho('s| is) (better|greater)\b",
    re.I,
)


def detect_comparison_intent(text: str) -> bool:
    """True when the query compares multiple things (entities, seasons, games)."""
    return bool(_COMPARISON_RE.search(text))


def detect_trend_intent(text: str) -> bool:
    """True when the query asks about multiple seasons / career history."""
    return bool(_TREND_RE.search(text) or _PEAK_RE.search(text))


def detect_improvement_intent(text: str) -> bool:
    """True only for most-improved/breakout wording (delta scoped here)."""
    return bool(_IMPROVEMENT_RE.search(text))


def resolve_per_mode(text: str) -> str:
    """'Totals' when the user says totals/combined/altogether, else 'PerGame'."""
    return "Totals" if _TOTALS_RE.search(text) else "PerGame"


def resolve_season_window(
    text: str, today: Optional[datetime.date] = None
) -> Dict[str, Optional[str]]:
    """Resolve a trend window: full history vs last-N vs since-season.

    Returns {"last_n_seasons": int|None, "since_season": 'YYYY-YY'|None}.
    Both None means full career/history. Explicit 'YYYY-YY' elsewhere still
    flows through resolve_season for single-season queries.
    """
    today = today or datetime.date.today()
    m = _LAST_N_SEASONS_RE.search(text)
    if m:
        try:
            return {"last_n_seasons": max(1, min(int(m.group(1)), 25)),
                    "since_season": None}
        except ValueError:
            pass
    m = _SINCE_SEASON_RE.search(text)
    if m:
        raw = m.group(1)
        if re.fullmatch(r"(19|20)\d{2}", raw):
            yyyy = int(raw)
            since = f"{yyyy}-{str(yyyy + 1)[2:]}"
        else:
            y, yy = raw.split("-")
            since = f"{y}-{yy}"
        return {"last_n_seasons": None, "since_season": since}
    return {"last_n_seasons": None, "since_season": None}


def _season_start_year(season: str) -> int:
    try:
        return int(str(season).split("-")[0])
    except (ValueError, IndexError, AttributeError):
        return 0


def apply_season_window(
    seasons: List[str],
    last_n_seasons: Optional[int] = None,
    since_season: Optional[str] = None,
) -> List[str]:
    """Filter a chronological season list to the requested window."""
    ordered = sorted(seasons, key=_season_start_year)
    if since_season:
        cutoff = _season_start_year(since_season)
        ordered = [s for s in ordered if _season_start_year(s) >= cutoff]
    if last_n_seasons:
        ordered = ordered[-max(1, last_n_seasons):]
    return ordered


def largest_yoy_jump(
    rows: List[Dict], metric: str, season_key: str = "SEASON"
) -> Tuple[Optional[str], Optional[float], Optional[str]]:
    """Largest single-season positive jump of metric across chronological rows.

    Returns (season, delta, note). Season None when fewer than 2 numeric rows.
    Scoped to improvement-worded queries only — callers decide when to use it.
    """
    best_season: Optional[str] = None
    best_delta: Optional[float] = None
    prev: Optional[float] = None
    for r in sorted(rows, key=lambda d: _season_start_year(str(d.get(season_key, "")))):
        try:
            cur = float(r[metric]) if r.get(metric) is not None else None
        except (TypeError, ValueError):
            cur = None
        if cur is None:
            prev = None
            continue
        if prev is not None:
            delta = cur - prev
            if best_delta is None or delta > best_delta:
                best_delta = delta
                best_season = str(r.get(season_key))
        prev = cur
    if best_season is None or best_delta is None or best_delta <= 0:
        return None, None, None
    note = f"largest {metric} jump: +{best_delta:.1f} ({best_season})"
    return best_season, best_delta, note


# ---------------------------------------------------------------------------
# Viz-hint inference (deterministic — React just switches on this)
# ---------------------------------------------------------------------------
def choose_viz_hint(spec: QuerySpec) -> VizHint:
    """Pick the chart type from a resolved QuerySpec."""
    label = ", ".join(spec.players or spec.teams) or "NBA"
    season = spec.season or ""
    metrics = spec.metrics or ["PTS"]

    if spec.intent in ("player_career_trend", "team_history_trend"):
        flavor = "totals" if getattr(spec, "per_mode", "PerGame") == "Totals" else "per game"
        window = ""
        if getattr(spec, "seasons", None):
            if len(spec.seasons) == 1:
                window = spec.seasons[0]
            elif spec.seasons:
                window = f"{spec.seasons[0]} to {spec.seasons[-1]}"
        title = f"{label} {'/'.join(metrics)} {flavor} by season {window}".strip()
        return VizHint(type="trend_line", title=title, x_key="SEASON", y_keys=metrics)
    if spec.intent == "compare_trends":
        series_key = "TEAM_NAME" if spec.teams and not spec.players else "PLAYER_NAME"
        flavor = "totals" if getattr(spec, "per_mode", "PerGame") == "Totals" else "per game"
        window = ""
        if getattr(spec, "seasons", None):
            if len(spec.seasons) > 1:
                window = f"{spec.seasons[0]} to {spec.seasons[-1]}"
            elif spec.seasons:
                window = spec.seasons[0]
        who = ", ".join(spec.players or spec.teams) or label
        title = f"{who} {'/'.join(metrics)} {flavor} by season {window}".strip()
        return VizHint(
            type="multi_trend",
            title=title,
            x_key="SEASON",
            y_keys=metrics,
            series_key=series_key,
        )
    if spec.intent == "compare_teams":
        return VizHint(
            type="comparison_bars",
            title=f"{' vs '.join(spec.teams)} {season} ({'/'.join(metrics)})".strip(),
            x_key="TEAM_NAME",
            y_keys=metrics,
            series_key="TEAM_NAME",
        )
    if spec.intent == "player_season_avg" and len(spec.players) == 1:
        return VizHint(
            type="single_stat",
            title=f"{spec.players[0]} {metrics[0]} {season}".strip(),
            y_keys=metrics,
        )
    if spec.intent == "compare_players":
        return VizHint(
            type="comparison_bars",
            title=f"{' vs '.join(spec.players)} {season} ({'/'.join(metrics)})".strip(),
            x_key="PLAYER_NAME",
            y_keys=metrics,
            series_key="PLAYER_NAME",
        )
    if spec.intent in ("player_game_logs", "team_game_logs"):
        return VizHint(
            type="time_series",
            title=f"{label} last {spec.last_n or 10} games {season}".strip(),
            x_key="GAME_DATE",
            y_keys=metrics,
        )
    if spec.intent == "team_stats":
        return VizHint(
            type="team_stat_card",
            title=f"{label} {season}".strip(),
            y_keys=metrics,
        )
    return VizHint(type="game_log_table", title=label, y_keys=metrics)
