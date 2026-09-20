"""Messy-NLP tolerance: stat spellings, LLM-owned name fallback — no network.

General cases across stat families, not the example queries. Names stay dumb
on purpose: unknown mentions fail clean so the agent retries (rule 6).

Run: .venv/bin/python -m pytest tests/data_tooling/test_messy_nlp.py -q
"""
import pytest

from src.data_tooling import resolver as R

FAKE = [{"full_name": n} for n in [
    "Stephen Curry", "Seth Curry", "Luka Doncic", "Jalen Brunson",
    "Rick Brunson", "Kevin Durant", "Damian Lillard", "Nikola Jokic",
]]


# --- stat normalization ----------------------------------------------------
def test_digit_hyphen_percentage():
    assert R.resolve_metrics("Curry's 3-point percentage vs Luka") == ["FG3_PCT"]


def test_word_hyphen_percentage():
    assert R.resolve_metrics("three-point percentage last 5 seasons") == ["FG3_PCT"]


def test_digit_space_percentage():
    assert R.resolve_metrics("best 3 point percentage in 2024-25") == ["FG3_PCT"]


def test_hyphen_free_throw():
    assert R.resolve_metrics("free-throw percentage leaders") == ["FT_PCT"]


def test_hyphen_field_goal():
    assert R.resolve_metrics("field-goal percentage trend") == ["FG_PCT"]


def test_digit_threes_still_fg3m():
    assert R.resolve_metrics("how many 3 pointers did he make") == ["FG3M"]


def test_bare_3_points_stays_quantity_pts():
    # No -er/-point marker: reads as a quantity, PTS default is the safe miss.
    assert R.resolve_metrics("he scored 3 points in the quarter") == ["PTS"]


def test_trey_slang():
    assert R.resolve_metrics("most treys in a season") == ["FG3M"]


def test_existing_forms_unchanged():
    assert R.resolve_metrics("How many three pointers did he make") == ["FG3M"]
    assert R.resolve_metrics("how many points did brunson average") == ["PTS"]


# --- names: deterministic layer stays dumb, LLM retries -----------------------
# No nickname map by design (it would rot with trades/rookies): unknown
# mentions fail clean with a retry invitation the agent acts on (rule 6).
def test_nickname_fails_clean_at_resolver():
    assert R.resolve_player("Steph Curry", FAKE) == (None, [])


def test_unknown_error_invites_formal_name_retry():
    from src.data_tooling.nba_tools import ToolError, get_player_id
    with pytest.raises(ToolError) as exc:
        get_player_id("Steph Curry")
    assert "full formal name" in str(exc.value)


def test_example_query_stat_and_shape_parse():
    q = "Show me Steph Curry's 3-point percentage vs. Luka Doncic's over the last 5 seasons"
    assert R.resolve_metrics(q) == ["FG3_PCT"]
    assert R.detect_comparison_intent(q)
    assert R.resolve_season_window(q) == {"last_n_seasons": 5, "since_season": None}
    # Name left for the agent: resolver fails clean, error invites retry.
    assert R.resolve_player("Steph Curry", FAKE) == (None, [])


def test_non_nicknames_unchanged():
    assert R.resolve_player("Mickey Mouse", FAKE) == (None, [])
    name, cands = R.resolve_player("Brunson", FAKE)
    assert name is None and sorted(cands) == ["Jalen Brunson", "Rick Brunson"]
    assert R.resolve_player("Luka Doncic", FAKE) == ("Luka Doncic", [])
