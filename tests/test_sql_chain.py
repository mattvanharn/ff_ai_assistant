"""Tests for ff_ai_assistant.sql_chain."""

from ff_ai_assistant.sql_chain import extract_select_sql


def test_extract_strips_sql_fence():
    raw = "```sql\nSELECT * FROM player_seasons\n```"
    assert extract_select_sql(raw) == "SELECT * FROM player_seasons"


def test_extract_strips_plain_fence():
    raw = "```\nSELECT * FROM player_seasons\n```"
    assert extract_select_sql(raw) == "SELECT * FROM player_seasons"


def test_extract_passes_through_clean_sql():
    sql = "SELECT * FROM player_seasons"
    assert extract_select_sql(sql) == sql
