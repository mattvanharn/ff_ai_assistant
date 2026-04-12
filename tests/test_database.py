"""Tests for ff_ai_assistant.database."""

import pytest

from ff_ai_assistant.database import execute_query, format_results, get_schema


def test_execute_query_rejects_non_select():
    with pytest.raises(ValueError):
        execute_query("DROP TABLE player_seasons")


def test_execute_query_rejects_insert():
    with pytest.raises(ValueError):
        execute_query("INSERT INTO player_seasons VALUES (1)")


def test_format_results_empty_list():
    assert format_results([]) == "(No results)"


def test_format_results_single_row():
    rows = [{"player": "CMC", "points": 312.4}]
    result = format_results(rows)
    assert "CMC" in result
    assert "312.4" in result


def test_get_schema_contains_table_names():
    schema = get_schema()
    assert "player_seasons" in schema
    assert "weekly_stats" in schema
