"""Smoke tests for RAG text generation (no parquet I/O)."""

import polars as pl

from ff_ai_assistant.ingest import ingest_row_filter_expr, player_season_to_text


def test_ingest_row_filter_expr_points_or_early_adp() -> None:
    df = pl.DataFrame(
        {
            "seasonal_fantasy_points": [0.0, 50.0, 5.0, 5.0],
            "adp": [2.0, None, 100.0, 250.0],
        }
    )
    out = df.filter(ingest_row_filter_expr(10.0, 192.0))
    assert len(out) == 3
    assert out["adp"].to_list()[-1] == 100.0


def test_player_season_to_text_minimal_row() -> None:
    row = {
        "player_display_name": "Test Player",
        "position": "WR",
        "team": "TST",
        "season": 2024,
        "seasonal_fantasy_points": 150.0,
    }
    out = player_season_to_text(row)
    assert "Test Player" in out and "WR" in out and "2024" in out
    assert "150.0 half-PPR" in out


def test_player_season_to_text_with_positional_value() -> None:
    row = {
        "player_display_name": "Breakout WR",
        "position": "WR",
        "team": "SEA",
        "season": 2023,
        "seasonal_fantasy_points": 200.0,
        "overall_points_rank": 5,
        "position_points_rank": 2,
        "adp": 48.0,
        "adp_position_rank": 12,
    }
    out = player_season_to_text(row)
    assert "Finished WR2 among WRs" in out
    assert "Overall finish rank 5" in out
    assert "Drafted at ADP 48.0" in out
    assert "Drafted as WR12 by ADP, finished WR2" in out
    assert "Exceptional value" in out


def test_player_season_to_text_bust() -> None:
    row = {
        "player_display_name": "Bust RB",
        "position": "RB",
        "team": "NYJ",
        "season": 2020,
        "seasonal_fantasy_points": 40.0,
        "position_points_rank": 55,
        "adp": 3.0,
        "adp_position_rank": 2,
        "games_played": 3,
    }
    out = player_season_to_text(row)
    assert "3 games" in out
    assert "13.3 PPG" in out
    assert "Drafted as RB2 by ADP, finished RB55" in out
    assert "Major disappointment" in out


def test_player_season_to_text_no_adp() -> None:
    row = {
        "player_display_name": "Undrafted WR",
        "position": "WR",
        "team": "GB",
        "season": 2022,
        "seasonal_fantasy_points": 120.0,
        "position_points_rank": 20,
    }
    out = player_season_to_text(row)
    assert "Finished WR20" in out
    assert "Drafted at ADP" not in out
