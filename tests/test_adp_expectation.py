"""Unit tests for ADP expectation helpers."""

import polars as pl

from ff_ai_assistant.adp_expectation import (
    AdpExpectTrainConfig,
    add_expected_finish_bucket_round_median,
)


def test_bucket_round_median_monotone_in_adp_for_rb() -> None:
    rows = []
    for adp in range(1, 25):
        rows.append(
            {
                "position": "RB",
                "adp": float(adp),
                "overall_points_rank": 10 + adp * 3,
                "seasonal_fantasy_points": 100.0,
            }
        )
    df = pl.DataFrame(rows)
    out = add_expected_finish_bucket_round_median(
        df, AdpExpectTrainConfig(league_size=12, min_samples_for_bucket=1)
    )
    e1 = out.filter(pl.col("adp") == 1.0)["expected_finish_bucket_round_median"].item()
    e12 = out.filter(pl.col("adp") == 12.0)["expected_finish_bucket_round_median"].item()
    e13 = out.filter(pl.col("adp") == 13.0)["expected_finish_bucket_round_median"].item()
    assert e1 == e12
    assert float(e13) > float(e1)
