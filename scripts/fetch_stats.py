# Fetch raw player stats from nflreadpy and save to CSV (Half-PPR scoring)

import nflreadpy as nfl
import polars as pl
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
SEASONS = list(range(2018, 2026))

# Fetch the data
weekly_stats = nfl.load_player_stats(SEASONS, summary_level="week")

# Filter to fantasy football positions and weeks 1-17
weekly_stats = weekly_stats.filter(
    pl.col("position").is_in(["QB", "RB", "WR", "TE", "K", "DST"])
).filter(pl.col("week") <= 17)

# Half-PPR: nflreadpy has no half-PPR option, so compute as (standard + PPR) / 2
weekly_stats = weekly_stats.with_columns(
    ((pl.col("fantasy_points") + pl.col("fantasy_points_ppr")) / 2).alias(
        "fantasy_points_half_ppr"
    )
)

# One row per player-season (no team in group_by — trades would split rows and break totals/ranks)
weekly_stats = weekly_stats.sort(["season", "player_id", "week"])
season_stats = weekly_stats.group_by(
    ["player_id", "player_display_name", "position", "season"]
).agg(
    pl.col("fantasy_points_half_ppr").sum().alias("seasonal_fantasy_points"),
    pl.col("fantasy_points_half_ppr").mean().alias("seasonal_fantasy_points_mean"),
    pl.col("fantasy_points_half_ppr").median().alias("seasonal_fantasy_points_median"),
    pl.col("fantasy_points_half_ppr").max().alias("seasonal_fantasy_points_max"),
    pl.col("fantasy_points_half_ppr").min().alias("seasonal_fantasy_points_min"),
    pl.col("team").last().alias("team"),
)

# Finish ranks: 1 = most points that season. Ties share the same rank (method="min").
# overall = all positions; positional = within QB/RB/WR/TE/K/DST for that season.
season_stats = season_stats.with_columns(
    pl.col("seasonal_fantasy_points")
    .rank(method="min", descending=True)
    .over("season")
    .cast(pl.Int32)
    .alias("overall_points_rank"),
    pl.col("seasonal_fantasy_points")
    .rank(method="min", descending=True)
    .over(["season", "position"])
    .cast(pl.Int32)
    .alias("position_points_rank"),
)

# Save the season stats and weekly stats
season_stats.write_csv(DATA_DIR / "player_stats_2018_2025_season.csv", separator="\t")
weekly_stats.write_csv(DATA_DIR / "player_stats_2018_2025_weekly.csv", separator="\t")
