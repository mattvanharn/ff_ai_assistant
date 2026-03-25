"""Convert historical data into text documents for a deferred RAG pipeline.

Each player-season becomes one Document whose ``page_content`` is a plain-English
summary suitable for embedding.  The text includes:

- name, position, team, season, total half-PPR points
- weekly average and best week
- positional finish rank and overall finish rank
- ADP (if drafted) with a simple within-position value note
- games played + PPG when available
- experience level (rookie / developing)
"""

from __future__ import annotations

import polars as pl
from langchain_core.documents import Document

from ff_ai_assistant.config import (
    COMBINED_PARQUET,
    INGEST_INCLUDE_IF_ADP_LE,
    INGEST_MIN_SEASONAL_POINTS,
    POSITIONS,
)


def _positional_value_note(row: dict) -> str:
    """Compare positional ADP rank to positional finish rank (within-position only).

    Returns a short sentence like "Drafted as RB5 by ADP, finished RB2 — good value
    relative to positional draft cost."  Returns "" when the needed columns are missing.
    """
    pos = row["position"]
    adp_pos_rank = row.get("adp_position_rank")
    finish_pos_rank = row.get("position_points_rank")
    if adp_pos_rank is None or finish_pos_rank is None:
        return ""

    adp_pos_rank = int(adp_pos_rank)
    finish_pos_rank = int(finish_pos_rank)
    delta = adp_pos_rank - finish_pos_rank  # positive = beat ADP

    label = f" Drafted as {pos}{adp_pos_rank} by ADP, finished {pos}{finish_pos_rank}."
    if delta >= 8:
        return label + " Exceptional value relative to positional draft cost."
    if delta >= 3:
        return label + " Good value relative to positional draft cost."
    if delta >= -2:
        return label + " Roughly matched positional draft cost."
    if delta >= -7:
        return label + " Below expectations for positional draft cost."
    return label + " Major disappointment relative to positional draft cost."


def player_season_to_text(row: dict) -> str:
    """Convert one player-season row (dict) into a text summary for embedding."""
    name = row["player_display_name"]
    pos = row["position"]
    team = row.get("team") or "FA"
    season = row["season"]
    pts = row["seasonal_fantasy_points"]

    text = f"{name} ({pos}, {team}) {season}: {pts:.1f} half-PPR points."

    games = row.get("games_played")
    if games is not None and int(games) > 0:
        ppg = pts / int(games)
        text += f" {int(games)} games, {ppg:.1f} PPG."

    mean_pts = row.get("seasonal_fantasy_points_mean")
    if mean_pts is not None:
        text += f" Weekly average {mean_pts:.1f}."

    max_pts = row.get("seasonal_fantasy_points_max")
    if max_pts is not None:
        text += f" Best week {max_pts:.1f}."

    pos_rank = row.get("position_points_rank")
    overall_rank = row.get("overall_points_rank")
    if pos_rank is not None:
        text += f" Finished {pos}{int(pos_rank)} among {pos}s."
    if overall_rank is not None:
        text += f" Overall finish rank {int(overall_rank)}."

    adp = row.get("adp")
    if adp is not None:
        text += f" Drafted at ADP {adp:.1f}."
        text += _positional_value_note(row)

    years = row.get("years_exp")
    if years is not None:
        if years == 0:
            text += " Rookie season."
        elif years <= 2:
            text += f" {years + 1}-year player."

    return text


def ingest_row_filter_expr(
    min_points: float = INGEST_MIN_SEASONAL_POINTS,
    include_if_adp_le: float = INGEST_INCLUDE_IF_ADP_LE,
) -> pl.Expr:
    """Polars predicate: keep rows with enough points or early (low) ADP."""
    return (pl.col("seasonal_fantasy_points") >= min_points) | (
        pl.col("adp").is_not_null() & (pl.col("adp") <= include_if_adp_le)
    )


def build_documents(
    min_points: float = INGEST_MIN_SEASONAL_POINTS,
    include_if_adp_le: float = INGEST_INCLUDE_IF_ADP_LE,
) -> list[Document]:
    """Load combined parquet and produce one Document per player-season.

    Rows are kept when ``seasonal_fantasy_points >= min_points`` **or** when ADP is
    non-null and ``<= include_if_adp_le`` (so early-pick busts/holdouts stay embedded).
    """
    df = pl.read_parquet(COMBINED_PARQUET)

    # Build adp_position_rank if not already present (rank within position per season).
    if "adp_position_rank" not in df.columns and "adp" in df.columns:
        df = df.with_columns(
            pl.col("adp")
            .rank(method="ordinal", descending=False)
            .over(["season", "position"])
            .cast(pl.Int32)
            .alias("adp_position_rank"),
        )

    df = df.filter(
        ingest_row_filter_expr(min_points, include_if_adp_le),
        pl.col("position").is_in(POSITIONS),
    )

    docs = []
    for row in df.iter_rows(named=True):
        text = player_season_to_text(row)
        metadata = {
            "player_id": row.get("player_id") or "",
            "player_name": row["player_display_name"],
            "position": row["position"],
            "season": row["season"],
            "fantasy_points": row["seasonal_fantasy_points"],
        }
        team = row.get("team") or row.get("roster_team")
        if team:
            metadata["team"] = team
        adp = row.get("adp")
        if adp is not None:
            metadata["adp"] = adp
        prank = row.get("position_points_rank")
        if prank is not None:
            metadata["position_points_rank"] = prank
        orank = row.get("overall_points_rank")
        if orank is not None:
            metadata["overall_points_rank"] = orank

        docs.append(Document(page_content=text, metadata=metadata))

    return docs


