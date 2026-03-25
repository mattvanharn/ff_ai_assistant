"""Simpler ADP → expected overall finish (rank) helpers.

Use in the notebook to compare methods; write the chosen column(s) to
``combined_stats_adp.parquet`` and set ``INGEST_EXPECTED_FINISH_COLUMN`` in config.

Training rows default to all player-seasons with non-null ``adp`` and
``overall_points_rank``. Set ``train_min_seasonal_points`` to down-weight
full-season injuries/holdouts in the *baseline* (not a perfect fix — PPG later).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl
from sklearn.linear_model import Ridge
from sklearn.preprocessing import KBinsDiscretizer


@dataclass(frozen=True)
class AdpExpectTrainConfig:
    league_size: int = 12
    max_round: int = 32
    train_min_seasonal_points: float | None = None
    min_samples_for_bucket: int = 3
    linear_ridge_alpha: float = 25.0
    log_linear_ridge_alpha: float = 400.0
    linear_min_samples: int = 40
    decile_min_samples: int = 40
    n_adp_bins: int = 10


def _training_frame(df: pl.DataFrame, cfg: AdpExpectTrainConfig) -> pl.DataFrame:
    t = df.filter(
        pl.col("adp").is_not_null(),
        pl.col("overall_points_rank").is_not_null(),
    )
    if cfg.train_min_seasonal_points is not None:
        t = t.filter(pl.col("seasonal_fantasy_points") >= cfg.train_min_seasonal_points)
    return t


def _draft_round_expr(league_size: int, max_round: int) -> pl.Expr:
    r = ((pl.col("adp") - 1) // league_size + 1).cast(pl.Int32)
    return (
        pl.when(r < 1)
        .then(pl.lit(1))
        .when(r > max_round)
        .then(pl.lit(max_round))
        .otherwise(r)
        .alias("_draft_round_tmp")
    )


def _drop_if_present(df: pl.DataFrame, names: list[str]) -> pl.DataFrame:
    keep = [c for c in df.columns if c not in names]
    return df.select(keep)


def add_expected_finish_bucket_round_median(
    df: pl.DataFrame, cfg: AdpExpectTrainConfig | None = None
) -> pl.DataFrame:
    """Draft round = ceil(ADP / league_size); expected rank = median overall rank in (pos, round)."""
    cfg = cfg or AdpExpectTrainConfig()
    train = _training_frame(df, cfg).with_columns(
        _draft_round_expr(cfg.league_size, cfg.max_round)
    )
    pos_fb = train.group_by("position").agg(
        pl.col("overall_points_rank").median().alias("_pos_med")
    )
    bucket = train.group_by("position", "_draft_round_tmp").agg(
        pl.col("overall_points_rank").median().alias("_b_med"),
        pl.len().alias("_b_n"),
    )
    out = df.with_columns(_draft_round_expr(cfg.league_size, cfg.max_round))
    out = out.join(
        bucket,
        left_on=["position", "_draft_round_tmp"],
        right_on=["position", "_draft_round_tmp"],
        how="left",
    ).join(pos_fb, on="position", how="left")
    out = out.with_columns(
        pl.when(pl.col("adp").is_null())
        .then(None)
        .when(pl.col("_b_n") >= cfg.min_samples_for_bucket)
        .then(pl.col("_b_med"))
        .otherwise(pl.col("_pos_med"))
        .cast(pl.Float64)
        .alias("expected_finish_bucket_round_median")
    )
    return _drop_if_present(out, ["_draft_round_tmp", "_b_med", "_b_n", "_pos_med"])


def add_expected_finish_bucket_round_mean(
    df: pl.DataFrame, cfg: AdpExpectTrainConfig | None = None
) -> pl.DataFrame:
    cfg = cfg or AdpExpectTrainConfig()
    train = _training_frame(df, cfg).with_columns(
        _draft_round_expr(cfg.league_size, cfg.max_round)
    )
    pos_fb = train.group_by("position").agg(
        pl.col("overall_points_rank").mean().alias("_pos_mean")
    )
    bucket = train.group_by("position", "_draft_round_tmp").agg(
        pl.col("overall_points_rank").mean().alias("_b_mean"),
        pl.len().alias("_b_n"),
    )
    out = df.with_columns(_draft_round_expr(cfg.league_size, cfg.max_round))
    out = out.join(
        bucket,
        left_on=["position", "_draft_round_tmp"],
        right_on=["position", "_draft_round_tmp"],
        how="left",
    ).join(pos_fb, on="position", how="left")
    out = out.with_columns(
        pl.when(pl.col("adp").is_null())
        .then(None)
        .when(pl.col("_b_n") >= cfg.min_samples_for_bucket)
        .then(pl.col("_b_mean"))
        .otherwise(pl.col("_pos_mean"))
        .cast(pl.Float64)
        .alias("expected_finish_bucket_round_mean")
    )
    return _drop_if_present(out, ["_draft_round_tmp", "_b_mean", "_b_n", "_pos_mean"])


def add_expected_finish_linear_ridge(
    df: pl.DataFrame, cfg: AdpExpectTrainConfig | None = None
) -> pl.DataFrame:
    """Per position: Ridge regression ADP → overall_points_rank (no splines)."""
    cfg = cfg or AdpExpectTrainConfig()
    train = _training_frame(df, cfg)
    max_rank = float(train["overall_points_rank"].max() or 1.0)
    models: dict[str, Ridge] = {}
    for pos in train["position"].unique().sort().to_list():
        sub = train.filter(pl.col("position") == pos)
        if sub.height < cfg.linear_min_samples:
            continue
        x = sub["adp"].to_numpy().reshape(-1, 1)
        y = sub["overall_points_rank"].to_numpy()
        m = Ridge(alpha=cfg.linear_ridge_alpha)
        m.fit(x, y)
        models[str(pos)] = m

    adp = df["adp"].to_numpy()
    pos = df["position"].to_list()
    pred = np.full(df.height, np.nan, dtype=np.float64)
    for i in range(df.height):
        if adp[i] is None or (isinstance(adp[i], float) and np.isnan(adp[i])):
            continue
        p = pos[i]
        if p not in models:
            continue
        v = float(models[p].predict(np.array([[float(adp[i])]], dtype=np.float64))[0])
        pred[i] = min(max(v, 1.0), max_rank)

    return df.with_columns(pl.Series("expected_finish_linear_ridge", pred, dtype=pl.Float64))


def add_expected_finish_adp_quantile_bin_median(
    df: pl.DataFrame, cfg: AdpExpectTrainConfig | None = None
) -> pl.DataFrame:
    """Per position: quantile bins on ADP (KBinsDiscretizer), median rank per bin."""
    cfg = cfg or AdpExpectTrainConfig()
    train = _training_frame(df, cfg)
    max_rank = float(train["overall_points_rank"].max() or 1.0)

    pred = np.full(df.height, np.nan, dtype=np.float64)
    adp_all = df["adp"].to_numpy()
    pos_all = df["position"].to_list()

    for pos in train["position"].unique().sort().to_list():
        sub = train.filter(pl.col("position") == pos)
        if sub.height < cfg.decile_min_samples:
            continue
        x = sub["adp"].to_numpy().reshape(-1, 1)
        y = sub["overall_points_rank"].to_numpy()
        n_bins = min(cfg.n_adp_bins, sub.height // max(cfg.min_samples_for_bucket, 1))
        if n_bins < 3:
            continue
        try:
            disc = KBinsDiscretizer(
                n_bins=n_bins,
                encode="ordinal",
                strategy="quantile",
                subsample=None,
                quantile_method="linear",
            )
            disc.fit(x)
        except ValueError:
            continue
        bin_train = disc.transform(x).astype(int).ravel()
        medians: dict[int, float] = {}
        counts: dict[int, int] = {}
        for b in range(int(bin_train.max()) + 1):
            mask = bin_train == b
            if not np.any(mask):
                continue
            medians[b] = float(np.median(y[mask]))
            counts[b] = int(np.sum(mask))
        fallback = float(np.median(y))

        idxs = [i for i in range(df.height) if pos_all[i] == pos]
        for i in idxs:
            a = adp_all[i]
            if a is None or (isinstance(a, float) and np.isnan(a)):
                continue
            b = int(disc.transform(np.array([[float(a)]], dtype=np.float64))[0, 0])
            if b in medians and counts.get(b, 0) >= cfg.min_samples_for_bucket:
                v = medians[b]
            else:
                v = fallback
            pred[i] = min(max(float(v), 1.0), max_rank)

    return df.with_columns(
        pl.Series("expected_finish_adp_quantile_median", pred, dtype=pl.Float64)
    )


def _slot_bin_id_expr(slot_width: float) -> pl.Expr:
    return ((pl.col("adp") - 1.0) // float(slot_width)).cast(pl.Int32).alias("_slot_bin_tmp")


def add_expected_finish_fixed_slot_median(
    df: pl.DataFrame,
    slot_width: float,
    *,
    column_name: str,
    cfg: AdpExpectTrainConfig | None = None,
) -> pl.DataFrame:
    """Fixed-width ADP bins (e.g. 6 ≈ half a 12-team round); median overall rank per (pos, bin)."""
    cfg = cfg or AdpExpectTrainConfig()
    w = float(slot_width)
    train = _training_frame(df, cfg).with_columns(_slot_bin_id_expr(w))
    pos_fb = train.group_by("position").agg(
        pl.col("overall_points_rank").median().alias("_pos_med")
    )
    bucket = train.group_by("position", "_slot_bin_tmp").agg(
        pl.col("overall_points_rank").median().alias("_b_med"),
        pl.len().alias("_b_n"),
    )
    out = df.with_columns(_slot_bin_id_expr(w))
    out = out.join(
        bucket,
        left_on=["position", "_slot_bin_tmp"],
        right_on=["position", "_slot_bin_tmp"],
        how="left",
    ).join(pos_fb, on="position", how="left")
    out = out.with_columns(
        pl.when(pl.col("adp").is_null())
        .then(None)
        .when(pl.col("_b_n") >= cfg.min_samples_for_bucket)
        .then(pl.col("_b_med"))
        .otherwise(pl.col("_pos_med"))
        .cast(pl.Float64)
        .alias(column_name)
    )
    return _drop_if_present(out, ["_slot_bin_tmp", "_b_med", "_b_n", "_pos_med"])


def add_expected_finish_log_adp_ridge(
    df: pl.DataFrame, cfg: AdpExpectTrainConfig | None = None
) -> pl.DataFrame:
    """Per position: Ridge on log1p(ADP) → overall_points_rank (gentler at low ADP than raw linear)."""
    cfg = cfg or AdpExpectTrainConfig()
    train = _training_frame(df, cfg)
    max_rank = float(train["overall_points_rank"].max() or 1.0)
    models: dict[str, Ridge] = {}
    y_min_by_pos: dict[str, float] = {}
    for pos in train["position"].unique().sort().to_list():
        sub = train.filter(pl.col("position") == pos)
        if sub.height < cfg.linear_min_samples:
            continue
        adp = sub["adp"].to_numpy().reshape(-1, 1)
        x = np.log1p(adp)
        y = sub["overall_points_rank"].to_numpy()
        y_min_by_pos[str(pos)] = float(np.min(y))
        m = Ridge(alpha=cfg.log_linear_ridge_alpha)
        m.fit(x, y)
        models[str(pos)] = m

    adp_all = df["adp"].to_numpy()
    pos_all = df["position"].to_list()
    pred = np.full(df.height, np.nan, dtype=np.float64)
    for i in range(df.height):
        a = adp_all[i]
        if a is None or (isinstance(a, float) and np.isnan(a)):
            continue
        p = pos_all[i]
        if p not in models:
            continue
        lx = np.log1p(float(a))
        v = float(models[p].predict(np.array([[lx]], dtype=np.float64))[0])
        lo = y_min_by_pos[str(p)]
        if np.isnan(v):
            continue
        pred[i] = min(max(v, lo), max_rank)

    return df.with_columns(
        pl.Series("expected_finish_log_adp_ridge", pred, dtype=pl.Float64)
    )


def add_all_expected_finish_methods(
    df: pl.DataFrame, cfg: AdpExpectTrainConfig | None = None
) -> pl.DataFrame:
    """Apply round buckets, fixed slot buckets, linear / log ridge, and quantile-bin median."""
    out = add_expected_finish_bucket_round_median(df, cfg)
    out = add_expected_finish_bucket_round_mean(out, cfg)
    out = add_expected_finish_fixed_slot_median(
        out, 6.0, column_name="expected_finish_slot6_median", cfg=cfg
    )
    out = add_expected_finish_linear_ridge(out, cfg)
    out = add_expected_finish_log_adp_ridge(out, cfg)
    out = add_expected_finish_adp_quantile_bin_median(out, cfg)
    return out


# Column names produced by ``add_all_expected_finish_methods`` (for notebooks / QA).
ALL_EXPECTED_FINISH_METHOD_COLUMNS: tuple[str, ...] = (
    "expected_finish_bucket_round_median",
    "expected_finish_bucket_round_mean",
    "expected_finish_slot6_median",
    "expected_finish_linear_ridge",
    "expected_finish_log_adp_ridge",
    "expected_finish_adp_quantile_median",
)
