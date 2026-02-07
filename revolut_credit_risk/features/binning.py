"""Optbinning wrapper for coarse binning and WoE transformation.

[Paper §2.2.5] Coarse binning via decision-tree-based splits.
[Paper §2.2.6] WoE transformation: WoE(X=a) = log(P(X=a|Bad) / P(X=a|Good)).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from optbinning import OptimalBinning

from revolut_credit_risk import config

logger = logging.getLogger(__name__)


@dataclass
class BinningResult:
    """Result of binning a single feature."""

    feature_name: str
    iv: float
    n_bins: int
    monotonic_trend: str
    status: str
    optb: OptimalBinning
    binning_table: pd.DataFrame


@dataclass
class BinningResults:
    """Aggregated binning results for all features."""

    results: dict[str, BinningResult] = field(default_factory=dict)

    @property
    def feature_names(self) -> list[str]:
        return list(self.results.keys())

    def get_iv(self, feature: str) -> float:
        return self.results[feature].iv

    def get_optb(self, feature: str) -> OptimalBinning:
        return self.results[feature].optb


def bin_features(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    variable_configs: dict[str, dict] | None = None,
) -> BinningResults:
    """Bin all features using optbinning and compute WoE/IV.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training features (raw, pre-WoE).
    y_train : pd.Series
        Binary target variable.
    variable_configs : dict, optional
        Per-variable binning config from variable_config module.

    Returns
    -------
    BinningResults
        Contains fitted OptimalBinning objects, IV values, binning tables.
    """
    variable_configs = variable_configs or {}
    results = BinningResults()
    n_ok = 0
    n_fail = 0

    for col in X_train.columns:
        vcfg = variable_configs.get(col, {})
        try:
            res = _bin_single_feature(col, X_train[col], y_train, vcfg)
            if res is not None:
                results.results[col] = res
                n_ok += 1
                logger.debug(
                    "Feature '%s': IV=%.4f, %d bins, monotonic=%s, status=%s",
                    col, res.iv, res.n_bins, res.monotonic_trend, res.status,
                )
            else:
                n_fail += 1
        except Exception as e:
            logger.warning("Binning failed for '%s': %s", col, e)
            n_fail += 1

    logger.info(
        "Binning complete: %d/%d features binned successfully (%d failed)",
        n_ok, n_ok + n_fail, n_fail,
    )
    return results


def _bin_single_feature(
    name: str,
    x: pd.Series,
    y: pd.Series,
    vcfg: dict,
) -> BinningResult | None:
    """Fit OptimalBinning on a single feature."""
    # Skip constant or near-constant columns
    if x.nunique() < 2:
        logger.debug("Skipping '%s': fewer than 2 unique values", name)
        return None

    dtype = vcfg.get("dtype", "numerical")
    monotonic = vcfg.get("monotonic_trend", config.BINNING_MONOTONIC_TREND)
    max_bins = vcfg.get("max_n_bins", config.BINNING_MAX_N_BINS)

    optb = OptimalBinning(
        name=name,
        dtype=dtype,
        monotonic_trend=monotonic,      # [Paper §2.2.5] monotonicity enforcement
        max_n_bins=max_bins,
        min_bin_size=config.BINNING_MIN_BIN_SIZE,  # [Paper §2.2.5] "e.g., 5%"
        min_prebin_size=config.BINNING_MIN_PREBIN_SIZE,
        solver=config.BINNING_SOLVER,
        divergence=config.BINNING_DIVERGENCE,       # [Paper §2.3.2]
    )

    x_clean = x.values.astype(float)
    y_clean = y.values.astype(int)

    # Remove NaN/inf for fitting
    mask = np.isfinite(x_clean)
    if mask.sum() < 50:
        logger.debug("Skipping '%s': fewer than 50 finite values", name)
        return None

    optb.fit(x_clean[mask], y_clean[mask])

    # [Common gotcha] check status after fitting
    if optb.status not in ("OPTIMAL", "FEASIBLE"):
        logger.debug("Skipping '%s': optbinning status=%s", name, optb.status)
        return None

    binning_table = optb.binning_table.build()
    # Total IV from binning table (exclude Totals/Special/Missing rows)
    iv_col = binning_table["IV"].values
    total_iv = float(iv_col[:-1].sum())  # last row is usually Totals

    n_bins = len(binning_table) - 2  # exclude Special and Totals rows

    return BinningResult(
        feature_name=name,
        iv=total_iv,
        n_bins=max(n_bins, 1),
        monotonic_trend=monotonic,
        status=optb.status,
        optb=optb,
        binning_table=binning_table,
    )


def transform_woe(
    X: pd.DataFrame,
    binning_results: BinningResults,
    features: list[str] | None = None,
) -> pd.DataFrame:
    """Apply WoE transformation using fitted binning objects.

    [Paper §2.2.6] WoE replaces each bin with log-odds ratio.

    Parameters
    ----------
    X : pd.DataFrame
        Raw features to transform.
    binning_results : BinningResults
        Fitted binning results.
    features : list[str], optional
        Subset of features to transform. If None, transform all binned features.

    Returns
    -------
    pd.DataFrame
        WoE-transformed features, prefixed with ``woe_``.
    """
    features = features or binning_results.feature_names
    woe_frames: dict[str, np.ndarray] = {}

    for feat in features:
        if feat not in binning_results.results:
            continue
        optb = binning_results.get_optb(feat)
        x_vals = X[feat].values.astype(float)
        woe_vals = optb.transform(x_vals, metric="woe")
        # [Common gotcha] WoE can be ±inf for empty bins — clip
        woe_vals = np.clip(woe_vals, -5.0, 5.0)
        woe_frames[f"woe_{feat}"] = woe_vals

    return pd.DataFrame(woe_frames, index=X.index)


def plot_binning_tables(
    binning_results: BinningResults,
    selected_features: list[str],
) -> None:
    """Save optbinning event-rate plots for each selected feature.

    [Paper §2.2.5, §2.2.6] Visualise bin boundaries, event (default) rate,
    and WoE per bin for every variable in the final model.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plots_dir = config.PLOTS_DIR / "binning"
    plots_dir.mkdir(parents=True, exist_ok=True)

    for feat in selected_features:
        if feat not in binning_results.results:
            continue
        optb = binning_results.get_optb(feat)

        # optbinning's built-in plot: event rate + WoE per bin
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

        table = optb.binning_table.build()
        # Exclude Special/Missing/Totals rows (last 2-3 rows in optbinning)
        exclude = {"Special", "Missing", "Totals"}
        data_rows = table[~table["Bin"].astype(str).isin(exclude)].iloc[:-1]
        bins = data_rows["Bin"].astype(str).values
        event_rate = pd.to_numeric(data_rows["Event rate"], errors="coerce").fillna(0).values
        woe = pd.to_numeric(data_rows["WoE"], errors="coerce").fillna(0).values
        count = pd.to_numeric(data_rows["Count"], errors="coerce").fillna(0).values

        x_pos = np.arange(len(bins))

        # Top: event rate (default rate) with count overlay
        bars = ax1.bar(x_pos, event_rate, color="#e74c3c", alpha=0.8, label="Default rate")
        ax1.set_ylabel("Default Rate")
        ax1.set_title(f"{feat}")
        ax1.grid(True, alpha=0.3, axis="y")

        # Annotate counts on bars
        for i, (bar, cnt) in enumerate(zip(bars, count)):
            ax1.text(
                bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"n={int(cnt)}", ha="center", va="bottom", fontsize=8,
            )

        # Bottom: WoE per bin
        colors = ["#2ecc71" if w >= 0 else "#3498db" for w in woe]
        ax2.bar(x_pos, woe, color=colors, alpha=0.8)
        ax2.set_ylabel("WoE")
        ax2.set_xlabel("Bin")
        ax2.axhline(y=0, color="black", linewidth=0.5)
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(bins, rotation=45, ha="right", fontsize=8)
        ax2.grid(True, alpha=0.3, axis="y")

        plt.tight_layout()

        # Sanitise feature name for filename
        safe_name = feat.replace(".", "_").replace("(", "").replace(")", "")
        safe_name = safe_name.replace(" ", "_").replace("=", "eq")
        path = plots_dir / f"{safe_name}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    logger.info(
        "Binning plots saved for %d selected features in %s",
        len(selected_features), plots_dir,
    )
