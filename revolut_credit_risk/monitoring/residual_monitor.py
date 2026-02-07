"""Automated value-adding feature monitoring (residual monitoring).

[Paper §3] "The combination of DFS and MIV permits to establish an automated
risk-splitting, value-adding feature monitoring along the lines of the famed,
but highly manual, residual monitoring (ReMo) approach."
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from revolut_credit_risk import config
from revolut_credit_risk.features.binning import BinningResults
from revolut_credit_risk.selection.miv_selector import _compute_miv

logger = logging.getLogger(__name__)


@dataclass
class MonitorResult:
    """Result of residual monitoring."""

    candidate_table: pd.DataFrame | None = None
    top_features: list[str] = field(default_factory=list)


def run_residual_monitoring(
    model_probs: np.ndarray,
    y_true: np.ndarray,
    X_woe: pd.DataFrame,
    X_raw: pd.DataFrame,
    selected_features: list[str],
    binning_results: BinningResults,
    top_n: int = 10,
) -> MonitorResult:
    """Compute MIV for held-out features against the current model.

    [Paper §3.1] "the orchestrator calculates the MIV for the full set of
    generated features against a selected delinquency target."

    Parameters
    ----------
    model_probs : np.ndarray
        Current model's predicted probabilities on the dataset.
    y_true : np.ndarray
        Observed binary target.
    X_woe : pd.DataFrame
        WoE-transformed features for all features.
    X_raw : pd.DataFrame
        Raw feature values (for the residual monitoring bar chart).
    selected_features : list[str]
        Features currently in the model.
    binning_results : BinningResults
        Fitted binning objects.
    top_n : int
        Number of top candidates to report.

    Returns
    -------
    MonitorResult
    """
    result = MonitorResult()

    # Candidates = binned features NOT in the current model
    all_binned = set(binning_results.feature_names)
    selected_set = set(selected_features)
    candidates = sorted(all_binned - selected_set)

    if not candidates:
        logger.info("No candidate features for residual monitoring")
        return result

    # Compute MIV for each candidate against current model
    rows = []
    for feat in candidates:
        woe_col = f"woe_{feat}"
        if woe_col not in X_woe.columns:
            continue

        optb = binning_results.get_optb(feat)
        miv = _compute_miv(
            X_woe[woe_col].values,
            y_true,
            model_probs,
            optb,
        )
        rows.append({"feature": feat, "miv": miv})

    df = pd.DataFrame(rows).sort_values("miv", ascending=False).reset_index(drop=True)
    result.candidate_table = df
    result.top_features = df.head(top_n)["feature"].tolist()

    logger.info(
        "Residual monitoring: %d candidates evaluated, top MIV=%.4f (%s)",
        len(df),
        df["miv"].iloc[0] if len(df) > 0 else 0,
        result.top_features[0] if result.top_features else "N/A",
    )

    # Plot bar chart for top candidate
    if result.top_features:
        _plot_residual_bar(
            result.top_features[0], model_probs, y_true, X_raw, binning_results,
        )

    return result


def _plot_residual_bar(
    feature_name: str,
    model_probs: np.ndarray,
    y_true: np.ndarray,
    X_raw: pd.DataFrame,
    binning_results: BinningResults,
) -> None:
    """Bar chart: bad rate by risk grade, segmented by candidate feature bins.

    [Paper Fig. 9] "the observed early delinquency rate by coarse risk
    segment, with bars grouped by feature value ranges."
    """
    if feature_name not in X_raw.columns:
        return
    if feature_name not in binning_results.results:
        return

    # [Paper §3.1, Fig. 9] Risk grades via quartile buckets on model scores
    risk_grades = _assign_risk_grades(model_probs)

    # Use the fitted optbinning object to assign feature bins
    optb = binning_results.get_optb(feature_name)
    feat_vals = X_raw[feature_name].values.astype(float)
    feat_bins = optb.transform(feat_vals, metric="bins")

    plot_df = pd.DataFrame({
        "risk_grade": risk_grades,
        "feature_bin": feat_bins,
        "is_default": y_true,
    })

    try:
        pivot = plot_df.groupby(
            ["risk_grade", "feature_bin"], observed=True,
        )["is_default"].mean().unstack()
    except Exception:
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    pivot.plot(kind="bar", ax=ax, alpha=0.8)
    ax.set_xlabel("Risk Grade")
    ax.set_ylabel("Default Rate")
    ax.set_title(f"Residual Monitoring: {feature_name}")
    ax.legend(title=f"{feature_name} bins")
    ax.grid(True, alpha=0.3, axis="y")
    plt.xticks(rotation=0)

    path = config.PLOTS_DIR / "residual_monitor.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Residual monitoring plot saved to %s", path)


def _assign_risk_grades(probs: np.ndarray) -> pd.Categorical:
    """Assign risk grades robustly, handling duplicate quantile boundaries.

    Uses rank-based assignment when probabilities cluster too tightly for
    pd.qcut to produce 4 unique bins.
    """
    labels = ["Very Low", "Low", "Average", "High"]
    try:
        return pd.qcut(probs, q=4, labels=labels, duplicates="drop")
    except ValueError:
        pass

    # Fallback: rank-based quartile assignment (handles ties gracefully)
    ranks = pd.Series(probs).rank(method="first")
    quartiles = pd.cut(ranks, bins=4, labels=labels)
    return quartiles
