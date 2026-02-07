"""Evaluation metrics and plots: Gini, AUC, Brier, KS, Lorenz, calibration.

[Paper §2.4.1] Performance evaluation: Gini, Lorenz curve, calibration,
Brier score.
"""
from __future__ import annotations

import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve, brier_score_loss

from revolut_credit_risk import config

logger = logging.getLogger(__name__)


def compute_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    label: str = "",
) -> dict[str, float]:
    """Compute all metrics for a single dataset split.

    Returns dict with keys: auc, gini, ks, brier.
    """
    auc = roc_auc_score(y_true, y_prob)
    gini = 2 * auc - 1  # [Paper §2.4.1]
    ks = _ks_statistic(y_true, y_prob)
    brier = brier_score_loss(y_true, y_prob)  # [Paper §2.4.1, Ref. 13]

    metrics = {"auc": auc, "gini": gini, "ks": ks, "brier": brier}

    if label:
        logger.info(
            "%s — AUC=%.4f, Gini=%.4f, KS=%.4f, Brier=%.4f",
            label, auc, gini, ks, brier,
        )
    return metrics


def compute_all_splits(
    y_train: np.ndarray, prob_train: np.ndarray,
    y_test: np.ndarray, prob_test: np.ndarray,
    y_oot: np.ndarray | None = None, prob_oot: np.ndarray | None = None,
) -> pd.DataFrame:
    """Compute metrics across train/test/OOT splits.

    Returns a DataFrame with columns [metric, train, test, oot].
    """
    m_train = compute_metrics(y_train, prob_train, "Train")
    m_test = compute_metrics(y_test, prob_test, "Test")
    m_oot = compute_metrics(y_oot, prob_oot, "OOT") if y_oot is not None else {}

    rows = []
    for key in ["auc", "gini", "ks", "brier"]:
        rows.append({
            "metric": key.upper() if key != "brier" else "Brier Score",
            "train": m_train[key],
            "test": m_test[key],
            "oot": m_oot.get(key, np.nan),
        })
    return pd.DataFrame(rows)


def _ks_statistic(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Kolmogorov-Smirnov statistic. [Assumption] Standard credit scoring metric."""
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    return float(np.max(tpr - fpr))


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_roc_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    label: str = "Model",
    save_path: Path | None = None,
) -> Path:
    """ROC curve plot. [Paper Fig. 5]"""
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, lw=2, label=f"{label} (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)

    path = save_path or config.PLOTS_DIR / "roc_curve.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("ROC curve saved to %s", path)
    return path


def plot_lorenz_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    save_path: Path | None = None,
) -> Path:
    """Lorenz curve (cumulative bads vs score quantile). [Paper §2.4.1, Fig. 6]"""
    # Sort by predicted probability descending
    order = np.argsort(-y_prob)
    y_sorted = y_true[order]
    cum_bads = np.cumsum(y_sorted) / y_sorted.sum()
    x = np.arange(1, len(y_sorted) + 1) / len(y_sorted)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(x, cum_bads, lw=2, label="Model")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random")
    ax.set_xlabel("Fraction of Population (sorted by score)")
    ax.set_ylabel("Cumulative Fraction of Defaults")
    ax.set_title("Lorenz Curve")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)

    path = save_path or config.PLOTS_DIR / "lorenz_curve.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Lorenz curve saved to %s", path)
    return path


def plot_calibration_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
    label: str = "Model",
    save_path: Path | None = None,
) -> Path:
    """Calibration curve: predicted vs observed default rate by decile.

    [Paper §2.4.1, Fig. 7]
    """
    # Bin by predicted probability
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_idx = np.digitize(y_prob, bin_edges) - 1
    bin_idx = np.clip(bin_idx, 0, n_bins - 1)

    observed = []
    predicted = []
    for i in range(n_bins):
        mask = bin_idx == i
        if mask.sum() > 0:
            observed.append(y_true[mask].mean())
            predicted.append(y_prob[mask].mean())

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(predicted, observed, "o-", lw=2, label=label)
    ax.plot([0, max(predicted) * 1.1], [0, max(predicted) * 1.1], "k--", lw=1, label="Perfect")
    ax.set_xlabel("Predicted Default Rate")
    ax.set_ylabel("Observed Default Rate")
    ax.set_title("Calibration Curve")
    ax.legend()
    ax.grid(True, alpha=0.3)

    path = save_path or config.PLOTS_DIR / "calibration_curve.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Calibration curve saved to %s", path)
    return path


def plot_miv_selection(
    steps: list,
    save_path: Path | None = None,
) -> Path:
    """Plot AUC vs step and MIV bar chart. [Paper Fig. 5]

    Parameters
    ----------
    steps : list[MIVStep]
        MIV selection steps with step, miv, auc_train, auc_test attributes.
    """
    step_nums = [s.step for s in steps]
    auc_train = [s.auc_train for s in steps]
    auc_test = [s.auc_test for s in steps]
    miv_vals = [s.miv for s in steps]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # Top: AUC vs step
    ax1.plot(step_nums, auc_train, "b-o", label="AUC Train")
    ax1.plot(step_nums, auc_test, "r-o", label="AUC Test")
    ax1.set_ylabel("ROC AUC")
    ax1.set_title("MIV Feature Selection Progress")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Bottom: MIV bar chart (log scale)
    ax2.bar(step_nums, miv_vals, color="steelblue", alpha=0.8)
    ax2.set_xlabel("Selection Step")
    ax2.set_ylabel("MIV (log scale)")
    ax2.set_yscale("log")
    ax2.grid(True, alpha=0.3, axis="y")

    path = save_path or config.PLOTS_DIR / "miv_selection.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("MIV selection plot saved to %s", path)
    return path
