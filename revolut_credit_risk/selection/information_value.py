"""Information Value (IV) calculation and bivariate analysis.

[Paper §2.3.2] IV formula: IV(X) = SUM [ (P(bin|Bad) - P(bin|Good)) * WoE(bin) ]
IV thresholds from Siddiqi [Paper Ref. 2]:
  <= 0.02  Poor
  0.02-0.1 Weak
  0.1-0.3  Medium
  0.3-0.5  Strong
  > 0.5    Very Strong (suspect)
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
import statsmodels.api as sm

from revolut_credit_risk import config
from revolut_credit_risk.features.binning import BinningResults

logger = logging.getLogger(__name__)


def iv_strength(iv: float) -> str:
    """Classify IV into strength categories. [Paper §2.3.2, Ref. 2]"""
    if iv <= 0.02:
        return "Poor"
    elif iv <= 0.1:
        return "Weak"
    elif iv <= 0.3:
        return "Medium"
    elif iv <= 0.5:
        return "Strong"
    else:
        return "Very Strong"


def collect_iv(binning_results: BinningResults) -> pd.DataFrame:
    """Collect IV from all fitted binning objects and rank features.

    [Paper §2.3.2] IV is computed directly by optbinning during binning.

    Returns
    -------
    pd.DataFrame
        Columns: feature, iv, iv_strength, n_bins, sorted descending by IV.
    """
    rows = []
    for name, res in binning_results.results.items():
        rows.append({
            "feature": name,
            "iv": res.iv,
            "iv_strength": iv_strength(res.iv),
            "n_bins": res.n_bins,
        })

    df = pd.DataFrame(rows).sort_values("iv", ascending=False).reset_index(drop=True)

    # Log summary
    counts = df["iv_strength"].value_counts()
    logger.info(
        "%d features analysed: %s",
        len(df),
        ", ".join(f"{counts.get(s, 0)} {s}" for s in
                  ["Strong", "Medium", "Weak", "Poor", "Very Strong"]),
    )
    return df


def filter_by_iv(
    iv_table: pd.DataFrame,
    threshold: float | None = None,
) -> list[str]:
    """Return feature names with IV >= threshold.

    [Paper §2.3.2] Pre-filter features with IV < 0.02 before MIV selection.
    """
    threshold = threshold or config.IV_THRESHOLD
    passing = iv_table[iv_table["iv"] >= threshold]
    logger.info(
        "Features passing IV >= %.3f: %d / %d",
        threshold, len(passing), len(iv_table),
    )
    return passing["feature"].tolist()


def bivariate_analysis(
    binning_results: BinningResults,
    X_woe: pd.DataFrame,
    y: pd.Series,
) -> pd.DataFrame:
    """Compute univariate Gini alongside IV for every binned feature.

    [Assumption] Bivariate analysis table is standard credit risk practice.
    Gini = 2 * AUC - 1  [Paper §2.4.1]

    Parameters
    ----------
    binning_results : BinningResults
        Fitted binning objects with IV.
    X_woe : pd.DataFrame
        WoE-transformed features (columns prefixed with ``woe_``).
    y : pd.Series
        Binary target variable.

    Returns
    -------
    pd.DataFrame
        Sorted by IV descending: feature, iv, iv_strength, univariate_gini, n_bins.
    """
    rows = []
    for name, res in binning_results.results.items():
        woe_col = f"woe_{name}"
        gini = np.nan
        if woe_col in X_woe.columns:
            woe_vals = X_woe[woe_col].values
            # Only compute Gini if there's variance in the WoE values
            if np.std(woe_vals) > 1e-9:
                try:
                    auc = roc_auc_score(y, woe_vals)
                    gini = 2 * auc - 1  # [Paper §2.4.1]
                except ValueError:
                    pass

        rows.append({
            "feature": name,
            "iv": res.iv,
            "iv_strength": iv_strength(res.iv),
            "univariate_gini": gini,
            "n_bins": res.n_bins,
        })

    df = pd.DataFrame(rows).sort_values("iv", ascending=False).reset_index(drop=True)
    df.index = df.index + 1  # 1-based ranking
    df.index.name = "#"

    logger.info(
        "Bivariate analysis: %d features, top IV=%.4f (%s)",
        len(df),
        df["iv"].iloc[0] if len(df) > 0 else 0,
        df["feature"].iloc[0] if len(df) > 0 else "N/A",
    )
    return df
