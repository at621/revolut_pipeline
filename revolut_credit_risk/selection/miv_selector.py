"""Marginal Information Value (MIV) greedy forward feature selection.

[Paper §2.3.2] MIV is the paper's core feature selection method.  The formula,
greedy forward selection, stopping criteria, and correlation filter are all
described in §2.3.2.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import chi2
from sklearn.metrics import roc_auc_score

from revolut_credit_risk import config
from revolut_credit_risk.features.binning import BinningResults

logger = logging.getLogger(__name__)


@dataclass
class MIVStep:
    """Record of a single MIV selection step."""

    step: int
    feature_added: str
    miv: float
    auc_train: float
    auc_test: float


@dataclass
class MIVSelectionResult:
    """Result of the MIV forward selection process."""

    selected_features: list[str] = field(default_factory=list)
    steps: list[MIVStep] = field(default_factory=list)
    stopping_reason: str = ""


def run_miv_selection(
    X_woe_train: pd.DataFrame,
    y_train: pd.Series,
    X_woe_test: pd.DataFrame,
    y_test: pd.Series,
    iv_table: pd.DataFrame,
    binning_results: BinningResults,
    candidate_features: list[str],
    X_raw_train: pd.DataFrame | None = None,
) -> MIVSelectionResult:
    """Run greedy forward MIV feature selection.

    [Paper §2.3.2] Algorithm:
    1. Start with highest-IV feature.
    2. Iteratively add feature with highest MIV (residual info not yet captured).
    3. Stop when MIV < threshold, chi-square test fails, or AUC plateaus.

    Parameters
    ----------
    X_woe_train, X_woe_test : pd.DataFrame
        WoE-transformed features (columns prefixed with ``woe_``).
    y_train, y_test : pd.Series
        Binary target.
    iv_table : pd.DataFrame
        IV table with columns ``feature`` and ``iv``.
    binning_results : BinningResults
        Fitted binning objects.
    candidate_features : list[str]
        Features that passed the IV threshold filter.
    X_raw_train : pd.DataFrame, optional
        Raw (pre-WoE) training features, used for correct bin assignment
        in MIV computation.  If None, falls back to WoE values (legacy).

    Returns
    -------
    MIVSelectionResult
    """
    result = MIVSelectionResult()

    # Map from original feature name to WoE column name
    woe_cols = {f: f"woe_{f}" for f in candidate_features
                if f"woe_{f}" in X_woe_train.columns}
    available = set(woe_cols.keys())

    if not available:
        result.stopping_reason = "No candidate features available"
        logger.warning(result.stopping_reason)
        return result

    # --- Step 1: Select feature with highest IV ---
    # [Paper §2.3.2] "begins by selecting the feature with the highest individual IV"
    iv_sorted = iv_table[iv_table["feature"].isin(available)].sort_values(
        "iv", ascending=False
    )
    first_feat = iv_sorted.iloc[0]["feature"]
    first_iv = iv_sorted.iloc[0]["iv"]

    selected = [first_feat]
    available.discard(first_feat)

    # Train initial model
    auc_train, auc_test, model = _fit_and_score(
        X_woe_train, y_train, X_woe_test, y_test,
        [woe_cols[f] for f in selected],
    )

    result.steps.append(MIVStep(
        step=1, feature_added=first_feat, miv=first_iv,
        auc_train=auc_train, auc_test=auc_test,
    ))
    logger.info(
        "Step 1: Added '%s' (IV=%.4f, AUC train=%.4f, test=%.4f)",
        first_feat, first_iv, auc_train, auc_test,
    )

    # --- Iterative MIV selection ---
    best_test_auc = auc_test
    no_improve_count = 0

    for step_num in range(2, config.MAX_FEATURES + 1):
        if not available:
            result.stopping_reason = "No more candidate features"
            break

        # Compute MIV for each remaining candidate
        # [Paper §2.3.2] MIV formula
        pred_probs = model.predict(
            sm.add_constant(X_woe_train[[woe_cols[f] for f in selected]])
        )
        best_miv = -np.inf
        best_feat = None

        for feat in list(available):
            woe_col = woe_cols[feat]

            # [Paper §2.3.2] Correlation check
            corr_ok = _check_correlation(
                X_woe_train[woe_col],
                X_woe_train[[woe_cols[f] for f in selected]],
            )
            if not corr_ok:
                continue

            # Compute MIV using raw values for bin assignment
            # [PDtoolkit Ref] optb.transform(metric="bins") expects raw-scale values
            if X_raw_train is not None and feat in X_raw_train.columns:
                x_raw = X_raw_train[feat].values.astype(float)
            else:
                x_raw = X_woe_train[woe_col].values  # legacy fallback

            miv, p_val = _compute_miv(
                x_raw,
                y_train.values,
                pred_probs,
                binning_results.get_optb(feat),
            )

            # [PDtoolkit Ref] Require both MIV > threshold AND chi-square significance
            if p_val >= config.MIV_CHI2_P_VALUE:
                logger.debug(
                    "Skipping '%s': chi-square p=%.4f >= %.2f",
                    feat, p_val, config.MIV_CHI2_P_VALUE,
                )
                continue

            if miv > best_miv:
                best_miv = miv
                best_feat = feat

        if best_feat is None:
            result.stopping_reason = (
                "No remaining features pass correlation + chi-square filters"
            )
            break

        # [Paper §2.3.2] "MIV falls below a set threshold (e.g. 2%)"
        if best_miv < config.MIV_THRESHOLD:
            result.stopping_reason = f"MIV ({best_miv:.4f}) below threshold ({config.MIV_THRESHOLD})"
            break

        # Add feature and retrain
        selected.append(best_feat)
        available.discard(best_feat)

        auc_train, auc_test, model = _fit_and_score(
            X_woe_train, y_train, X_woe_test, y_test,
            [woe_cols[f] for f in selected],
        )

        result.steps.append(MIVStep(
            step=step_num, feature_added=best_feat, miv=best_miv,
            auc_train=auc_train, auc_test=auc_test,
        ))
        logger.info(
            "Step %d: Added '%s' (MIV=%.4f, AUC train=%.4f, test=%.4f)",
            step_num, best_feat, best_miv, auc_train, auc_test,
        )

        # [Paper §2.3.2] "model performance on a test set plateaus"
        # [Assumption] patience=2 consecutive steps
        if auc_test > best_test_auc:
            best_test_auc = auc_test
            no_improve_count = 0
        else:
            no_improve_count += 1
            if no_improve_count >= config.MIV_PATIENCE:
                result.stopping_reason = (
                    f"AUC plateau after step {step_num} "
                    f"(no improvement for {config.MIV_PATIENCE} steps)"
                )
                break

    result.selected_features = selected
    logger.info(
        "MIV selection complete: %d features selected. Reason: %s",
        len(selected), result.stopping_reason or "max features reached",
    )
    return result


def _fit_and_score(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    woe_cols: list[str],
) -> tuple[float, float, sm.Logit]:
    """Fit logistic regression and return AUC on train/test.

    [Paper §2.4.2] Logistic regression on WoE features.
    """
    X_tr = sm.add_constant(X_train[woe_cols])
    X_te = sm.add_constant(X_test[woe_cols])

    # [Common gotcha] Use bfgs if default fails to converge
    try:
        model = sm.Logit(y_train, X_tr).fit(disp=False, maxiter=100)
    except Exception:
        model = sm.Logit(y_train, X_tr).fit(disp=False, method="bfgs", maxiter=100)

    pred_train = model.predict(X_tr)
    pred_test = model.predict(X_te)

    auc_train = roc_auc_score(y_train, pred_train)
    auc_test = roc_auc_score(y_test, pred_test)

    # [Assumption] Warn if AUC < 0.5 — indicates inverted predictions
    if auc_test < 0.5:
        logger.warning(
            "Test AUC=%.4f < 0.5 — model predictions may be inverted. "
            "Check WoE signs and target encoding.",
            auc_test,
        )

    return auc_train, auc_test, model


def _compute_miv(
    x_raw: np.ndarray,
    y: np.ndarray,
    pred_probs: np.ndarray,
    optb,
) -> tuple[float, float]:
    """Compute Marginal Information Value for a candidate feature.

    [Paper §2.3.2] MIV formula:
    MIV(f) = SUM over bins [ (P(bin|Bad) - P(bin|Good)) * (WoE_observed - WoE_expected) ]

    WoE_expected is computed using the current model's predicted probabilities.

    [PDtoolkit Ref] Also computes a marginal chi-square (G-test) p-value
    to validate that the observed vs expected distributions are statistically
    significantly different.

    Parameters
    ----------
    x_raw : np.ndarray
        Raw (pre-WoE) feature values for correct bin assignment.
    y : np.ndarray
        Binary target.
    pred_probs : np.ndarray
        Current model's predicted probabilities.
    optb : OptimalBinning
        Fitted binning object.

    Returns
    -------
    tuple[float, float]
        (miv_value, chi_square_p_value)
    """
    # Get bin assignments using raw values (not WoE)
    try:
        bins = optb.transform(x_raw, metric="bins")
    except Exception:
        return 0.0, 1.0

    unique_bins = np.unique(bins)
    total_bad = y.sum()
    total_good = len(y) - total_bad

    if total_bad == 0 or total_good == 0:
        return 0.0, 1.0

    total_expected_bad = pred_probs.sum()
    total_expected_good = (1 - pred_probs).sum()

    miv = 0.0
    g_test_sum = 0.0
    n_valid_bins = 0

    for b in unique_bins:
        mask = bins == b
        if mask.sum() == 0:
            continue

        # Observed counts
        n_bad_obs = y[mask].sum()
        n_good_obs = mask.sum() - n_bad_obs
        p_bin_bad = n_bad_obs / total_bad if total_bad > 0 else 0
        p_bin_good = n_good_obs / total_good if total_good > 0 else 0

        # WoE observed: log(P(bin|Bad) / P(bin|Good))
        if p_bin_bad > 0 and p_bin_good > 0:
            woe_observed = np.log(p_bin_bad / p_bin_good)
        else:
            continue

        # Expected counts (from current model predictions)
        pred_in_bin = pred_probs[mask]
        n_bad_exp = pred_in_bin.sum()
        n_good_exp = (1 - pred_in_bin).sum()

        p_bin_bad_exp = n_bad_exp / total_expected_bad if total_expected_bad > 0 else 0
        p_bin_good_exp = n_good_exp / total_expected_good if total_expected_good > 0 else 0

        if p_bin_bad_exp > 0 and p_bin_good_exp > 0:
            woe_expected = np.log(p_bin_bad_exp / p_bin_good_exp)
        else:
            continue

        # [Paper §2.3.2] MIV contribution
        miv += (p_bin_bad - p_bin_good) * (woe_observed - woe_expected)

        # [PDtoolkit Ref] G-test contribution: n_obs * log(n_obs / n_exp)
        if n_good_exp > 0 and n_bad_exp > 0:
            g_good = n_good_obs * np.log(n_good_obs / n_good_exp) if n_good_obs > 0 else 0.0
            g_bad = n_bad_obs * np.log(n_bad_obs / n_bad_exp) if n_bad_obs > 0 else 0.0
            g_test_sum += g_good + g_bad

        n_valid_bins += 1

    # [PDtoolkit Ref] Chi-square p-value with df = n_bins - 1
    if n_valid_bins > 1:
        g_stat = 2.0 * g_test_sum
        p_val = 1.0 - chi2.cdf(g_stat, df=n_valid_bins - 1)
    else:
        p_val = 1.0

    miv_val = float(miv)
    if not np.isfinite(miv_val):
        miv_val = 0.0
        p_val = 1.0

    return miv_val, p_val


def _check_correlation(
    candidate: pd.Series,
    selected_woe: pd.DataFrame,
) -> bool:
    """Check that candidate is not too correlated with already-selected features.

    [Paper §2.3.2] "pairwise Pearson correlation threshold (e.g. 40%-60%)"
    [Assumption] We use 0.6 (upper end).
    """
    for col in selected_woe.columns:
        corr = candidate.corr(selected_woe[col])
        if abs(corr) > config.CORRELATION_THRESHOLD:
            return False
    return True
