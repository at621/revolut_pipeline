"""Logistic regression scorecard using statsmodels.

[Paper §2.1, §2.4.2] Rank-ordering application scorecard: "the MIV feature
selection technique is theoretically related to logistic regression."
[Assumption] Using statsmodels for full statistical inference output.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import roc_auc_score

from revolut_credit_risk import config
from revolut_credit_risk.features.binning import BinningResults

logger = logging.getLogger(__name__)


@dataclass
class ScorecardResult:
    """Container for the trained scorecard and diagnostics."""

    model: sm.BinaryResultsWrapper | None = None
    selected_features: list[str] = field(default_factory=list)
    woe_columns: list[str] = field(default_factory=list)
    summary_text: str = ""
    coefficient_table: pd.DataFrame | None = None
    scorecard_points: pd.DataFrame | None = None
    benchmark_table: pd.DataFrame | None = None


def train_scorecard(
    X_woe_train: pd.DataFrame,
    y_train: pd.Series,
    selected_features: list[str],
    binning_results: BinningResults,
) -> ScorecardResult:
    """Train logistic regression scorecard on WoE features.

    Parameters
    ----------
    X_woe_train : pd.DataFrame
        WoE-transformed training features.
    y_train : pd.Series
        Binary target.
    selected_features : list[str]
        Original feature names selected by MIV.
    binning_results : BinningResults
        For extracting WoE bin mappings for scorecard points.

    Returns
    -------
    ScorecardResult
    """
    result = ScorecardResult()
    result.selected_features = selected_features
    woe_cols = [f"woe_{f}" for f in selected_features]
    result.woe_columns = woe_cols

    # --- Fit logistic regression with statsmodels ---
    # [Paper §2.4.2] logistic regression
    X_const = sm.add_constant(X_woe_train[woe_cols])

    logit_model = sm.Logit(y_train, X_const)
    # [Common gotcha] try default MLE, fall back to bfgs if needed
    try:
        fit_result = logit_model.fit(disp=True, maxiter=100)
    except Exception:
        logger.warning("Default MLE failed, falling back to BFGS")
        fit_result = logit_model.fit(disp=True, method="bfgs", maxiter=100)

    result.model = fit_result

    # Full summary -> log and report
    result.summary_text = str(fit_result.summary())
    logger.info("\n%s", result.summary_text)

    # --- Coefficient sign check ---
    # [Paper §2.2.6] All WoE coefficients should be positive
    coefs = fit_result.params
    for col in woe_cols:
        if col in coefs.index and coefs[col] < 0:
            logger.warning(
                "Negative coefficient for '%s' (%.4f) — "
                "WoE features should have positive coefficients",
                col, coefs[col],
            )

    # --- Statistical significance check ---
    # [Assumption] Flag p-value > 0.05
    pvals = fit_result.pvalues
    for col in woe_cols:
        if col in pvals.index and pvals[col] > 0.05:
            logger.warning(
                "Feature '%s' not significant (p=%.4f > 0.05)", col, pvals[col]
            )

    # Build coefficient table
    result.coefficient_table = _build_coefficient_table(fit_result, woe_cols)

    # --- Score conversion ---
    # [Paper §2.1, Ref. 2, 3] Convert log-odds to points-based scorecard
    result.scorecard_points = _build_scorecard_points(
        fit_result, selected_features, binning_results
    )

    return result


def _build_coefficient_table(
    fit_result: sm.BinaryResultsWrapper,
    woe_cols: list[str],
) -> pd.DataFrame:
    """Extract coefficient table from statsmodels result."""
    params = fit_result.params
    pvals = fit_result.pvalues
    conf = fit_result.conf_int()

    rows = []
    for col in ["const"] + woe_cols:
        if col in params.index:
            rows.append({
                "variable": col,
                "coefficient": params[col],
                "p_value": pvals[col],
                "ci_lower": conf.loc[col, 0],
                "ci_upper": conf.loc[col, 1],
            })
    return pd.DataFrame(rows)


def _build_scorecard_points(
    fit_result: sm.BinaryResultsWrapper,
    selected_features: list[str],
    binning_results: BinningResults,
) -> pd.DataFrame:
    """Convert WoE model to a points-based scorecard.

    [Paper §2.1, Ref. 2, 3] Score = Offset + Factor * ln(odds)
    Factor = PDO / ln(2), Offset = BaseScore - Factor * ln(BaseOdds)
    [Assumption] PDO=20, BaseScore=600, BaseOdds=50:1
    """
    factor = config.PDO / np.log(2)
    offset = config.BASE_SCORE - factor * np.log(config.BASE_ODDS)

    params = fit_result.params
    n_features = len(selected_features)

    rows = []
    for feat in selected_features:
        woe_col = f"woe_{feat}"
        coef = params.get(woe_col, 0)
        if feat not in binning_results.results:
            continue

        optb = binning_results.get_optb(feat)
        bt = optb.binning_table.build()

        # For each bin, compute points
        for idx, row in bt.iterrows():
            bin_label = str(row.get("Bin", idx))
            if bin_label in ("Special", "Missing", "Totals"):
                continue
            woe_val = row.get("WoE", 0)
            try:
                woe_float = float(woe_val)
            except (TypeError, ValueError):
                continue
            if pd.isna(woe_float):
                continue

            # Points for this bin
            # Intercept contribution is split equally across features
            intercept_part = (offset + factor * params.get("const", 0)) / n_features
            points = intercept_part - factor * coef * woe_float

            rows.append({
                "feature": feat,
                "bin": bin_label,
                "woe": round(woe_float, 4),
                "points": round(float(points), 1),
            })

    df = pd.DataFrame(rows)
    logger.info("Scorecard: %d bin-point mappings across %d features", len(df), n_features)
    return df


def benchmark_models(
    X_train_raw: pd.DataFrame,
    y_train: pd.Series,
    X_test_raw: pd.DataFrame,
    y_test: pd.Series,
    X_oot_raw: pd.DataFrame | None = None,
    y_oot: pd.Series | None = None,
    lr_auc_train: float = 0.0,
    lr_auc_test: float = 0.0,
    lr_auc_oot: float = 0.0,
) -> pd.DataFrame:
    """Benchmark logistic regression against tree ensembles.

    [Paper §2.4.2, Table 1] "logistic regression models achieved a higher
    or equal Gini coefficient compared to tree ensembles."
    [Assumption] Tree models use raw (non-WoE) features for max flexibility.
    """
    # Fill NaN for tree models
    X_tr = X_train_raw.fillna(0)
    X_te = X_test_raw.fillna(0)
    X_oot_clean = X_oot_raw.fillna(0) if X_oot_raw is not None else None

    results = []

    # LR results (passed in from the scorecard)
    results.append({
        "model": "WoE Logistic Regression",
        "gini_train": 2 * lr_auc_train - 1,
        "gini_test": 2 * lr_auc_test - 1,
        "gini_oot": 2 * lr_auc_oot - 1 if lr_auc_oot else np.nan,
    })

    # Gradient Boosted Trees
    # [Paper §2.4.2] "tree ensembles (gradient boosted trees, random forests)"
    gbt = GradientBoostingClassifier(
        n_estimators=200, max_depth=4, random_state=config.RANDOM_SEED,
    )
    gbt.fit(X_tr, y_train)
    gbt_auc_train = roc_auc_score(y_train, gbt.predict_proba(X_tr)[:, 1])
    gbt_auc_test = roc_auc_score(y_test, gbt.predict_proba(X_te)[:, 1])
    gbt_auc_oot = (
        roc_auc_score(y_oot, gbt.predict_proba(X_oot_clean)[:, 1])
        if X_oot_clean is not None and y_oot is not None else np.nan
    )
    results.append({
        "model": "Gradient Boosted Trees",
        "gini_train": 2 * gbt_auc_train - 1,
        "gini_test": 2 * gbt_auc_test - 1,
        "gini_oot": 2 * gbt_auc_oot - 1,
    })

    # Random Forest
    rf = RandomForestClassifier(
        n_estimators=200, max_depth=8, random_state=config.RANDOM_SEED,
    )
    rf.fit(X_tr, y_train)
    rf_auc_train = roc_auc_score(y_train, rf.predict_proba(X_tr)[:, 1])
    rf_auc_test = roc_auc_score(y_test, rf.predict_proba(X_te)[:, 1])
    rf_auc_oot = (
        roc_auc_score(y_oot, rf.predict_proba(X_oot_clean)[:, 1])
        if X_oot_clean is not None and y_oot is not None else np.nan
    )
    results.append({
        "model": "Random Forest",
        "gini_train": 2 * rf_auc_train - 1,
        "gini_test": 2 * rf_auc_test - 1,
        "gini_oot": 2 * rf_auc_oot - 1,
    })

    df = pd.DataFrame(results)
    logger.info("Benchmark results:\n%s", df.to_string(index=False))
    return df
