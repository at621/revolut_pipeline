"""Tests for MIV feature selection."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from revolut_credit_risk.features.binning import bin_features, transform_woe
from revolut_credit_risk.selection.information_value import collect_iv, filter_by_iv
from revolut_credit_risk.selection.miv_selector import (
    run_miv_selection,
    MIVSelectionResult,
    _compute_miv,
    _check_correlation,
)


@pytest.fixture
def miv_data():
    """Create WoE data and IV table for MIV selection testing."""
    rng = np.random.default_rng(42)
    n = 1000

    f1 = rng.normal(0, 1, n)
    f2 = rng.normal(0, 1, n)
    f3 = rng.normal(0, 1, n)
    # f1 is strongest signal, f2 is medium, f3 is weak
    latent = 0.8 * f1 + 0.4 * f2 + 0.1 * f3
    p = 1 / (1 + np.exp(-latent))
    y = (rng.random(n) < p).astype(int)

    X = pd.DataFrame({"f1": f1, "f2": f2, "f3": f3})
    y_series = pd.Series(y, name="target")

    # Split
    n_train = 700
    X_train, X_test = X.iloc[:n_train], X.iloc[n_train:]
    y_train, y_test = y_series.iloc[:n_train], y_series.iloc[n_train:]

    # Bin and WoE
    results = bin_features(X_train, y_train)
    X_woe_train = transform_woe(X_train, results)
    X_woe_test = transform_woe(X_test, results)

    iv_table = collect_iv(results)
    candidates = filter_by_iv(iv_table)

    return X_train, X_woe_train, y_train, X_woe_test, y_test, iv_table, results, candidates


def test_miv_selection_returns_result(miv_data):
    X_train, X_woe_train, y_train, X_woe_test, y_test, iv_table, results, candidates = miv_data
    result = run_miv_selection(
        X_woe_train, y_train, X_woe_test, y_test,
        iv_table, results, candidates,
        X_raw_train=X_train,
    )

    assert isinstance(result, MIVSelectionResult)
    assert len(result.selected_features) > 0
    assert len(result.steps) > 0


def test_first_step_is_highest_iv(miv_data):
    X_train, X_woe_train, y_train, X_woe_test, y_test, iv_table, results, candidates = miv_data
    result = run_miv_selection(
        X_woe_train, y_train, X_woe_test, y_test,
        iv_table, results, candidates,
        X_raw_train=X_train,
    )

    # First step should have the highest-IV feature
    highest_iv_feat = iv_table.iloc[0]["feature"]
    assert result.steps[0].feature_added == highest_iv_feat


def test_check_correlation():
    candidate = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    # Perfectly correlated -> should fail
    selected = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0, 5.0]})
    assert _check_correlation(candidate, selected) is False

    # Uncorrelated -> should pass
    selected2 = pd.DataFrame({"b": [5.0, 3.0, 1.0, 4.0, 2.0]})
    assert _check_correlation(candidate, selected2) is True
