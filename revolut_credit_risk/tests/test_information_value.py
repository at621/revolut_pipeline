"""Tests for Information Value calculation."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from revolut_credit_risk.features.binning import bin_features, transform_woe
from revolut_credit_risk.selection.information_value import (
    iv_strength,
    collect_iv,
    filter_by_iv,
    bivariate_analysis,
)


def test_iv_strength_categories():
    assert iv_strength(0.01) == "Poor"
    assert iv_strength(0.02) == "Poor"  # boundary: <= 0.02
    assert iv_strength(0.05) == "Weak"
    assert iv_strength(0.15) == "Medium"
    assert iv_strength(0.35) == "Strong"
    assert iv_strength(0.60) == "Very Strong"


@pytest.fixture
def binned_data():
    """Create binning results from hand-crafted data."""
    rng = np.random.default_rng(42)
    n = 1000

    strong_signal = rng.normal(0, 1, n)
    p = 1 / (1 + np.exp(-(strong_signal - 0.5)))
    y = (rng.random(n) < p).astype(int)

    weak_signal = rng.normal(0, 1, n)

    X = pd.DataFrame({
        "strong": strong_signal,
        "weak": weak_signal,
    })
    y_series = pd.Series(y, name="target")

    results = bin_features(X, y_series)
    woe_df = transform_woe(X, results)
    return results, woe_df, y_series


def test_collect_iv_sorted(binned_data):
    results, _, _ = binned_data
    iv_table = collect_iv(results)

    assert "feature" in iv_table.columns
    assert "iv" in iv_table.columns
    # Should be sorted descending
    ivs = iv_table["iv"].values
    assert all(ivs[i] >= ivs[i + 1] for i in range(len(ivs) - 1))


def test_filter_by_iv(binned_data):
    results, _, _ = binned_data
    iv_table = collect_iv(results)
    passing = filter_by_iv(iv_table, threshold=0.02)

    assert isinstance(passing, list)
    # "strong" should pass
    assert "strong" in passing


def test_bivariate_analysis(binned_data):
    results, woe_df, y = binned_data
    biv = bivariate_analysis(results, woe_df, y)

    assert "feature" in biv.columns
    assert "iv" in biv.columns
    assert "univariate_gini" in biv.columns
    assert len(biv) > 0
