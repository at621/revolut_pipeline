"""Tests for binning and WoE module using small hand-crafted DataFrames."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from revolut_credit_risk.features.binning import (
    bin_features,
    transform_woe,
    BinningResults,
)


@pytest.fixture
def simple_data():
    """Hand-crafted DataFrame with known signal for WoE/IV testing."""
    rng = np.random.default_rng(42)
    n = 1000

    # Feature with clear signal: higher values -> more defaults
    risky_feature = rng.normal(0, 1, n)
    # Default probability increases with feature value
    p_default = 1 / (1 + np.exp(-(risky_feature - 0.5)))
    y = (rng.random(n) < p_default).astype(int)

    # Constant feature (should be skipped)
    constant = np.ones(n) * 5.0

    # Weak feature (random noise)
    noise = rng.normal(0, 1, n)

    X = pd.DataFrame({
        "risky": risky_feature,
        "constant": constant,
        "noise": noise,
    })
    y_series = pd.Series(y, name="target")

    return X, y_series


def test_bin_features_returns_results(simple_data):
    X, y = simple_data
    results = bin_features(X, y)

    assert isinstance(results, BinningResults)
    # "risky" should be binned successfully
    assert "risky" in results.results
    # "constant" should be skipped (< 2 unique values)
    assert "constant" not in results.results


def test_iv_positive_for_signal(simple_data):
    X, y = simple_data
    results = bin_features(X, y)

    # risky feature should have non-trivial IV
    iv = results.get_iv("risky")
    assert iv > 0.02, f"IV for 'risky' should be > 0.02, got {iv:.4f}"


def test_woe_transform_shape(simple_data):
    X, y = simple_data
    results = bin_features(X, y)

    woe_df = transform_woe(X, results)
    assert woe_df.shape[0] == X.shape[0]
    # WoE columns should be prefixed
    for col in woe_df.columns:
        assert col.startswith("woe_")


def test_woe_values_are_finite(simple_data):
    X, y = simple_data
    results = bin_features(X, y)
    woe_df = transform_woe(X, results)

    for col in woe_df.columns:
        assert np.all(np.isfinite(woe_df[col])), f"Inf/NaN in {col}"


def test_transform_woe_subset(simple_data):
    X, y = simple_data
    results = bin_features(X, y)
    woe_df = transform_woe(X, results, features=["risky"])

    assert "woe_risky" in woe_df.columns
    assert woe_df.shape[1] == 1
