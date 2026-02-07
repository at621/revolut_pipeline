"""Tests for scorecard model."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from revolut_credit_risk.features.binning import bin_features, transform_woe
from revolut_credit_risk.model.scorecard import train_scorecard, benchmark_models, ScorecardResult


@pytest.fixture
def scorecard_data():
    """Create WoE-transformed data for scorecard testing."""
    rng = np.random.default_rng(42)
    n = 800

    f1 = rng.normal(0, 1, n)
    f2 = rng.normal(0, 1, n)
    latent = 0.6 * f1 + 0.3 * f2
    p = 1 / (1 + np.exp(-latent))
    y = (rng.random(n) < p).astype(int)

    X = pd.DataFrame({"f1": f1, "f2": f2})
    y_series = pd.Series(y, name="target")

    results = bin_features(X, y_series)
    X_woe = transform_woe(X, results)

    return X_woe, y_series, ["f1", "f2"], results


def test_train_scorecard(scorecard_data):
    X_woe, y, features, binning_results = scorecard_data
    result = train_scorecard(X_woe, y, features, binning_results)

    assert isinstance(result, ScorecardResult)
    assert result.model is not None
    assert len(result.summary_text) > 0
    assert result.coefficient_table is not None


def test_coefficients_in_table(scorecard_data):
    X_woe, y, features, binning_results = scorecard_data
    result = train_scorecard(X_woe, y, features, binning_results)

    coef_table = result.coefficient_table
    assert "const" in coef_table["variable"].values
    assert "woe_f1" in coef_table["variable"].values


def test_scorecard_points_generated(scorecard_data):
    X_woe, y, features, binning_results = scorecard_data
    result = train_scorecard(X_woe, y, features, binning_results)

    assert result.scorecard_points is not None
    assert len(result.scorecard_points) > 0
    assert "feature" in result.scorecard_points.columns
    assert "points" in result.scorecard_points.columns


def test_benchmark_models():
    rng = np.random.default_rng(42)
    n = 500
    X_train = pd.DataFrame({"a": rng.normal(0, 1, n), "b": rng.normal(0, 1, n)})
    y_train = pd.Series((rng.random(n) > 0.5).astype(int))
    X_test = pd.DataFrame({"a": rng.normal(0, 1, 200), "b": rng.normal(0, 1, 200)})
    y_test = pd.Series((rng.random(200) > 0.5).astype(int))

    bench = benchmark_models(
        X_train, y_train, X_test, y_test,
        lr_auc_train=0.75, lr_auc_test=0.72,
    )

    assert len(bench) == 3
    assert "WoE Logistic Regression" in bench["model"].values
    assert "Gradient Boosted Trees" in bench["model"].values
    assert "Random Forest" in bench["model"].values
