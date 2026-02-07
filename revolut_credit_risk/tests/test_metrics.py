"""Tests for evaluation metrics."""
from __future__ import annotations

import numpy as np
import pytest

from revolut_credit_risk.evaluation.metrics import (
    compute_metrics,
    compute_all_splits,
    _ks_statistic,
)


def test_compute_metrics():
    rng = np.random.default_rng(42)
    y_true = np.array([0, 0, 0, 1, 1, 1, 0, 1, 0, 1])
    y_prob = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9, 0.4, 0.6, 0.15, 0.85])

    metrics = compute_metrics(y_true, y_prob)

    assert "auc" in metrics
    assert "gini" in metrics
    assert "ks" in metrics
    assert "brier" in metrics
    assert 0 < metrics["auc"] <= 1
    assert metrics["gini"] == pytest.approx(2 * metrics["auc"] - 1)


def test_gini_formula():
    """Gini = 2 * AUC - 1 [Paper §2.4.1]"""
    y_true = np.array([0, 0, 1, 1])
    y_prob = np.array([0.1, 0.4, 0.6, 0.9])  # perfect ranking
    metrics = compute_metrics(y_true, y_prob)

    assert metrics["auc"] == 1.0
    assert metrics["gini"] == 1.0


def test_ks_statistic():
    y_true = np.array([0, 0, 0, 1, 1, 1])
    y_prob = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    ks = _ks_statistic(y_true, y_prob)
    assert 0 < ks <= 1.0


def test_compute_all_splits():
    rng = np.random.default_rng(42)
    y = (rng.random(100) > 0.5).astype(int)
    p = rng.random(100)

    df = compute_all_splits(y, p, y, p)
    assert len(df) == 4
    assert "metric" in df.columns
    assert "train" in df.columns
    assert "test" in df.columns
