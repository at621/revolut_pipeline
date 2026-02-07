"""Tests for PD calibration."""
from __future__ import annotations

import numpy as np
import pytest

from revolut_credit_risk.model.calibration import (
    calibrate_platt,
    calibrate_isotonic,
    CalibrationResult,
)


@pytest.fixture
def calibration_data():
    """Generate raw model scores and targets for calibration testing."""
    rng = np.random.default_rng(42)
    n_train, n_test = 500, 200

    scores_train = rng.beta(2, 5, n_train)  # right-skewed like real PD
    y_train = (rng.random(n_train) < scores_train).astype(int)

    scores_test = rng.beta(2, 5, n_test)
    y_test = (rng.random(n_test) < scores_test).astype(int)

    return scores_train, y_train, scores_test, y_test


def test_platt_scaling(calibration_data):
    scores_train, y_train, scores_test, y_test = calibration_data
    result = calibrate_platt(scores_train, y_train, scores_test, y_test)

    assert isinstance(result, CalibrationResult)
    assert result.method == "Platt Scaling"
    assert result.calibrated_probs_test is not None
    assert len(result.calibrated_probs_test) == len(y_test)
    # Calibrated probs should be in [0, 1]
    assert np.all(result.calibrated_probs_test >= 0)
    assert np.all(result.calibrated_probs_test <= 1)


def test_isotonic_regression(calibration_data):
    scores_train, y_train, scores_test, y_test = calibration_data
    result = calibrate_isotonic(scores_train, y_train, scores_test, y_test)

    assert isinstance(result, CalibrationResult)
    assert result.method == "Isotonic Regression"
    assert result.calibrated_probs_test is not None
    assert len(result.calibrated_probs_test) == len(y_test)


def test_brier_score_computed(calibration_data):
    scores_train, y_train, scores_test, y_test = calibration_data
    result = calibrate_isotonic(scores_train, y_train, scores_test, y_test)

    assert result.brier_before > 0
    assert result.brier_after > 0
    assert result.brier_before < 1
    assert result.brier_after < 1
