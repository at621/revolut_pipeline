"""PD calibration — Platt scaling and isotonic regression.

[Paper §2.1, §2.4.1] "A separate calibration step is used to transform the
rank-ordering scores into accurate PD estimates."
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss

logger = logging.getLogger(__name__)


@dataclass
class CalibrationResult:
    """Container for calibration outputs."""

    method: str
    brier_before: float
    brier_after: float
    calibrated_probs_test: np.ndarray | None = None
    calibrator: object = None


def calibrate_platt(
    raw_scores_train: np.ndarray,
    y_train: np.ndarray,
    raw_scores_test: np.ndarray,
    y_test: np.ndarray,
) -> CalibrationResult:
    """Platt scaling: logistic regression on raw scores vs observed defaults.

    [Paper §2.4.1, Ref. 10, 12] Platt (1999).
    """
    brier_before = brier_score_loss(y_test, raw_scores_test)

    # Fit logistic regression: score -> P(default)
    lr = LogisticRegression(max_iter=1000)
    lr.fit(raw_scores_train.reshape(-1, 1), y_train)
    calibrated = lr.predict_proba(raw_scores_test.reshape(-1, 1))[:, 1]

    brier_after = brier_score_loss(y_test, calibrated)

    logger.info(
        "Platt scaling: Brier before=%.4f, after=%.4f",
        brier_before, brier_after,
    )

    return CalibrationResult(
        method="Platt Scaling",
        brier_before=brier_before,
        brier_after=brier_after,
        calibrated_probs_test=calibrated,
        calibrator=lr,
    )


def calibrate_isotonic(
    raw_scores_train: np.ndarray,
    y_train: np.ndarray,
    raw_scores_test: np.ndarray,
    y_test: np.ndarray,
) -> CalibrationResult:
    """Isotonic regression calibration.

    [Paper §2.4.1, Ref. 12] "isotonic regression being preferred for larger
    datasets, as it can capture non-monotonic relationships."
    """
    brier_before = brier_score_loss(y_test, raw_scores_test)

    iso = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip")
    iso.fit(raw_scores_train, y_train)
    calibrated = iso.predict(raw_scores_test)

    brier_after = brier_score_loss(y_test, calibrated)

    logger.info(
        "Isotonic regression: Brier before=%.4f, after=%.4f",
        brier_before, brier_after,
    )

    return CalibrationResult(
        method="Isotonic Regression",
        brier_before=brier_before,
        brier_after=brier_after,
        calibrated_probs_test=calibrated,
        calibrator=iso,
    )
