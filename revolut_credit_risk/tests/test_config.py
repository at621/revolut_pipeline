"""Tests for config module."""
from __future__ import annotations

from pathlib import Path

from revolut_credit_risk import config


def test_paths_are_pathlib():
    assert isinstance(config.PROJECT_ROOT, Path)
    assert isinstance(config.DATASET_PATH, Path)
    assert isinstance(config.OUTPUTS_DIR, Path)


def test_default_values():
    assert config.RANDOM_SEED == 42
    assert config.N_CUSTOMERS == 10_000
    assert 0 < config.DEFAULT_RATE < 1
    assert config.DFS_DEPTH == 2
    assert config.IV_THRESHOLD == 0.02
    assert config.MIV_THRESHOLD == 0.02
    assert config.CORRELATION_THRESHOLD == 0.6
    assert config.TRAIN_RATIO + config.TEST_RATIO + config.OOT_RATIO == 1.0


def test_use_llm_config_is_false_by_default():
    assert config.USE_LLM_CONFIG is False
