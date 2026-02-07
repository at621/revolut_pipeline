"""Global constants and hyperparameters for the credit risk pipeline.

All configuration is centralised here. Each parameter is annotated with its
provenance: [Paper §X.Y.Z] for paper-sourced values, [Assumption] for our
engineering choices.
"""
from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
DATASET_PATH = PROJECT_ROOT / "data" / "datasets" / "synthetic_v1"  # [Assumption]
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
PLOTS_DIR = OUTPUTS_DIR / "plots"
VARIABLE_CONFIG_PATH = OUTPUTS_DIR / "variable_config.yaml"  # [Assumption]
LOG_FILE = OUTPUTS_DIR / "pipeline.log"
REPORT_FILE = OUTPUTS_DIR / "pipeline_report.md"

# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------
GENERATE_NEW_DATA: bool = True  # [Assumption] True = generate; False = load
RANDOM_SEED: int = 42  # [Assumption]
N_CUSTOMERS: int = 10_000  # [Assumption]
DEFAULT_RATE: float = 0.06  # [Assumption] ~6 % target default rate
PREDICTION_HORIZON_MONTHS: int = 12  # [Paper §2.2.3, Fig. 3]

# ---------------------------------------------------------------------------
# Deep Feature Synthesis
# ---------------------------------------------------------------------------
DFS_DEPTH: int = 2  # [Paper §2.2.4] recursive depth
DFS_MAX_FEATURES: int = 500  # [Assumption] cap for tractability

# ---------------------------------------------------------------------------
# Binning & WoE
# ---------------------------------------------------------------------------
BINNING_MONOTONIC_TREND: str = "auto_asc_desc"  # [Assumption] inspired by [Paper §2.2.5]
BINNING_MAX_N_BINS: int = 10  # [Assumption]
BINNING_MIN_BIN_SIZE: float = 0.05  # [Paper §2.2.5] "e.g., 5%"
BINNING_MIN_PREBIN_SIZE: float = 0.02  # [Assumption]
BINNING_SOLVER: str = "mip"  # [Assumption] mixed integer programming
BINNING_DIVERGENCE: str = "iv"  # [Paper §2.3.2]

# ---------------------------------------------------------------------------
# LLM variable config (optional)
# ---------------------------------------------------------------------------
USE_LLM_CONFIG: bool = False  # [Assumption] pipeline must work without LLM
LLM_MODEL: str = "claude-sonnet-4-5-20250929"  # [Assumption]

# ---------------------------------------------------------------------------
# Feature selection
# ---------------------------------------------------------------------------
IV_THRESHOLD: float = 0.02  # [Paper §2.3.2] "e.g. 2%"
MIV_THRESHOLD: float = 0.02  # [Paper §2.3.2] "e.g. 2%"
CORRELATION_THRESHOLD: float = 0.6  # [Paper §2.3.2] upper end of 40-60%
MAX_FEATURES: int = 20  # [Assumption]
MIV_PATIENCE: int = 2  # [Assumption] patience=2 for AUC plateau

# ---------------------------------------------------------------------------
# Scorecard
# ---------------------------------------------------------------------------
PDO: int = 20  # [Assumption] industry convention, per Siddiqi [Paper Ref. 2]
BASE_SCORE: int = 600  # [Assumption] industry convention
BASE_ODDS: int = 50  # [Assumption] industry convention

# ---------------------------------------------------------------------------
# Data splits
# ---------------------------------------------------------------------------
TRAIN_RATIO: float = 0.6  # [Assumption]
TEST_RATIO: float = 0.2  # [Assumption]
OOT_RATIO: float = 0.2  # [Assumption]
