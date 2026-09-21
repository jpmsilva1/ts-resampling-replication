#!/usr/bin/env python3
"""Single source of truth for every path the evaluation pipeline touches.

Before this existed, 14 modules each opened with

    ROOT = Path("/Users/joaopms/Documents/002_ts_resampling_strategies_isolated")

which meant the pipeline only ran on one laptop. ROOT is now derived from this
file's own location, so a clone anywhere works with no edit. Set the
EXP002_ROOT environment variable to override (useful when the results tree is
mounted somewhere other than the code tree).

Import as `from paths import RESULTS_DATA, LATEX_DIR, FIG_DIR` -- every caller
already puts `Eval_Metrics Code/` on sys.path to reach tables_common.
"""
import os
from pathlib import Path

# .../<ROOT>/Results (Clean)/Eval_Metrics Code/paths.py -> parents[2] is ROOT
ROOT = Path(os.environ.get("EXP002_ROOT", Path(__file__).resolve().parents[2]))

RESULTS_CLEAN = ROOT / "Results (Clean)"
RESULTS_DATA = RESULTS_CLEAN / "Results Data"
EVAL_METRICS = RESULTS_CLEAN / "Evaluation Metrics"
CODE_DIR = RESULTS_CLEAN / "Eval_Metrics Code"

LATEX_DIR = EVAL_METRICS / "Latex Tables"
FIG_DIR = EVAL_METRICS / "Figures"
CAUSAL_DIR = EVAL_METRICS / "Causal"
REGRESSION_DIR = EVAL_METRICS / "Regression"

# Stage-1 inputs, produced by the R extraction scripts in scratch/.
RAW_F1 = RESULTS_DATA / "raw_iterations_by_dataset_v2"
RAW_PREDICTIONS = RESULTS_DATA / "raw_predictions_by_dataset"
RAW_RMSE = RESULTS_DATA / "raw_rmse_by_dataset"
RAW_SERA = RESULTS_DATA / "raw_sera_by_dataset"

# The pre-cap_svm_train extraction. Kept so previously-published F1 tables stay
# reproducible; NOT an input to any current generator. See CLAUDE.md 2026-08-26.
RAW_F1_LEGACY = RESULTS_DATA / "raw_iterations_by_dataset"

DATA_DIR = ROOT / "data/paper_datasets_csv"
MANIFEST = DATA_DIR / "manifest.csv"


def require(path: Path) -> Path:
    """Fail loudly and early when a stage's input is missing, instead of
    letting a generator emit a silently empty table."""
    if not path.exists():
        raise FileNotFoundError(
            f"{path} does not exist. Check EXP002_ROOT (currently {ROOT}), "
            f"and that the upstream stage in EXPERIMENTAL_PROTOCOL.md has run."
        )
    return path
