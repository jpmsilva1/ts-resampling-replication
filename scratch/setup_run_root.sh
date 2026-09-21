#!/usr/bin/env bash
# Scaffolds runs/<run_id>/ as a valid EXP002_ROOT (see Results (Clean)/Eval_Metrics
# Code/paths.py) for evaluating a Tier 2 variant run without touching the canonical
# tree or its "Results (Clean)/" output. Only creates what
# Paper Comparison's generate_paper_comparison.py actually reads (data/
# paper_datasets_csv for the manifest, raw_iterations_by_dataset_v2 for F1)
# -- Tier 2's decision gate (see the plan) only needs that one generator, not
# the full 22-generator pipeline.
#
# Usage: scratch/setup_run_root.sh <run_id>
# Then:  Rscript src/stage0/rdata_to_csv.R "src/adapted/results/<run_id>" \
#          "runs/<run_id>/Results (Clean)/Results Data/raw_iterations_by_dataset_v2" \
#          data/paper_datasets_csv/manifest.csv
#        EXP002_ROOT="$PWD/runs/<run_id>" python3 \
#          "Results (Clean)/Eval_Metrics Code/Paper Comparison/generate_paper_comparison.py"
set -euo pipefail
RUN_ID="${1:?usage: setup_run_root.sh <run_id>}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_ROOT="$ROOT/runs/$RUN_ID"

mkdir -p "$RUN_ROOT/data"
mkdir -p "$RUN_ROOT/Results (Clean)/Results Data/raw_iterations_by_dataset_v2"

# Relative target, not $ROOT/... -- an absolute symlink breaks the moment this
# folder is cloned onto a different machine (self-contained-project rule).
ln -sfn "../../../data/paper_datasets_csv" "$RUN_ROOT/data/paper_datasets_csv"

echo "Scaffolded $RUN_ROOT"
