#!/bin/bash
#SBATCH --job-name=exp002_baseline_search
#SBATCH --output=logs/exp002_baseline_search_%j.out
#SBATCH --error=logs/exp002_baseline_search_%j.err
#SBATCH --partition=short-simple
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=128G
#SBATCH --time=2-00:00:00
#SBATCH --requeue

# Tier 2 Experiment 2, Phase 2a (~/.claude/plans/luminous-noodling-bear.md):
# baseline-only hyperparameter search for SVM/RPART. Much cheaper than the
# main Exps.R run (2 workflows x 15 combos x nReps=10, vs. 52 workflows x
# nReps=50, no auto.arima) -- kept on the same partition/cpu/mem envelope
# as run_apuana.slurm for consistency and headroom, not because it needs it.

cd $SLURM_SUBMIT_DIR
mkdir -p logs

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

export PATH="$HOME/.local/bin:$PATH"
eval "$(micromamba shell hook --shell bash)"

if ! micromamba env list | grep -q "exp002_env"; then
    echo "ERROR: micromamba env 'exp002_env' not found. Run setup_r_env.sh first."
    exit 1
fi
micromamba activate exp002_env

echo "=========================================================="
echo "Exp002 Tier 2 Experiment 2: baseline-only SVM/RPART hyperparameter search"
echo "Submission dir: $SLURM_SUBMIT_DIR"
echo "SLURM job ID: $SLURM_JOB_ID"
echo "CPUs allocated: $SLURM_CPUS_PER_TASK"
echo "=========================================================="

# Resumable via its own checkpoint CSV (src/adapted/baseline_params_svm_rpart.csv) --
# a requeue re-enters the same per-dataset skip-if-present loop, same pattern
# as Exps.R's .Rdata checkpoint.
Rscript src/adapted/BaselineParamSearch.R

echo "=========================================================="
echo "Search finished. Checking completeness..."
COMPLETED=$(python3 -c "import csv,sys; print(sum(1 for _ in csv.DictReader(open('src/adapted/baseline_params_svm_rpart.csv'))))" 2>/dev/null || echo 0)
echo "Datasets searched: $COMPLETED / 24"
if [ "$COMPLETED" -eq 24 ]; then
    echo "All 24 datasets searched. Pull src/adapted/baseline_params_svm_rpart.csv back and submit Experiment 2's rerun."
else
    echo "WARNING: Only $COMPLETED/24 datasets searched. Resubmit to resume from checkpoint."
fi
echo "=========================================================="
