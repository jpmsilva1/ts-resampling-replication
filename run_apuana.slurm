#!/bin/bash
#SBATCH --job-name=exp002_ts_resampling
#SBATCH --output=logs/exp002_%j.out
#SBATCH --error=logs/exp002_%j.err
#SBATCH --partition=short-simple
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=128G
#SBATCH --time=2-00:00:00
#SBATCH --requeue

# Navigate to experiment root (safe regardless of where sbatch was called from)
cd $SLURM_SUBMIT_DIR

# Tier 2 traceability: RUN_ID names this run's results dir. Defaults to
# data_v2 (today's canonical run) for a plain `sbatch run_apuana.slurm`.
# For a variant run, submit with --export=ALL,EXP002_RUN_ID=<id>[,EXP002_SAMPLE_KIND=...]
# -- see the step-by-step in the Tier 2 plan. #SBATCH directives above are
# fixed at submit time (SLURM parses them before this shell runs), so the
# job name/log filenames stay generic (%j = job ID); RUN_MANIFEST.txt
# (written by src/adapted/run_config.R) is the source of truth linking a
# job ID to its run_id -- this banner just makes the .out log self-identifying.
RUN_ID="${EXP002_RUN_ID:-data_v2}"

# Ensure output directories exist
mkdir -p logs
mkdir -p "src/adapted/results/${RUN_ID}"

# Prevent R's BLAS/OpenMP backends from oversubscribing threads inside each
# of the 48 forked parallelMap workers (nested parallelism can thrash the node).
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

# Activate micromamba R environment
export PATH="$HOME/.local/bin:$PATH"
eval "$(micromamba shell hook --shell bash)"

if ! micromamba env list | grep -q "exp002_env"; then
    echo "ERROR: micromamba env 'exp002_env' not found. Run setup_r_env.sh first."
    exit 1
fi
micromamba activate exp002_env

TOTAL=24

echo "=========================================================="
echo "Exp002: TS Resampling Strategies (full 24-dataset rerun, 48-core parallel)"
echo "RUN_ID: $RUN_ID"
echo "EXP002_SAMPLE_KIND: ${EXP002_SAMPLE_KIND:-(default)}"
echo "EXP002_BASELINE_PARAMS: ${EXP002_BASELINE_PARAMS:-(none)}"
echo "Submission dir: $SLURM_SUBMIT_DIR"
echo "SLURM job ID: $SLURM_JOB_ID"
echo "CPUs allocated: $SLURM_CPUS_PER_TASK"
echo "=========================================================="

# Run the adapted R script (picks up EXP002_RUN_ID/EXP002_SAMPLE_KIND/
# EXP002_BASELINE_PARAMS from this job's environment via run_config.R)
Rscript src/adapted/Exps.R

echo "=========================================================="
echo "Exp002 R script finished. Checking completeness..."
COMPLETED=$(ls "src/adapted/results/${RUN_ID}"/results_dataset_*.Rdata 2>/dev/null | wc -l)
echo "Datasets completed: $COMPLETED / $TOTAL"
if [ "$COMPLETED" -eq "$TOTAL" ]; then
    echo "All $TOTAL datasets done. Run merge_results.R to consolidate."
else
    echo "WARNING: Only $COMPLETED/$TOTAL datasets completed. Resubmit to resume from checkpoint."
fi
echo "=========================================================="
