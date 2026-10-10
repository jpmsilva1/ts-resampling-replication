# `runs/` — Tier 2 variant run registry

Each subdirectory here is one non-canonical experiment run, scaffolded by
`scratch/setup_run_root.sh <run_id>` and usable as an `EXP002_ROOT` override for
`Results (Clean)/Eval_Metrics Code/` (see `paths.py`).

A run's `.Rdata` live at `src/adapted/results/<run_id>/` (written by `Exps.R` when
submitted with `EXP002_RUN_ID=<run_id>`), next to that run's `RUN_MANIFEST.txt` —
the run/code/RNG/hyperparameter/SLURM-job record for exactly what produced those
files. `runs/<run_id>/` itself holds only the small CSVs derived from them.

See `~/.claude/plans/luminous-noodling-bear.md` for the full Tier 2 plan and
`DIVERGENCE_ANALYSIS.md` §§10–11 for what each experiment found.

## Runs

| run_id | Purpose | Status |
|---|---|---|
| `data_v4` | **Canonical since 2026-10-10.** `data_v3` with the paper's NA handling (`knnImputation`) on DS12 and DS24, from run `na_knn_rounding` (job 16745, RNG `Rounding`); assembled as symlinks by `src/stage0/build_data_v4.sh`, which also writes its `RUN_MANIFEST.txt`. Backs `Results (Clean)/` and `src/adapted/run_config.R`'s defaults | Canonical |
| `na_knn_rounding` | DS12 + DS24 rerun with `EXP002_NA_METHOD=knn`, `EXP002_SAMPLE_KIND=Rounding`, job 16745 | Complete; source of `data_v4` DS12/DS24 |
| `na_knn` | Same, job 16744 — ran with RNG `Rejection` (stale cluster `run_config.R`) and a row pre-filter later dropped | Discarded; kept for reference |
| `data_v3` | Canonical 2026-09-21 to 2026-10-10. Copy of `rng_rounding` promoted after its `.Rdata` (with `trues`/`preds`/`phi_ctrl` already captured) confirmed the replication; source of 22 of `data_v4`'s 24 datasets | Superseded by `data_v4` |
| `rng_rounding` | Tier 2 Experiment 1 (Cause A: R 3.6.0 RNG change) — `EXP002_SAMPLE_KIND=Rounding`, job 15570. Combined with the median-vs-mean Win/Loss sign fix (§11), reproduces the paper's Table 3 within ±1 dataset on all 15 cells | Complete; source of `data_v3` |
| `baseline_tuned` | Tier 2 Experiment 2 (Cause B: untuned baseline hyperparameters), job 15725 — moved SVM/RPART Win% *away* from the paper under the corrected statistic; Cause B refuted, not promoted | Complete; not canonical |

`data_v2` is the pre-promotion predecessor (not kept on this machine); it is
reproducible via `EXP002_RUN_ID=data_v2 EXP002_SAMPLE_KIND=Rejection
EXP002_NA_METHOD=complete.cases`. `data_v3` via `EXP002_RUN_ID=data_v3
EXP002_NA_METHOD=complete.cases`.
