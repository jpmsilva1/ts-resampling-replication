# Resampling Strategies for Imbalanced Time Series Forecasting

A replication of Moniz, Branco & Torgo (2017), *Resampling Strategies for Imbalanced Time
Series Forecasting*, Int J Data Sci Anal 3:161–181
([DOI](https://doi.org/10.1007/s41060-017-0044-3)), extended with two metrics the paper
predates (SERA, plain RMSE), critical-difference diagrams, Bayesian signed-rank tests, and
an exact-randomization causal analysis.

## Status: replicated

All 24 of the paper's datasets, 5 model families × 10 resampling strategies (+ `arima`,
`BDES`), 50 Monte Carlo folds each. The paper's central claim (H.1: resampling beats an
unresampled baseline under utility-based $F_1^\phi$) is confirmed, and every published
Table 3 cell reproduces the paper within ±1 dataset out of 24 — six cells exactly. See
`DIVERGENCE_ANALYSIS.md` for the full account, including two defects that were found and
fixed along the way (an R version-dependent RNG change, and a statistic bug in this
replication's own paired-comparison code) and why both had to be fixed together before the
result would reproduce.

## What's in this repo

| Path | What it is |
|---|---|
| `src/original/` | Snapshot of the paper's own released code (`nunompmoniz/TSResampStrat_JDSA2017`) |
| `src/adapted/` | The replication pipeline: `Exps.R` (the experiment driver), `run_config.R` (traceability knobs), `PairedComparisons.R` |
| `data/paper_datasets_csv/` | The 24 source datasets, plus `manifest.csv` mapping dataset id ↔ description |
| `Results (Clean)/` | Everything from tidy per-fold CSV onward: generator code (`Eval_Metrics Code/`), every table and figure, causal analysis |
| `runs/` | Registry of non-canonical experiment variants (see `runs/README.md`) |
| `DIVERGENCE_ANALYSIS.md` | Why some results disagreed with the paper, what was found, what was fixed |
| `REPLICATION_LOG.md` | Day-by-day log of getting the pipeline running |
| `Results (Clean)/EXPERIMENTAL_PROTOCOL.md` | The full operational protocol — read this before evaluating new results |

## From a clean checkout to every table and figure

Stage 0 (producing the raw `.Rdata`) needs the SLURM cluster the paper's own compute scale
requires; everything after that (Stage 1–5) runs locally from the checked-in CSVs, no
cluster access needed:

```bash
cd "Results (Clean)/Eval_Metrics Code"
python3 run_all.py --tests   # 58 tests, all green
python3 run_all.py           # regenerates every table and figure, ~5-10 min
```

To reproduce Stage 0 itself (only needed if you want to regenerate the raw `.Rdata`, not
to reproduce any table or figure already checked in): see `Results (Clean)/
EXPERIMENTAL_PROTOCOL.md` §2, and the `apuana` runbook it references for the SLURM
submission steps.

## What's excluded, and why

This repo tracks exactly what an independent reviewer needs to check the result and rerun
the evaluation — nothing more. Excluded, all mechanically regenerable from
`src/adapted/Exps.R` given cluster access: the `.Rdata` result files (~1.8 GB per run),
per-fold prediction CSVs (~4 GB), and the two GB-scale legacy artifact archives this project
accumulated along the way (`_archive/`, `_pre_phi_fix_backup_20260903/`). Also excluded, by
design rather than size: no compiled report or narrative paper lives in this repo — only the
code that produces every table and figure, and the tables and figures themselves.

What *is* tracked: all code, the 24 source dataset CSVs, the small derived per-iteration
CSVs (~17 MB), every table and figure this repo currently produces, and each run's
`RUN_MANIFEST.txt` (records exactly what code, RNG state, and hyperparameters produced it —
the traceability layer built during Tier 2, see `runs/README.md`).
