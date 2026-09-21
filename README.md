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

## Reproducing every table and figure

Everything in this repo — every table, every figure, every statistic — regenerates locally
from the checked-in CSVs. No cluster, no GPU, no network, ~2–3 minutes:

```bash
pip install -r requirements.txt
cd "Results (Clean)/Eval_Metrics Code"
python3 run_all.py --tests   # 58 tests, all green
python3 run_all.py           # regenerates every table and figure
```

The only thing this *doesn't* reproduce is the raw `.Rdata` those CSVs were extracted
from — that requires re-running the R experiment itself on a SLURM cluster (the paper's own
scale of compute). Not needed to check any result already in this repo; see `Results
(Clean)/EXPERIMENTAL_PROTOCOL.md` §2 if you want to do it anyway.

## What's tracked, and what isn't

This repo tracks exactly what an independent reviewer needs to check the result and rerun
the evaluation — nothing more.

**Tracked:** all code, the 24 source dataset CSVs, every per-fold result (F1/RMSE/SERA and
the raw predictions behind them, ~4.3 GB), every table and figure this repo produces, and
each run's `RUN_MANIFEST.txt` (records exactly what code, RNG state, and hyperparameters
produced it — see `runs/README.md`).

**Excluded:**
- The raw `.Rdata` result files (~1.8 GB per run, individual files up to 464 MB). These are
  R's serialized form of the same numbers the tracked CSVs already hold at full per-fold
  granularity — nothing is lost by leaving them out, and several exceed GitHub's 100 MB
  per-file limit. (Regenerating them needs the cluster — see above.)
- Two legacy artifact archives this project accumulated along the way (`_archive/`,
  `_pre_phi_fix_backup_20260903/`) — superseded snapshots, not the current result.
- No compiled report or narrative paper — only the code that produces every table and
  figure, and the tables and figures themselves.
