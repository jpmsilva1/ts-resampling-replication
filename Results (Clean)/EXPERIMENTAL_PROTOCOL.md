# Experimental Evaluation Protocol

**Scope.** How to take raw experimental results and turn them into every table,
figure and statistic this project publishes — reproducibly, in a fixed order,
with no discretionary choices left to the person running it.

**Audience.** Someone on a different computer who has never seen this project.
Follow this file top to bottom and you will reproduce the full evaluation.

**One-line answer to "does `Results (Clean)/` contain the whole setup?"**
Almost. `Results (Clean)/` owns everything from *tidy CSV* onward — Stages 1–5
below. The two steps *before* that (running the R experiment on the cluster, and
unserializing its `.Rdata` into CSV) live outside this folder, in
`src/adapted/`, `run_apuana.slurm` and `src/stage0/`. They are Stage 0 here, and
they are part of the protocol.

---

## 0. Ground rules — read before running anything

These exist because each one has already cost this project a wrong number.

1. **Never hardcode the dataset count.** Derive it from what is on disk
   (`tables_common.datasets_on_disk()`, `paired_comparisons.dataset_ids()`,
   `panel["dataset_id"].nunique()`). Seven separate "hardcoded 20" bugs have been
   found; each silently dropped or mis-scaled an added dataset instead of
   erroring. `run_all.py --tests` guards this class.
2. **Never hardcode an absolute path.** `Eval_Metrics Code/paths.py` is the single
   source of truth; override it with the `EXP002_ROOT` environment variable.
   There are zero `/Users/...` literals anywhere in the pipeline.
3. **Run the stages in order.** Stage 1 *writes* the inputs Stages 3–5 read.
   `run_all.py` enforces this; do not run generators individually unless you know
   which stage they are in.
4. **Regenerate everything, never a subset**, when the source data changes. A
   partially regenerated `Evaluation Metrics/` mixes runs and is the single
   easiest way to publish an inconsistent table.
5. **Adding a dataset means editing exactly two catalogues** — see §7. Adding it
   to one and not the other is caught by the test suite, not by silence.
6. **Take a shasum manifest before regenerating** (§6). "Which numbers moved" is a
   question you will be asked, and it is only answerable if you measured.

---

## 1. Environment

### Local (evaluation — Stages 1–5, Python)

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # numpy, pandas, scipy, matplotlib, statsmodels, pytest
```

Python ≥ 3.10 (the code uses `X | None` type syntax and `str.removeprefix`).
No R is needed for Stages 1–5. No GPU, no network. Full pipeline runs in ~2.5 min.

### Cluster (Stage 0 — the R experiment)

R with: `operators`, `gdata`, `parallelMap`, and from GitHub
`cran/performanceEstimation`, `cran/DMwR`, `cran/DMwR2`, `cran/UBL`,
`rpribeiro/uba` (see `r_pkgs.txt`); conda side in `conda_pkgs.txt`.

Two known build traps, both already solved and worth not rediscovering:

- **`uba` will not compile under GCC ≥ 14** (`cannot use keyword 'false' as
  enumeration constant` — GCC now defaults to C23 where `true`/`false`/`bool`
  are reserved, breaking `uba`'s legacy `typedef enum {false,true} bool;`).
  Fix with a user-level `~/.R/Makevars`:
  ```
  CFLAGS = -std=gnu17 -Wno-incompatible-pointer-types -Wno-implicit-function-declaration
  CXXFLAGS = -std=gnu++17
  ```
- **`scratch/uba/` ships stale precompiled `.o`/`.so` files.** Delete
  `scratch/uba/src/*.o` and `*.so` before `R CMD INSTALL scratch/uba`, or you get
  `"r2phi" not resolved from current namespace` at runtime.

On the CIn/UFPE Apuana cluster the environment already exists as the micromamba
env `exp002_env` — reuse it, do not rebuild. Apuana requires the CIn VPN plus an
interactive SSH password, so **no script can reach it**; every cluster command in
§2 is one a human pastes into their own terminal.

---

## 2. Stage 0 — produce the raw results (outside `Results (Clean)/`)

Skip this entire section if you already have populated
`Results Data/raw_iterations_by_dataset_v2/` and `raw_predictions_by_dataset/`.

### 2.1 The experiment

| Item | Value | Where it is set |
|---|---|---|
| Datasets | 24 (DS01–DS24) | `data/paper_datasets_csv/manifest.csv` |
| Model families | 5 — LM, SVM, MARS, RF, RPART | `src/adapted/Exps.R` |
| Strategies | 10 — `baseline` + UNDER/OVER/SMOTE × B/T/TPhi | `src/adapted/Exps.R` |
| Extra workflows | `mc.arima`, `mc.BDES` (52 workflows total) | `src/adapted/Exps.R` |
| Resampling | Monte Carlo, `nReps=50` | `Exps.R`, `MC_SZ_TRAIN`/`MC_SZ_TEST` |
| Split | 50% train / 25% test, **except** DS21–22 at 10%/5% and DS23–24 at 20%/10% | `Exps.R` (paper §5, p.169) |
| Embedding | phase-space, k = 10 | `Exps.R` |
| Relevance φ | `uba::phi.control(method="extremes")`, `event.thr = 0.9` | `Exps.R::eval.stats` |
| Hyperparameters | per-dataset, `OPT_PARAMS` (24 rows, paper Annex 1 Table 7) | `Exps.R` |

**Do not restore the `#EXAMPLE PARAMETRIZATION` block.** One fixed
`cost=150, gamma=0.001` reused across all datasets made baseline SVM predict zero
rare cases on 12/20 datasets (F1 exactly 0), which manufactured a spurious
resampling sweep. `OPT_PARAMS` replaced it and the collapse is gone (0/21).

### 2.2 Submit the job

```bash
# VPN first, then:
ssh <login>@slurm-client1.cin.ufpe.br
tmux new -s exp002                      # the VPN drops every ~30 min
cd ~/002_ts_resampling_strategies
mkdir -p logs
sbatch run_apuana.slurm                 # short-simple, 2 days, 48 cpus, 128G

squeue -u $USER --format="%.10i %.12j %.8T %.10M %.6D %R"
ls src/adapted/results/data_v3/*.Rdata | wc -l    # want 24
```

DS21–DS24 are far slower than DS01–DS20 (larger series, CSS `auto.arima` order
search). A single dataset failing no longer kills the run — the
`performanceEstimation()` call is wrapped in `try()`, writes no `.Rdata`, and a
resubmit retries only the missing indices.

### 2.3 Pull results back and unserialize

```bash
scp -r <login>@slurm-client1.cin.ufpe.br:~/.../results/data_v3/ src/adapted/results/

Rscript src/stage0/rdata_to_csv.R              # -> Results Data/raw_iterations_by_dataset_v2/
Rscript src/stage0/rdata_to_csv_predictions.R  # -> Results Data/raw_predictions_by_dataset/
```

Both scan `data_v3/` (or whatever `EXP002_RUN_ID` points at) for whatever `results_dataset_N.Rdata` files exist, so they
work on a partial run. **R is mandatory here and only here** — the results are
`performanceEstimation::ComparisonResults` S4 objects and no Python reader
handles S4. Everything downstream is Python (project convention).

**Expected shapes.** `raw_iterations_by_dataset_v2/DSxx_<slug>.csv`: 2,600 rows
(52 workflows × 50 iterations), columns `dataset_id, dataset_name, workflow,
iteration, prec, rec, F1`. `raw_predictions_by_dataset/DSxx/<workflow>_predictions.csv`
plus `_phi_ctrl.csv`: 104 files per dataset. If a count is off, stop — do not
proceed to Stage 1 with a truncated extraction.

### 2.4 Tier 2 variant runs — traceability knobs

Three env vars redirect `Exps.R` to a non-canonical run, all sourced from
`src/adapted/run_config.R`. **Defaults were promoted 2026-09-21** — a plain
`Rscript Exps.R` now reproduces the corrected, paper-replicating experiment
(`data_v3`, RNG pinned to `Rounding`) rather than the pre-promotion baseline;
see `DIVERGENCE_ANALYSIS.md` §11 for why. `data_v2` is still fully reachable,
just no longer the default:

| Env var | Effect | Default |
|---|---|---|
| `EXP002_RUN_ID` | `RESULTS_DIR` → `src/adapted/results/<id>` | `data_v3` (was `data_v2`) |
| `EXP002_SAMPLE_KIND` | `RNGkind(sample.kind=...)`, applied once before any fork (multicore forking inherits it automatically) | `Rounding` (was unset — R's post-2019 default) |
| `EXP002_BASELINE_PARAMS` | CSV of per-dataset `svm_cost,svm_gamma,rpart_minsplit,rpart_cp` read **only** by the plain (unresampled) `mc.svm`/`mc.rpart` workflows; every resampled variant keeps Table 7 | unset — Table 7 for baseline too |

To reproduce the pre-promotion `data_v2` run exactly: `EXP002_RUN_ID=data_v2
EXP002_SAMPLE_KIND=Rejection Rscript src/adapted/Exps.R`.

Submit with `sbatch --export=ALL,EXP002_RUN_ID=<id>[,EXP002_SAMPLE_KIND=...][,EXP002_BASELINE_PARAMS=...] run_apuana.slurm`.
Every run writes `src/adapted/results/<id>/RUN_MANIFEST.txt` (run id, timestamp,
R version, actual `RNGkind()`, the two override values, SLURM job id/node/CPU
count, `Exps.R`'s md5, and one status line per dataset) — read this before
trusting any number from a variant run against a "which code produced this"
question.

To evaluate a variant without touching `Results (Clean)/`: `scratch/setup_run_root.sh <id>`
scaffolds `runs/<id>/` as a standalone `EXP002_ROOT`, then
`Rscript src/stage0/rdata_to_csv.R "src/adapted/results/<id>" "runs/<id>/Results (Clean)/Results Data/raw_iterations_by_dataset_v2" data/paper_datasets_csv/manifest.csv`
followed by `EXP002_ROOT="$PWD/runs/<id>" python3 "Eval_Metrics Code/Paper Comparison/generate_paper_comparison.py"`
answers the Tier 2 decision-gate question with only the one generator that
needs it, not the full 22-generator pipeline.

---

## 3. Stage 1–5 — the evaluation pipeline

### Run it

```bash
cd "Results (Clean)/Eval_Metrics Code"
python3 run_all.py --tests     # ALWAYS first. 56 tests, ~3s.
python3 run_all.py             # 19 generators, ~2.5 min
```

Other entry points: `--list` prints the DAG without running; `--stage N` runs one
stage. Each generator runs as a subprocess with its own directory as cwd.

### The DAG — why this order

| Stage | Generators | Reads | Writes |
|---|---|---|---|
| **1** | `RMSE Metric/generate_rmse_tables.py`, `SERA Metric/generate_sera_tables.py` | `raw_predictions_by_dataset/` | `Latex Tables/{RMSE,SERA}/`, **`raw_rmse_by_dataset/`, `raw_sera_by_dataset/`** |
| **2** | `F1 Metric/generate_f1_tables.py` | `raw_iterations_by_dataset_v2/` | `Latex Tables/F1/` |
| **3** | `CD Metrics`, `Bayes Signed-Rank`, `Summary Tables`, `Rank Distribution`, 4× `Heatmaps`, 2× `Overview Figures` | all three raw metric folders | `Figures/{CD,Bayes,Rank Distribution,Heatmaps,Overview}/`, `Latex Tables/Summary/` |
| **4** | `Regression/generate_regression.py`, then `Causal/` × 4 | `raw_*_by_dataset/` | `Evaluation Metrics/{Regression,Causal}/`, `Latex Tables/Causal/`, `Figures/{Causal,Metric Diagnostics}/` |
| **5** | `Paper Comparison/generate_paper_comparison.py` | `raw_iterations_by_dataset_v2/` | `Latex Tables/Paper Comparison/` |

**Stage 1 must run first**: it is the only producer of per-iteration RMSE and
SERA. Every Stage-3/4 ranking, chart and regression consumes those files. Running
Stage 3 against a stale `raw_sera_by_dataset/` is the failure mode this ordering
exists to prevent.

Stage 4's internal order matters (`Causal` imports from `Regression`). Stage 5 is
independent of 2–4 and runs last so its console summary is the last thing printed.

---

## 4. What each stage produces, and how to read it

### Stage 1–2 — per-metric appendix tables

`Latex Tables/{F1,RMSE,SERA}/<metric>_table_{lm,svm,mars,rf,rpart}.tex` — 5 files
per metric, one per model family (never one table averaged across families).
Rows = datasets, columns = the 10 strategies, cells = `mean ± sd` to 4 decimals
over the 50 Monte Carlo iterations. **Bold marks the row maximum for F1 and the
row minimum for RMSE/SERA** — F1 is higher-is-better, the other two are
lower-is-better. Layout follows Avelino et al. (2024) appendix Tables B1–B24;
LaTeX aesthetic comes from `tables_common.table_star()` (booktabs,
`makebox`+`resizebox` centering, bold-lead-in caption).

### Stage 3 — rankings, significance and overviews

- **`Figures/CD/`** — Demšar critical-difference diagrams, one per (metric,
  family). Ranks all 10 strategies jointly across datasets. **The Friedman
  omnibus test gates the Nemenyi post-hoc**: a panel that fails it is still drawn
  but titled `NOT SIGNIFICANT` — do not cite its bars. The critical distance
  scales as 1/√N, so N is derived per matrix, never a constant.
- **`Figures/Bayes/`** — Bayesian signed-rank (Benavoli et al. 2017), each
  strategy vs. baseline only, ungated. Stacked bar = P(strategy wins) / P(ROPE
  draw) / P(baseline wins), 50,000 Dirichlet draws. ROPE is `±0.01` for F1 (raw
  difference) and `±log(1.05)` for RMSE/SERA (log ratio, since both are unbounded
  and differ wildly in scale across datasets).
- **`Figures/Rank Distribution/`** — per-dataset rank boxplots. Adds what CD
  cannot show: whether a strategy is consistently mid-pack or swings between best
  and worst. **The y-axis is deliberately inverted** so rank 1 is at the top.
- **`Latex Tables/Summary/`** — 9 files: win counts by family, best/worst combo
  per dataset, global average rank over all 50 family×strategy combos, plus three
  combined F1+SERA versions. Ties are decided on the value **rounded to 4
  decimals** (matching displayed precision), not raw float equality.
- **`Figures/Heatmaps/`, `Figures/Overview/`** — value/delta/Bayes/win-count
  heatmaps and per-metric overview panels.

### Stage 4 — effect size

- **`Evaluation Metrics/Regression/`** — two-way fixed-effects OLS:
  `value ~ C(strategy, Treatment('baseline')) + C(dataset_id) + C(model_family)`,
  one row per (dataset, family, strategy) mean (never per raw iteration — that
  would be pseudo-replication). SERA enters as `log(SERA)`. SEs clustered by
  dataset. **Dataset-dummy rows show `std err ≈ 1e-16` — this is a genuine
  cluster-robust degeneracy, not a real result. Never read their p-values.** The
  strategy and family coefficients are trustworthy.
- **`Evaluation Metrics/Causal/`, `Latex Tables/Causal/`, `Figures/Causal/`** —
  the design is a fully-crossed randomized experiment (the experimenter assigned
  strategy), so dataset and family are **blocking factors for precision, not
  confounding controls**, and the strategy effects are causal by design.
  Estimands: `tau_k` (mean paired difference vs. baseline), exact randomization
  inference by permuting strategy labels **within** each block, percentile
  bootstrap CIs, Holm correction for the pooled 9-strategy family, plus win
  probability and rank shift as distribution-free secondaries.
- **`leave_one_source_out`** — the 24 datasets are **7 independent sources**, not
  24 independent draws: bike_daily (DS01–04), bike_hourly (DS05–08),
  icelandic_river (DS09), porto_weather (DS10–13), istanbul_stock (DS14–20),
  australian_electricity (DS21–22), porto_water (DS23–24). Any sensitivity claim
  must be made at the *source* level. `SOURCE_MAP` raises `KeyError` on an
  unmapped dataset rather than silently keeping it in every fold.

### Stage 5 — fidelity against the source paper

`Latex Tables/Paper Comparison/` — 14 files comparing this replication against
Moniz, Branco & Torgo (2017): Table 1 (%Rare, a standing regression guard on the
φ implementation), Tables 3/4/5 (paired Wilcoxon Win/sigWin/Loss/sigLoss at
α=0.05), Table 6 (absolute F1), a precision-vs-recall breakdown, and
`cmp_sensitivity.{txt,tex}`.

**`cmp_sensitivity` is labelled DIAGNOSTIC ONLY and must stay that way.** It is a
leave-one-dataset-out sweep, *not* a jackknife: it implies no standard error,
because the 24 datasets are 7 correlated sources and leave-one-out values are not
independent draws. The statistically clean cut is leave-one-**source**-out, which
is what `Causal/causal_effects.py:leave_one_source_out()` does.

Counts are reported as **proportions as well as raw counts** — this replication
has 24 datasets and the paper has 24, but a partial run has fewer, and raw counts
are not comparable across different denominators.

---

## 5. Standing caveats that must travel with every number

State these in any writeup. They are properties of the pipeline, not bugs to fix.

1. **SERA on time series is novel.** SERA (Ribeiro & Moniz 2020) postdates the
   paper being replicated and, per a literature search of 64 papers, has never
   been applied to a time-series forecasting benchmark. Present it as a novel
   application, not established practice.
2. **The accuracy-vs-rare-case trade-off is real, not a bug.** Baseline sweeps
   RMSE (5/5 families) and now also wins the SERA majority, while resampling wins
   F1 decisively. RMSE and SERA are error-weighted accuracy metrics; F1_φ is a
   detection metric. A rank-flip bug could not reverse itself between metrics
   while staying uniform across five independently fitted families.
3. **No compiled report lives in this repo, by design** — see `README.md`'s
   "What's excluded, and why". The narrative previously built on top of these tables
   (and an earlier, unrelated `Papers/unified_paper/` bundle generated before the
   2026-09-03 SERA φ fix and never regenerated) has been removed; `DIVERGENCE_ANALYSIS.md`
   is this repo's canonical, kept-up-to-date account of what was found and fixed.

---

## 6. Protocol for evaluating a NEW set of results

Follow in order. Do not skip step 1 or step 2.

```bash
cd "Results (Clean)/Eval_Metrics Code"

# 1. Guards first -- catches a dataset added to one catalogue but not the other.
python3 run_all.py --tests

# 2. Snapshot the current outputs so you can say exactly what moved.
cd "../Evaluation Metrics"
find . -type f \( -name '*.tex' -o -name '*.txt' -o -name '*.csv' \) \
  -exec shasum {} \; | sort > /tmp/before.sha

# 3. Stage 0: refresh the raw CSVs (§2.3), if the .Rdata changed.

# 4. Regenerate EVERYTHING, in order.
cd "../Eval_Metrics Code" && python3 run_all.py

# 5. Diff.
cd "../Evaluation Metrics"
find . -type f \( -name '*.tex' -o -name '*.txt' -o -name '*.csv' \) \
  -exec shasum {} \; | sort > /tmp/after.sha
diff /tmp/before.sha /tmp/after.sha
```

**Then interpret the diff before believing anything.** Ask: does the set of files
that moved match the change you made? A φ-relevance change should move only
SERA-derived files. A new dataset should move everything. An F1-only change that
somehow moved an RMSE table is a bug, not a result.

Finally, append a dated entry to `CLAUDE.md` recording what changed, what moved,
and what did not — the project's own convention and the only durable record of
why a number differs between two runs.

---

## 7. Protocol for ADDING a dataset

The whole "hardcoded 20" bug class comes from this path, so it is spelled out.

1. **`data/paper_datasets_csv/`** — add the CSV, add its row to `manifest.csv`
   with the next `DSxx` id.
2. **`Eval_Metrics Code/tables_common.py::DS_LABELS`** — add `"DS25": "DS25 (Short
   Description)"`. Omitting this drops the dataset from every appendix table
   silently. `test_tables_common.py` fails if `DS_LABELS` and the manifest
   disagree in either direction.
3. **`Eval_Metrics Code/Causal/causal_effects.py::SOURCE_MAP`** — assign it to a
   source. A new independent source gets a new source name; a new variable from
   an existing source reuses that source's name. `leave_one_source_out()` raises
   `KeyError` if you forget.
4. **`src/adapted/Exps.R`** — `TOTAL_DS`, `OPT_PARAMS` (one row of 9
   hyperparameters), and `MC_SZ_TRAIN`/`MC_SZ_TEST` if the series is large enough
   to need a smaller split.
   `test_paired_comparisons.py::test_r_opt_params_and_mc_split_match_paper_reference`
   parses those literals straight out of `Exps.R` and asserts they match the
   Python transcription in `paper_reference.py` — this is the drift guard.
5. Re-run Stage 0, then §6.

**Nothing else needs editing.** Every generator derives its dataset count from
disk. If you find yourself editing a number in a generator to accommodate a new
dataset, that is a bug in the generator — fix it to derive instead.

---

## 8. Folder map

```
002_ts_resampling_strategies_isolated/
├── data/paper_datasets_csv/          # 24 source datasets + manifest.csv
├── src/adapted/Exps.R                # STAGE 0: the R experiment
├── run_apuana.slurm                  # STAGE 0: cluster submission
├── src/stage0/rdata_to_csv.R            # STAGE 0: .Rdata -> F1 CSVs
├── src/stage0/rdata_to_csv_predictions.R# STAGE 0: .Rdata -> predictions CSVs
├── requirements.txt / r_pkgs.txt / conda_pkgs.txt
├── CLAUDE.md                         # dated change log -- append to it
└── Results (Clean)/
    ├── EXPERIMENTAL_PROTOCOL.md      # this file
    ├── Results Data/                 # tidy inputs (and Stage-1 outputs)
    │   ├── raw_iterations_by_dataset_v2/   # prec/rec/F1, 2600 rows/dataset  [Stage 0 out]
    │   ├── raw_predictions_by_dataset/     # trues/preds/phi_ctrl per workflow [Stage 0 out]
    │   ├── raw_rmse_by_dataset/            # [Stage 1 out]
    │   ├── raw_sera_by_dataset/            # [Stage 1 out]
    │   ├── raw_iterations_by_dataset/      # LEGACY pre-cap-fix run, do not use
    │   └── original_paper_reference_results/
    ├── Eval_Metrics Code/            # ALL generator code
    │   ├── run_all.py                # the orchestrator -- start here
    │   ├── paths.py                  # every path, derived from EXP002_ROOT
    │   ├── tables_common.py          # DS_LABELS, families, strategies, table_star()
    │   ├── plot_style.py
    │   └── <Metric or Analysis>/     # one folder per generator + its tests
    └── Evaluation Metrics/           # ALL outputs
        ├── Latex Tables/{F1,RMSE,SERA,Summary,Causal,Paper Comparison}/
        ├── Figures/{CD,Bayes,Rank Distribution,Heatmaps,Overview,Causal,Metric Diagnostics}/
        ├── Regression/
        └── Causal/
```

Two conventions hold everywhere and should hold for anything added:
**code in `Eval_Metrics Code/<Name>/`, output in `Evaluation Metrics/`**, and
**one file per table and per figure** — never several tables concatenated into one
`.tex`.

---

## 9. Verification checklist

A run is complete when all of these hold:

- [ ] `python3 run_all.py --tests` — 56 passed.
- [ ] `python3 run_all.py` — all 19 generators `ok`, none `FAILED`.
- [ ] `grep -rn "/Users/" "Eval_Metrics Code" --include='*.py'` returns only
      `paths.py`'s docstring.
- [ ] File counts: 5 `.tex` each for F1/RMSE/SERA, 9 Summary, 5 Causal, 14 Paper
      Comparison; 30 CD figures, 30 Bayes, 8 Heatmaps, 16 Overview, 6 Rank
      Distribution, 3 Causal, 9 Metric Diagnostics.
- [ ] The shasum diff (§6) moved exactly the files the change should have moved.
- [ ] Every Stage-1 raw folder has one file per dataset present in Stage 0.
- [ ] A dated entry appended to `CLAUDE.md`.

---

## 10. Known open items

Superseded 2026-09-21 (kept for history): the items below described a
21-dataset partial run and pre-promotion SERA prose. Both are resolved —
`data_v3` covers all 24 datasets for F1/RMSE/SERA alike, and `main.tex` was
re-synced in the same pass. The SVM/RPART fidelity gap (`DIVERGENCE_ANALYSIS.md`
§§1–10) is also resolved as of the same date — see §11 there: the residual
was two compounding defects, an RNG-algorithm change (Cause A, unavoidable,
now pinned) and a mean-vs-median Win/Loss sign bug in this replication's own
code (Cause C, fixed). All 15 of the paper's Table 3 cells now match within
±1 dataset. No open items remain from that investigation.

Superseded entries, for the record:
- ~~`raw_predictions_by_dataset/` currently holds DS01–DS20 only~~ — resolved,
  all 24 datasets covered since the `data_v3` promotion.
- ~~Apuana job 14106 partial run~~ — resolved, `data_v3` is a complete 24-dataset
  run (job 15570, `rng_rounding`, promoted).
- ~~`main.tex` SERA prose pre-dates the φ fix~~ — resolved, re-synced 2026-09-21
  alongside the fidelity-section rewrite.
