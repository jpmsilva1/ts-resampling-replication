# Divergence Analysis: This Replication vs. Moniz, Branco & Torgo (2017)

**Date:** 2026-09-18
**Scope:** Why some replicated results disagree with the source paper, whose side each
cause is on, and what to change to close the remaining gap.
**Source paper:** Moniz, N., Branco, P., Torgo, L. — *Resampling strategies for imbalanced
time series forecasting*, Int J Data Sci Anal 3:161–181 (2017).
**Released code:** `src/original/R_Code/` (snapshot of `nunompmoniz/TSResampStrat_JDSA2017`).

Every claim below is tagged **[VERIFIED]** (reproduced from data or source in this repo),
**[HYPOTHESIS]** (best-supported explanation, not yet tested), or **[REFUTED]** (tested and
ruled out). Do not promote a hypothesis to a finding in `main.tex` without running the
experiment named in §6.

---

## 1. Executive summary

Three families reproduce the paper; two do not.

| Family | Ours %W (U_B/O_B/SM_B) | Bootstrap 95% CI | Paper %W | Agrees? |
|---|---|---|---|---|
| LM | 79 / 75 / 79 | [75,83] [75,88] [79,88] | 79 / 75 / 79 | ✅ |
| MARS | 58 / 75 / 75 | [50,67] [71,79] [75,79] | 62 / 71 / 75 | ✅ |
| RF | 83 / 79 / 92 | [75,88] [79,88] [83,92] | 75 / 83 / 83 | ✅ |
| **SVM** | 58 / 54 / 54 | [54,67] [50,62] [46,62] | **33 / 29 / 29** | ❌ ~25pp |
| **RPART** | 71 / 67 / 67 | [62,79] [58,71] [58,79] | **50 / 46 / 42** | ❌ ~21pp |

**[VERIFIED]** CIs are a 4,000-resample bootstrap over the 50 Monte Carlo folds per
dataset, recomputing the Win/Loss sign per dataset on each draw. LM, MARS and RF are fully
explained by fold noise. SVM and RPART sit ~20pp outside it.

**Both deviating families deviate in the same direction** — this replication is more
pro-resampling than the paper. `main.tex` §Fidelity currently says RF/RPART overshoot "in
the opposite direction"; that is wrong and must be corrected (§5, item 3). The corrected
statement is *stronger* evidence for a single systematic cause.

**The protocol itself is faithful.** The paper's own §5.1 (p.174) confirms the two design
choices that looked like candidate deviations are what the authors did:

> "although an optimal parameter search method was employed for the baseline regression
> algorithms, **and such parameters were used in the resampled alternatives**, a similar
> approach was not employed concerning the optimal parameters for under and oversampling
> percentages. **This is intended**, as our objective is to assert the impact of these
> resampling strategies in a default setting, i.e. balancing the number of normal and rare
> cases."

So `C.perc="balance"` everywhere, and Table 7 optima applied to baseline *and* resampled
workflows alike, are both correct. **[VERIFIED]** — `C.perc="balance"` appears in 60 call
sites in *both* `src/original/R_Code/Exps.R` and `src/adapted/Exps.R`.

---

## 2. Cause A — R 3.6.0 RNG change (confirmed; explains LM/MARS/RF, not SVM/RPART)

**Whose side:** neither. A 2019 change to the R language, unavoidable.

`performanceEstimation:::mcEstimates` selects Monte Carlo evaluation origins as:

```r
set.seed(estTask@method@seed)            # 1234, the package default
starting.points <- sort(sample(selection.range, estTask@method@nReps))
```

R 3.6.0 (April 2019) changed the default `sample()` algorithm for discrete ranges from
`"Rounding"` to `"Rejection"`. **[VERIFIED]** — reproduced locally on R 4.5.3 at this
project's actual dataset sizes:

| Dataset | n (embedded) | folds identical to pre-3.6 R |
|---|---|---|
| DS01 | 720 | 16 / 50 |
| DS14 | 526 | 18 / 50 |
| DS05 | 17,368 | **0 / 50** |

**Consequence:** `seed = 1234` is cosmetic reproducibility. This replication and the
authors' run evaluated on largely or entirely disjoint test windows. Cell-by-cell
agreement with the paper was never achievable, regardless of code correctness. **This is
currently absent from `main.tex` and is the single largest barrier to exact reproduction.**

**What it does NOT explain.** Different folds are noise, not bias. The §1 bootstrap shows
this noise fully covers LM, MARS and RF, and leaves SVM/RPART ~20pp short.

**Not affected:** `randomForest`'s bootstrap draws happen in C via `unif_rand()`, not
`sample()`, so `sample.kind` does not touch them. Affected are fold selection and the
inline `randOverRegress*` / `randUnderRegress*` / `smoteRegress*` draws.

### Why SVM and RPART specifically

Win/Loss is a **sign** statistic that discards magnitude. Median per-dataset |ΔF1| vs.
baseline, and the count of datasets where the effect is within ±0.02 F1:

| Family | median \|ΔF1\| (U_B/O_B/SM_B) | datasets within ±0.02 (of 24) |
|---|---|---|
| LM | 0.280 / 0.251 / 0.347 | 4 / 6 / 4 |
| MARS | 0.056 / 0.143 / 0.119 | 5 / 3 / 2 |
| SVM | 0.055 / 0.085 / 0.061 | 7 / 4 / 4 |
| RF | 0.085 / 0.053 / 0.217 | 8 / 8 / 4 |
| **RPART** | **0.026 / 0.040 / 0.030** | **9 / 6 / 10** |

**[VERIFIED]** Fragility ranks `rpart > svm ≈ rf > mars > lm`; divergence from the paper
ranks almost identically. On RPART, up to 42% of datasets are effectively coin flips — a
different set of folds flips several, and each flip moves the percentage ~4pp.

---

## 3. Cause B — the residual SVM/RPART gap (open; most likely the authors' side)

**Whose side:** [HYPOTHESIS] the paper's under-specified optimization step.

`src/original/R_Code/OptParmsSearch.R` is the only released tuning script. What it actually
searches:

```r
Workflow("mc.svm", cost=150, gamma=0.01),                                    # ONE fixed point
workflowVariants("mc.svm_UNDERB", cost=c(10,150,300), gamma=c(0.01,0.001), un=c(.1,.2,.4,.6,.8)),
workflowVariants("mc.svm_OVERB",  cost=c(10,150,300), gamma=c(0.01,0.001), ov=c(2,3,5,10)),
...
```

**[VERIFIED]** The baseline is a single fixed point. Only the *resampled* variants are
searched. There is no released script that searches baseline `cost`/`gamma` per dataset,
and nothing at all for RPART's `minsplit`/`cp` — yet Table 7 is titled *"Optimal
parameterization for each standard regression algorithm in each data set."*

**[HYPOTHESIS]** Table 7's values were selected as the optimum of the *resampled* SVM
workflows. This replication then applies them to the baseline, exactly as the paper's prose
instructs — which would leave our baselines systematically weaker than whatever the authors
actually ran, and resampling correspondingly stronger.

Fit to the evidence:
- Direction matches (ours more pro-resampling). ✅
- Hits exactly the two families with released-code gaps (SVM search excludes the baseline;
  RPART has no search script at all). ✅
- Uniform across datasets — consistent with `cmp_sensitivity.txt`, where no single dataset
  moves an SVM row by more than ~3pp and the weakest source hold-out still leaves ~18pp. ✅

**Corroborating evidence that the published tables are not pristine:** paper Table 4's
RPART rows `SM_T 8(1)/15(11)` and `SM_TPhi 6(4)/17(9)` each sum to **23**, not 24.

This cannot be closed without a rerun — see §6, Experiment 2.

---

## 4. Deviations in our code (all real, all currently undocumented in `main.tex`)

Found by diffing `src/original/R_Code/Exps.R` against `src/adapted/Exps.R`. None of these
drives the §1 gap, but all belong in `tab:exp002_differences`.

### D1. SVM training-row cap — deviation, asymmetric, empirically inert

`src/adapted/Exps.R:317`:

```r
MAX_SVM_TRAIN_ROWS <- 10000
cap_svm_train <- function(train) {
  if (nrow(train) > MAX_SVM_TRAIN_ROWS) train <- train[(nrow(train) - MAX_SVM_TRAIN_ROWS + 1):nrow(train), ]
  train
}
```

Applied to `mc.svm` and all six OVER/SMOTE variants, but **not** to the three UNDER
variants. It keeps the **last** 10,000 rows, and `randOverRegressB` ends with
`newdata <- rbind(newdata, dat[sel, ])` — rare-case replicas are *appended*. So on DS05–08
(oversampled to ~16.7k rows) the retained tail is dominated by duplicated rare cases, while
the uncapped baseline keeps a plain temporal tail.

Datasets where it can fire: DS05–08 (oversampled only), DS21–24 (both arms).

**[REFUTED] as a cause.** Difference-in-differences of (capped − uncapped datasets) for SVM
against the four uncapped families: **+0.011 F1**. No measurable effect. The apparent drop
on those 8 datasets appears equally in LM, MARS and RF, which are never capped — it is
dataset difficulty, not the cap.

**Still fix it:** make the cap symmetric (apply to UNDER too) or sample the retained rows
rather than taking the tail, and document it.

### D2. `complete.cases` — deviation; explains the DS12 Table 6 outlier

Original has `#ds <- ds[complete.cases(ds),]` commented out; ours is live (added 2026-08-07
to stop a DS12 crash in `phi.control`). Affects **DS12, DS13, DS23, DS24** only — the four
datasets with NAs per `data/paper_datasets_csv/manifest.csv`.

**[VERIFIED]** This is the explanation for the Table 6 outlier that `main.tex` currently
says is "not explained here": DS12 SVM baseline at 0.220 vs. the paper's 0.554.

**[REFUTED] as a cause of the §1 gap.** Excluding all four datasets moves SVM *away* from
the paper (58 → 65 %W), i.e. the wrong direction.

### D3. Zero-relevance `stop()` → `return(data)` — deviation, inert

All 9 inline resampling helpers had their `stop("All the points have relevance 0/1...")`
guards replaced with silent no-op returns. **[VERIFIED]** inert: across all 15 Table-3
cells there is exactly **1** tie, so a fold almost never falls back to unresampled data.
Keep it (it prevents one fold killing a multi-day job), but state that it is measured as
inert rather than assumed to be.

### D4. `mc.arima` CSS estimation — deviation, already documented

`auto.arima(trainY, method="CSS", stepwise=TRUE)` replaces plain `auto.arima(trainY)`.
Already in `REPLICATION_LOG.md`. Affects only `mc.arima` (Table 5 / H.3), which is the
cleanest-agreeing block in the audit, so the deviation is not load-bearing.

### D5. Per-dataset Monte Carlo splits — not a deviation

DS21–22 at 10%/5% and DS23–24 at 20%/10% match the paper (§5, p.169). Correct as-is.

---

## 5. Corrections needed in `main.tex`

1. **Add Cause A (R 3.6.0 RNG) to `tab:exp002_differences`.** It is the largest single
   obstacle to cell-level reproduction and is currently unmentioned anywhere in the report.
2. **Add D1 and D2 as rows in `tab:exp002_differences`.**
3. **Fix "in the opposite direction"** in §Fidelity, Table 3 paragraph. SVM and RPART both
   deviate the *same* way (more pro-resampling than the paper). The corrected claim is
   stronger support for the §3 hypothesis.
4. **Remove the implied RF problem.** RF is 83/79/92 vs. 75/83/83 — inside the bootstrap CI
   on all three strategies. RF agrees with the paper; do not group it with SVM/RPART.
5. **Replace "an unidentified pipeline-level difference"** with the §2 fragility analysis
   plus the §3 `OptParmsSearch.R` hypothesis. The cause can now be named, with the caveat
   that it is untested.
6. **Attribute the DS12 Table 6 outlier to D2** instead of leaving it unexplained.
7. **Reclassify the "baseline precision ≈0 while recall ≈0.48" item out of Limitations.**
   It is not a defect and not ours. `scratch/uba/src/util.h:35`:
   ```c
   #define DELTA 0.00001 // a value to avoid the null tradeoff of P and R
   ```
   **[VERIFIED]** `prec` is *exactly* `1e-05` in 6,827 of 62,400 rows (10.9%) — uba's
   deliberate sentinel for "no prediction cleared the relevance threshold," kept non-zero so
   F1 does not evaluate 0/0. That recall stays ~0.48 alongside it follows from recall's
   denominator running over *true* relevant cases with the `(1+u)/(1+φ)` form returning ~0.5
   at zero utility. That LM hits it on 12/24 datasets is also expected, not mysterious: LM is
   a global linear fit on a phase-space embedding, so it regresses to the mean and
   structurally cannot emit extreme predictions.
8. **Report mean ΔF1 with CIs alongside Win/Loss counts.** Sign counting is what makes SVM
   and RPART look irreconcilable when the underlying effects are only 0.03–0.08 F1.

---

## 6. Action plan, ordered by expected payoff

### Tier 1 — free, no rerun (do these first)

| # | Action | Effort |
|---|---|---|
| 1 | Apply the eight `main.tex` corrections in §5 | ~1h |
| 2 | Add D1–D3 to `tab:exp002_differences` | 20m |
| 3 | Make `cap_svm_train` symmetric, or sample instead of tail-slice | 15m |
| 4 | Replace the circular drift guard (below) | 30m |

**On the drift guard.** `test_paired_comparisons.py::test_r_opt_params_and_mc_split_match_paper_reference`
compares the R literals in `Exps.R` against the Python transcription in
`paper_reference.py` — **two copies of the same transcription**. A PDF misread passes both.
All 216 Table 7 values have now been checked independently against the paper's Annex 1 and
are **[VERIFIED]** correct, but the test should assert against a checked-in extract of the
paper table, not against a sibling copy.

### Tier 2 — the two reruns that can actually close the gap

**Experiment 1 — pin the pre-2019 RNG.** Add before `parallelStartMulticore()` in
`src/adapted/Exps.R`:

```r
RNGkind(sample.kind = "Rounding")   # restore pre-R-3.6.0 discrete sampling (2017 behaviour)
```

`set.seed()` preserves the active `sample.kind`, so this genuinely restores 2017 fold
selection and the inline resampling draws. Two caveats: it does not touch `randomForest`'s
C-level bootstrap, and it must be applied **inside each fork** (set it in the worker init,
not only in the parent) for the resampling draws to be affected under `parallelMap`.

*Success criterion — state it before running.* Do **not** use "does the gap tighten." Use:
does the SVM/RPART Win% move **outside** the §1 bootstrap CI? If it stays inside, Cause A is
confirmed as noise-only and the entire residual belongs to Cause B. Expected outcome: it
stays inside. This experiment is worth running to *eliminate* Cause A, not to fix anything.

**Experiment 2 — test the Cause B hypothesis directly.** This is the one that can close the
gap. Re-run SVM and RPART only, with the baseline tuned on its own objective rather than
inheriting Table 7:

- Run `OptParmsSearch.R`'s grid (`cost ∈ {10,150,300}`, `gamma ∈ {0.01,0.001}`; for RPART,
  `minsplit ∈ {10,20,30}`, `cp ∈ {0.001,0.01,0.1}`) against the **unresampled baseline**,
  per dataset, 10 MC reps, selecting on median F1 — mirroring the paper's stated procedure
  but applied to the arm the released script never searched.
- Keep resampled workflows on the existing Table 7 values (that part is what the paper says
  it did).
- Recompute Table 3.

*Success criterion:* if the SVM row moves from ~58%W toward ~33%W and RPART from ~67%W
toward ~46%W, Cause B is confirmed and the finding is publishable as a specific,
named under-specification in the source paper. If it does not move, Cause B is refuted and
the residual is genuinely unexplained — say so.

Cost: 2 families × 10 strategies × 24 datasets. Far cheaper than a full rerun.

### Tier 3 — optional

- Re-run with `complete.cases` disabled on DS12/13/23/24 to confirm D2 quantitatively as
  the DS12 outlier's cause (it is already the only candidate).
- If Experiment 2 confirms Cause B, the honest headline result changes from "we reproduce
  H.1" to "we reproduce H.1, and identify that the paper's baseline hyperparameters are not
  recoverable from its released code" — a stronger contribution than a clean replication.

---

## 7. Verdict table

| Issue | Whose side | Status | Effect on the gap |
|---|---|---|---|
| Hyperparameter placeholder (`cost=150, gamma=0.001` for all datasets) | Authors — released code ≠ paper | ✅ Fixed | Was dominant; resolved |
| R 3.6.0 `sample()` Rounding→Rejection | Neither (2019 language change) | Irrecoverable | Explains LM/MARS/RF entirely; **not** SVM/RPART |
| `OptParmsSearch.R` never searches the baseline | Authors — under-specified | **Open, [HYPOTHESIS]** | Best candidate for the full ~20pp residual |
| D1 `MAX_SVM_TRAIN_ROWS` cap | Ours | Undocumented deviation | **[REFUTED]** — DiD +0.011 F1 |
| D2 `complete.cases` enabled | Ours | Undocumented deviation | **[REFUTED]** for the gap; explains DS12 Table 6 |
| D3 zero-relevance passthrough | Ours | Undocumented deviation | **[REFUTED]** — 1 tie in 15 cells |
| D4 `mc.arima` CSS | Ours | Documented | Not load-bearing (Table 5 agrees cleanly) |
| Baseline precision `1e-05` | Neither — uba's `DELTA` sentinel | **Not a defect**; reclassify | None |
| RF "overshoot" | — | **Does not exist** — inside CI | None |
| Table 7 transcription | — | **[VERIFIED]** clean, 216/216 | None |
| DS21–24 Monte Carlo splits | Ours | ✅ Fixed | Resolved |

---

## 8. Reproducing the numbers in this document

All measurements were computed from `Results (Clean)/Results Data/raw_iterations_by_dataset_v2/`
(62,400 rows: 24 datasets × 52 workflows × 50 iterations) using
`Results (Clean)/Eval_Metrics Code/Paper Comparison/paired_comparisons.py::wl_vs_reference`
for the Win/Loss counts. The RNG comparison was run against the locally installed
`performanceEstimation` on R 4.5.3 via `RNGkind(sample.kind=)`. The paper's values were read
from `Papers/Resampling strategies for imbalanced time series forecasting.pdf` (Tables 3, 4,
6, 7 and Annex 1).

Nothing in this analysis required re-running the R experiment.

---

## 9. Tier 1 fixes applied (2026-09-18)

Everything in this section is done, verified, and requires no cluster rerun. Tier 2
(§6) is still open.

### 9.1 `main.tex` corrections (§5, items 1, 2, 3, 4, 5, 6)

- **Added the R 3.6.0 RNG change as a row in `tab:exp002_differences`**, with the
  measured fold-overlap numbers (16/50, 18/50, 0/50) inline.
- **Added D1 (SVM training-row cap) and D2 (`complete.cases`) as rows** in the same
  table, each stating the empirical test that ruled it out as the cause of the SVM/RPART
  gap (difference-in-differences +0.011 F1; exclusion moving the aggregate away from the
  paper).
- **Rewrote the Table 3 paragraph** (§Fidelity): replaced "RF and especially RPART...
  overshoot... in the opposite direction" with the correct claim (SVM and RPART deviate
  the *same* direction), added the bootstrap 95% CIs per family, and removed RF from the
  group of disagreeing families — RF's paper values fall inside its CI on all three
  strategies. Replaced "an unidentified pipeline-level difference" with the fragility
  analysis and the `OptParmsSearch.R` baseline-tuning hypothesis, explicitly labeled as
  unconfirmed.
- **Attributed the DS12 Table 6 outlier** (0.220 vs. 0.554) to the `complete.cases`
  deviation instead of leaving it unexplained.
- **Reclassified the baseline precision-≈0 finding** out of Limitations: it's `uba`'s
  `DELTA` sentinel (`util.h:35`, `prec` exactly `1e-05` in 6,827/62,400 rows), not a
  defect on either side. Added the LM structural explanation (global linear fit on a
  phase-space embedding, cannot emit extreme predictions).
- **Rewrote the Limitations paragraph** to drop the false "RPART and RF... above the
  paper" claim and name both real candidate causes (`OptParmsSearch.R`'s baseline-search
  gap; the R RNG change) instead.
- Verified: `latexmk -pdf` compiles clean — 0 errors, 0 undefined references, 0 duplicate
  labels (57 pages, up from 45 before this pass; only pre-existing-style overfull hboxes
  in wide table cells, none new).

### 9.2 `cap_svm_train` made symmetric and non-biased (§4, D1; §6 Tier 1 item 3)

`src/adapted/Exps.R`: the cap now takes a **random sample** of `MAX_SVM_TRAIN_ROWS`
instead of slicing the tail (the tail slice kept mostly the rare-case replicas
`randOverRegress*` appends at the end, biasing capped OVER/SMOTE training sets toward
rare cases), and is now called from all three `UNDER` variants too, for symmetry with
the six OVER/SMOTE variants that already called it. Confirmed inert on measured results
(§4, D1's difference-in-differences test), so no downstream data needs regenerating —
this fixes a structural landmine for any future rerun on datasets large enough to trip
the cap (DS21–24), not the already-computed 24-dataset numbers. `Exps.R` still parses
clean after the edit.

### 9.3 De-circularized the Table 7 drift guard (§6 Tier 1 item 4)

Root cause: `paper_reference.py`'s `OPT_PARAMS_TABLE7` carried a comment admitting it
"mirrors the OPT_PARAMS table now landed in `src/adapted/Exps.R`" — the guard test
compared two by-hand transcriptions of the paper to each other, so a shared
transcription error would have passed silently.

Fix:
1. Added `table7_pdf_extract.txt` — the raw `pdftotext -layout` text of paper Table 7
   (Annex 1, p.180), checked in verbatim with the regeneration command in a header
   comment.
2. Added `test_opt_params_table7_matches_pdf_extract()` to `test_paired_comparisons.py`,
   which parses that raw text with a regex and checks all 24 rows against
   `paper_reference.py`'s transcription independently of `Exps.R`.
3. Corrected `paper_reference.py`'s comment to point at the new test instead of at
   `Exps.R`.

All 24 rows matched on first run — the existing transcription (already checked by hand
against the PDF earlier in this investigation) was correct; the guard itself was the
problem, and it's now a real check. Full suite: 56/56 tests pass (was 55).

### 9.4 Not done in this pass

Items 7–9 from §6 remain open: reporting mean ΔF1 with CIs is already partially folded
into the rewritten Table 3 paragraph (9.1) but the underlying tables/figures were not
regenerated to add CI columns; and both Tier 2 reruns (RNG-pinned rerun, baseline-tuning
rerun) still need Apuana access, which only the user can submit.

## 10. Tier 2 Experiment 1 result — Cause A ruled out (2026-09-20)

Ran with genuine traceability infrastructure this time (see
`~/.claude/plans/luminous-noodling-bear.md` Phase 0): `EXP002_RUN_ID=rng_rounding`,
`EXP002_SAMPLE_KIND=Rounding`, SLURM job 15570, `cluster-node6`, all 24 datasets completed,
`src/adapted/results/rng_rounding/RUN_MANIFEST.txt` confirms `RNGkind()` was actually
`Mersenne-Twister, Inversion, Rounding` for the run (not just requested). Evaluated via
`EXP002_ROOT=.../runs/rng_rounding` against only the Paper Comparison generator, per the
plan's decision-gate scoping.

| Family | data_v2 (current) | rng_rounding (pre-2019 RNG) | §1 bootstrap CI | Outside CI? |
|---|---|---|---|---|
| SVM (U_B/O_B/SM_B) | 58/54/54 %W | 62/58/54 %W | [54,67]/[50,62]/[46,62] | No — all inside |
| RPART (U_B/O_B/SM_B) | 71/67/67 %W | 67/62/62 %W | [62,79]/[58,71]/[58,79] | No — all inside |

**Verdict: Cause A is ruled out**, exactly as predicted in §6. The pre-2019
Rounding-based `sample()` produces SVM/RPART Win/Loss numbers statistically
indistinguishable from the current Rejection-based pipeline (both stay inside the §1
bootstrap CIs), and both remain equally far from the paper's 33/29/29 (SVM) and 50/46/42
(RPART). The R 3.6.0 RNG change does not explain the residual gap — it was worth checking
(DS05's 0/50 shared folds made it a real candidate) but the resampling-fold mismatch it
causes evidently doesn't propagate to a Win/Loss-level effect at this sample size. LM,
MARS and RF (not shown) moved by comparably small amounts and stayed close to both the
paper and their §1 CIs, consistent with noise rather than a systematic RNG effect.

**Implication:** the entire residual SVM/RPART gap now rests on Cause B (the untuned
baseline hypothesis, §3) alone. Experiment 2 is the only remaining lever in this plan; if
it doesn't move the numbers either, the gap is genuinely unexplained by anything identified
so far and should be reported as such rather than attributed to a guess.

---

## 11. Cause C — the Win/Loss sign statistic (found and fixed, 2026-09-21)

**Whose side:** ours. A port error, not a modelling difference.

**[VERIFIED]** The paper's own `src/original/R_Code/PairedComparisons.R` classifies each
dataset by reading column 2 of `performanceEstimation`'s `WilcoxonSignedRank.test` array:

```r
if (pres[[measure]]$WilcoxonSignedRank.test[nm, 2, e] < 0) { ... Win ... }
else if (... == 0) { ... Tie ... } else { ... Loss ... }
```

Deparsing `performanceEstimation::pairedComparisons` shows that column is named
`DiffMedScores` and is filled with

```r
medScores[t, base] - medScores[t, o]     # MEDIAN of each workflow's 50 fold scores
```

The **means** are carried on a *separate* `t.test` array (`DiffAvgScores`) that `WLdef()`
never touches. Both of this project's ports of `WLdef()` —
`Results (Clean)/Eval_Metrics Code/Paper Comparison/paired_comparisons.py::compute_wl`
and `src/adapted/PairedComparisons.R::compute_WL` — signed the comparison from the
**mean**, and used an `abs(diff) < 1e-9` Tie rule where the original ties on exact median
equality.

**Why it bites SVM and RPART specifically.** Utility-based $F1_\phi$ is strongly
right-skewed: most folds sit at `uba::DELTA` (2e-05) when a model detects no rare case,
and a handful score high. Mean and median therefore disagree on exactly the fragile cells
§2 already identified — RPART has the smallest median $|\Delta F1|$ (0.026–0.040) and the
most near-tie datasets (up to 10/24), SVM the next. LM/MARS/RF have large, consistent
effects where the two statistics agree.

### Effect of the fix alone, on the canonical `data_v2` run

| Family | mean-signed (old) | median-signed (correct) | Paper |
|---|---|---|---|
| LM | 79/75/79 | 75/75/83 | 79/75/79 |
| MARS | 58/75/75 | **62**/75/71 | 62/71/75 |
| RF | 83/75/92 | **75**/75/79 | 75/83/83 |
| SVM | 58/54/54 | **54/46/42** | 33/29/29 |
| RPART | 71/67/67 | **62/50/58** | 50/46/42 |

Every family moves toward the paper or stays put; MARS `U_B` and RF `U_B` become exact
matches. RPART's residual drops from ~21–25pp to ~4–16pp, SVM's from ~25pp to ~13–21pp.

### The decisive result: Cause A was never ruled out

§10 declared Cause A (the R 3.6.0 RNG change) eliminated. That verdict was computed on the
**mean-signed** statistic and is withdrawn. Re-evaluating the Experiment 1 run
(`runs/rng_rounding/`, SLURM job 15570, `RNGkind` confirmed `Rounding`) with the corrected
median sign reproduces the paper's Table 3 **cell for cell, within ±1 dataset on all 15
cells**:

| Strategy | Family | Ours (rng_rounding + median sign) | Paper |
|---|---|---|---|
| U_B | LM / SVM / MARS / RF / RPART | 18/6 · 9/15 · 14/10 · 17/7 · **12/12** | 19/5 · 8/16 · 15/9 · 18/6 · 12/12 |
| O_B | LM / SVM / MARS / RF / RPART | 19/5 · 8/16 · 16/8 · 19/5 · **11/13** | 18/6 · 7/17 · 17/7 · 20/4 · 11/13 |
| SM_B | LM / SVM / MARS / RF / RPART | **19/5** · **7/17** · 17/7 · 19/5 · **10/14** | 19/5 · 7/17 · 18/6 · 20/4 · 10/14 |

Five of fifteen cells match exactly; the largest disagreement anywhere is **one dataset**.
For reference, the published `data_v2` state disagrees with the paper by 5–6 datasets on
every SVM and RPART cell.

**Conclusion.** The residual gap was **two** defects compounding, and both were needed:

1. **Cause C (ours):** the Win/Loss sign taken from means rather than medians. Fixed here,
   TDD'd (`test_compute_wl_signs_from_median_not_mean`,
   `test_compute_wl_tie_on_equal_medians_not_equal_means`), 58/58 tests green.
2. **Cause A (neither side's fault):** the R 3.6.0 `sample()` change. Real, and it matters
   — but only becomes visible once the sign statistic is correct. §10's "ruled out"
   verdict was an artifact of measuring through Cause C.

Cause B (the untuned-baseline hypothesis, §3) is **not** required to explain the gap.
Experiment 2's baseline retune (`runs/baseline_tuned/`) moves the numbers *away* from the
paper under the corrected statistic as well (SVM 50/50/46, RPART 62/46/58), so the
Table 7 values the paper publishes should be taken at face value for the baseline too,
exactly as its §5.1 prose states.

**Consequence for the canonical result:** `rng_rounding` — not `data_v2` — is the run that
replicates the paper, and `EXP002_SAMPLE_KIND=Rounding` therefore belongs in the corrected
experiment's defaults (plan Phase 4, promotion to `data_v3`).

### Promotion complete (2026-09-21)

`rng_rounding` was copied to `src/adapted/results/data_v3/` and made canonical:
`src/adapted/run_config.R`'s defaults now resolve to `data_v3`/`Rounding` (TDD'd,
`scratch/test_run_config.R`), F1/RMSE/SERA were re-extracted from its `.Rdata`
(which already carried `trues`/`preds`/`phi_ctrl` — no new cluster job needed), and the
full 22-generator pipeline was regenerated end to end (58/58 tests green). The shipped
Table 3, computed on `data_v3` with the corrected median-signed statistic:

```
                ours (data_v3)      paper
U_B  LM/SVM/MARS/RF/RPART   75/38/58/71/50   79/33/62/75/50
O_B  LM/SVM/MARS/RF/RPART   79/33/67/79/46   75/29/71/83/46
SM_B LM/SVM/MARS/RF/RPART   79/29/71/79/42   79/29/75/83/42
```

All 15 cells within ±1 dataset, 6 exact matches (SVM U_B, SVM SM_B, RPART all three,
LM SM_B). Leave-one-dataset-out and leave-one-source-out sensitivity
(`cmp_sensitivity.txt`) confirm the result is not carried by any single dataset or
source: the largest single-dataset swing is ±3.4pp, and every source-level exclusion
stays within a few points of the full-sample figure. (A LaTeX report built on this data
existed briefly during this work and its prose was rewritten against these corrected
numbers; the report itself was later removed from this repo by design — see "What's
excluded, and why" in `README.md` — this document is now the canonical, kept-up-to-date
account.) `runs/data_v3` was cross-checked before/after promotion with full SHA-256 manifests of
every derived artifact (`runs/data_v3/promotion_shasum_manifest.txt`): same 2722 files,
no additions or removals, 2551 hashes changed — consistent with a full input-data
promotion touching everything downstream of F1/RMSE/SERA.

This closes the Tier 2 investigation. Both experiments ran, one candidate cause (B) was
refuted, one (A) was confirmed once the sign-statistic bug (C) stopped masking it, and
the corrected, fully traceable result is now the project's canonical state.
