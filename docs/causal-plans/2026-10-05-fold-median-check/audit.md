# Audit Report: Fold-aggregation (median) robustness check

**Date**: 2026-10-05
**Method audited**: Designed experiment (fully crossed 24 datasets × 5 families × 10 strategies), pooled paired effect with cell bootstrap; new median-over-folds robustness check (report §4.2 step 5, Table `tab:exp002_fold_median_check`)
**Overall assessment**: Yellow (fixable concerns)

## Summary

The check's main conclusion holds. Every strategy is worse than the baseline on SERA and RMSE whether cells use the mean or the median of their folds, and the design gives no route for confounding. But three claims in the new report text go further than the evidence:
- the blow-up mechanism is attributed to MARS in general, when it is specific to one fitted model;
- resampled models are said to blow up "more often than the baseline", which is false for moderate blow-ups under SMOTE;
- the median magnitudes read as a corrected effect, when they are a different estimand.

The SERA *magnitudes* also rest heavily on 3 of the 120 cells. That deserves a between-cell robustness check, which the median-over-folds check does not provide.

Diagnostics: `audit_diagnostics.py` (this folder), run against `Results (Clean)/Results Data/raw_sera_by_dataset/`.

## Findings

### Serious: SERA magnitudes are driven by three of 120 cells
**Category**: 4 Statistical
**Explanation**: The median-over-folds check only addresses outliers *within* a cell. Dropping three whole cells (DS22/MARS, DS23/LM, DS23/MARS) from the *mean-based* estimate already explains 38–91% of the gap between the mean and median estimates (OVERT: 91%; UNDER*: 51–53%; SMOTETPhi: 38%). The pooled estimator, an unweighted mean of 120 log ratios, is itself sensitive to a few extreme cells. Neither the median-over-folds check nor the leave-one-source-out check isolates single cells; the source check only drops whole sources.
**Diagnostic**:
```python
dom3 = [("DS22","mars"), ("DS23","lm"), ("DS23","mars")]
res = pd.DataFrame({"mean": pooled("mean"),
                    "mean_drop3cells": pooled("mean", dom3),
                    "median": pooled("median")})
# OVERT: mean 0.285 -> 0.128 without 3 cells -> 0.112 median
```
**Suggested fix**: Add a between-cell robust pooled estimate next to the mean, for example a 10% trimmed mean of the 120 log ratios or the Hodges–Lehmann estimator, or report a leave-one-cell-out influence range. Quote SERA magnitudes as a range across estimators rather than one number.
**Why this matters**: The *direction* is safe, since every interval excludes zero under every variant. But figures like "SMOTETPhi +0.426" or "OVERT +0.285" are fragile. A reader comparing strategies by magnitude (OVERT vs. OVERB, for example) could rank them wrongly: OVERT looks clearly worse than OVERB under the mean (0.285 vs. 0.180), and the two are tied under the median (0.112 vs. 0.104).

### Minor: The median is a different estimand, not a corrected one
**Category**: 2 Identification
**Explanation**: The mean over folds estimates the *expected* error of deploying the strategy. The median over folds estimates the error on a *typical* fold. For a forecaster, a fold where the model predicts 3.8×10⁸ for a price of about 9,400 is a real cost. The report's wording ("the conclusion holds but the size shrinks") could be read as the median giving the true size.
**Diagnostic**: Within a typical cell the two agree (the median across cells of mean/median per cell is 0.98–1.01 for every arm), so the estimands differ only in the tails, which is exactly where they disagree.
**Suggested fix**: Add one sentence: the mean-based figures measure expected error including rare failures, and the median-based figures measure typical-fold error; both are valid, and they answer different questions.
**Why this matters**: Without it, a reader may discard the tail risk that is a genuine downside of some resampled models.

### Minor: "Resampled models produce these failures more often than the baseline" is only partly true
**Category**: 3 Data
**Explanation**: Share of folds whose SERA exceeds 100× their cell's median:

| Arm | Folds above 100× the cell median |
|---|---|
| baseline | 0.00% |
| SMOTE / OVER | 0.02–0.05% |
| UNDER | 0.12–0.22% |

At a 10× threshold the baseline is at 0.60%, *above* SMOTET (0.42%) and SMOTETPhi (0.53%), close to SMOTEB (0.63%) and OVER* (0.70–0.78%), and well below UNDER* (1.35–1.50%). So the claim holds for extreme failures and for undersampling, not for resampling in general.
**Diagnostic**: Part (b) of `audit_diagnostics.py`.
**Suggested fix**: Reword to say that extreme failures (more than 100× the typical fold) occur only under resampling, and that moderate ones are most frequent under undersampling.
**Why this matters**: As written, the sentence overstates a mechanism the data only partly supports.

### Minor: The blow-up is specific to the OVERT fit, not to MARS
**Category**: 1 Assumptions (mechanism claim)
**Explanation**: The report says MARS's piecewise-linear terms "keep extending beyond the training range". On the same four test rows in fold 16, baseline MARS predicts 1,123–1,808, and OVERB, UNDERB and SMOTEB predict 456–3,105. All of these models see the same out-of-range lags (up to 9,702), but only OVERT's fitted model has terms steep enough to reach 10⁸.
**Diagnostic**:
```python
for w in ["mc.mars", "mc.mars_OVERB", "mc.mars_UNDERB", "mc.mars_SMOTEB", "mc.mars_OVERT"]:
    p = pd.read_csv(f"raw_predictions_by_dataset/DS22/{w}_predictions.csv")
    print(w, p[(p.iteration == 16) & p.obs_index.between(9385, 9390)].y_pred.round(0).tolist())
```
**Suggested fix**: Say that the out-of-range inputs reach every model, but only the OVERT fit on that fold turns them into a blow-up. The training-window maximum (about 8,600) was reconstructed from the split sizes. It can be confirmed exactly from the `train` indices saved in `results_dataset_22.Rdata` (`iterationsInfo`).
**Why this matters**: A reader could conclude that MARS is unsafe in general. The evidence shows one unlucky fitted model.

### Minor: The "12 dominated cells" count depends on the threshold and mixes metrics
**Category**: 4 Statistical
**Explanation**: The 12 are 9 SERA plus 3 RMSE (dataset, family, strategy) entries. For SERA alone, the count moves from 2 (more than 80% of the fold sum and more than 1,000× the median) to 13 (more than 30% and more than 10×).
**Suggested fix**: State "9 SERA and 3 RMSE cells" and the threshold used, or describe the cells qualitatively ("a handful, all on DS22/DS23").
**Why this matters**: Precision only. No conclusion depends on the count.

### Minor: The bootstrap assumes independent cells (already disclosed)
**Category**: 4 Statistical
**Explanation**: The intervals in the new table use the same cell bootstrap as Figure 7. Cells share datasets and sources, so the intervals are too narrow. The report already says so in step (3). The median columns inherit the same problem.
**Suggested fix**: Optionally, a cluster bootstrap that resamples datasets (24) or sources (7) with all their cells. The direction will almost certainly survive, since every lower bound is at least 0.07.
**Why this matters**: Intervals only; direction is unaffected.

### Checked and fine
- **Random assignment and confounding**: every strategy is applied to every (dataset, family, fold), so assignment is design-balanced and no backdoor path exists. Dataset and family are blocking factors, not confounders.
- **Measurement consistency**: φ for SERA is fit on the *pre-resampling* training data in every workflow (`ph <- uba::phi.control(train[...])` comes before `randOverRegressT(...)` in `Exps.R`), so all arms are scored against the same relevance function.
- **SUTVA**: all arms share identical folds and test sets; there is no interference between arms.
- **Attrition**: the `complete.cases` row loss (DS12, DS13, DS23, DS24) applies identically to all arms, so it is not differential.
- **Compliance**: the zero-relevance guard (D3) means a few folds receive no resampling. This is measured as essentially absent (1 tie across 15 cells), so it causes negligible dilution.
- **Multiple analyses**: the median check was added after seeing the outlier, which is a forking path. That is mitigated because both versions are reported side by side and no conclusion is switched to the median.

### External validity (Category 5)
The failure mode is limited to models that can extrapolate (MARS, LM). RF and RPART cannot predict outside the training range, and none of the dominated cells involve them. It also needs series with extreme regime spikes (DS22 price, DS23 water). The finding should not be generalized to "resampling causes instability" across learners.

## Recommendations
1. **(Serious)** Add a between-cell robust pooled estimate (trimmed mean or Hodges–Lehmann) next to Table 4, and quote SERA magnitudes as a range.
2. Reword three sentences in step (5): the mechanism (specific to the OVERT fit), the frequency claim (extreme failures only, most common under undersampling), and the estimand framing (expected error vs. typical-fold error).
3. Make the dominated-cell count precise (9 SERA + 3 RMSE, with the threshold).
4. Optional: a cluster bootstrap by source, and confirming the training-window maximum from the saved `train` indices.

## Next Steps

> **Variable selection check**: Want to verify your variable choices are safe? Run `/causal-dag` to map the causal structure and check for bad controls.

## Resolution (2026-10-05)
All four recommendations applied in `Exp_Report_Main/main.tex` §4.2 step (5) and
`Causal/generate_fold_median_check.py`:
1. Table 4 now has a third column per metric, *Trimmed across cells* (10% trimmed mean of the
   120 cell log ratios, same cell bootstrap). It agrees closely with the median of folds (e.g.
   OVERT +0.113 vs. +0.112) and both sit 10–60% below the plain mean for SERA. The text asks
   readers to take SERA magnitudes as a range, and notes that OVERT vs. OVERB is a tie under
   both robust versions.
2. Reworded: the blow-up is specific to the OVERT fit (baseline/OVERB/UNDERB/SMOTEB predict
   456–3,105 on the same rows); frequency claim split into >100× (resampling only) and >10×
   (most under undersampling, SMOTE not above baseline); estimand framing (expected error vs.
   typical-fold error).
3. Cell count stated as 9 SERA + 3 RMSE under the named rule, with its 2–13 sensitivity.
4. Not done (optional): cluster bootstrap by source; exact training-window maximum from the saved
   `train` indices.
