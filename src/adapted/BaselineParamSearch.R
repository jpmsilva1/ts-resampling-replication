#!/usr/bin/env Rscript
# Experiment 2 (Cause B), Phase 2a of ~/.claude/plans/luminous-noodling-bear.md.
#
# Tests whether the untuned BASELINE itself (not untuned resampling) explains
# the residual SVM/RPART disagreement with Moniz et al. (2017) Table 3: the
# paper's own OptParmsSearch.R tuned every workflow, including the
# unresampled baseline, on its own objective; src/adapted/Exps.R applies one
# Table 7 row to a whole family instead. This searches SVM/RPART's baseline
# hyperparameters per dataset, against the UNRESAMPLED baseline only, 10 MC
# reps, selecting on median F1 -- mirroring src/original/R_Code/
# OptParmsSearch.R's method (per-dataset grid search over Workflow() calls,
# selecting on median F1 across nReps), but scoped to only the two workflows
# and grids this experiment needs.
#
# Grids are the paper's own published search space (Annex 1), the same one
# Results (Clean)/Eval_Metrics Code/Paper Comparison/paper_reference.py's
# PARAM_GRIDS already validates OPT_PARAMS_TABLE7 against -- see
# scratch/test_baseline_param_search.R's cross-check.
#
# Output: src/adapted/baseline_params_svm_rpart.csv, one row per dataset:
# dataset,svm_cost,svm_gamma,svm_median_f1,rpart_minsplit,rpart_cp,rpart_median_f1
# This is what EXP002_BASELINE_PARAMS (src/adapted/run_config.R) reads.
#
# Reuses Exps.R's function definitions (create.data, cap_svm_train, mc.svm,
# mc.rpart, eval.stats) via the same "source only definitions before the
# driver loop" pattern as scratch/test_exps_capture.R -- not re-implemented.
# Per-dataset try() + checkpoint-skip (resumable CSV), same pattern as
# Exps.R's own driver.
suppressMessages(library(performanceEstimation))
suppressMessages(library(uba))
suppressMessages(library(UBL))
suppressMessages(library(DMwR))
suppressMessages(library(lubridate))
suppressMessages(library(xts))
suppressMessages(library(e1071))
suppressMessages(library(rpart))
library(parallelMap)

.baseline_search_funcs <- local({
  lines <- readLines("src/adapted/Exps.R")
  split_at <- grep("^#DEFINITION OF VARIABLES FOR RESAMPLING", lines)
  stopifnot(length(split_at) == 1)
  funcs_file <- tempfile(fileext = ".R")
  writeLines(lines[1:(split_at - 1)], funcs_file)
  funcs_file
})
source(.baseline_search_funcs)

TOTAL_DS <- 24
MC_SZ_TRAIN <- c(rep(.5, 20), .1, .1, .2, .2)
MC_SZ_TEST  <- c(rep(.25, 20), .05, .05, .1, .1)
N_REPS <- 10
OUT_FILE <- "src/adapted/baseline_params_svm_rpart.csv"

SVM_GRID   <- expand.grid(cost = c(10, 150, 300), gamma = c(0.01, 0.001))
RPART_GRID <- expand.grid(minsplit = c(10, 20, 30), cp = c(0.001, 0.01, 0.1))

# Deterministic selection: argmax, first row wins a tie. Pure function --
# unit-tested directly in scratch/test_baseline_param_search.R without
# needing a real performanceEstimation() run.
pick_best_row <- function(grid, scores) {
  grid$median_f1 <- scores
  grid[which.max(scores), ]
}

median_f1_of <- function(exp, task, wf, n_reps) {
  fs <- sapply(seq_len(n_reps), function(it) exp[[task]][[wf]]@iterationsInfo[[it]]$evaluation["F1"])
  median(fs)
}

# One performanceEstimation() call per grid row (not workflowVariants()'s
# auto-numbered variants) -- keeps the row/parameter mapping explicit rather
# than relying on the package's internal naming order.
search_grid <- function(form, ds, mc_train, mc_test, workflow_name, grid, param_names) {
  scores <- numeric(nrow(grid))
  for (r in seq_len(nrow(grid))) {
    args <- as.list(grid[r, , drop = FALSE])
    names(args) <- param_names
    wf <- do.call(Workflow, c(list(workflow_name), args))
    exp <- performanceEstimation(PredTask(form, ds), c(wf),
                                  EstimationTask("totTime", method = MonteCarlo(nReps = N_REPS, szTrain = mc_train, szTest = mc_test)),
                                  cluster = TRUE)
    task <- taskNames(exp)[1]
    wfname <- workflowNames(exp)[1]
    scores[r] <- median_f1_of(exp, task, wfname, N_REPS)
  }
  pick_best_row(grid, scores)
}

#DRIVER LOOP
load("src/original/Data/data_NM_PB_LT_DSAA2016.Rdata")

results <- if (file.exists(OUT_FILE)) {
  read.csv(OUT_FILE)
} else {
  data.frame(dataset = integer(0), svm_cost = numeric(0), svm_gamma = numeric(0),
             svm_median_f1 = numeric(0), rpart_minsplit = numeric(0), rpart_cp = numeric(0),
             rpart_median_f1 = numeric(0))
}

for (i in 1:TOTAL_DS) {
  if (i %in% results$dataset) {
    cat(sprintf("Dataset %d already searched. Skipping.\n", i))
    next
  }

  ds <- create.data(data[[i]], 10)
  ds <- ds[complete.cases(ds), ]
  form <- as.formula(V10 ~ .)

  ncores <- as.numeric(Sys.getenv("SLURM_CPUS_PER_TASK", "48"))
  parallelStartMulticore(cpus = ncores)
  out <- try({
    svm_best   <- search_grid(form, ds, MC_SZ_TRAIN[i], MC_SZ_TEST[i], "mc.svm",   SVM_GRID,   c("cost", "gamma"))
    rpart_best <- search_grid(form, ds, MC_SZ_TRAIN[i], MC_SZ_TEST[i], "mc.rpart", RPART_GRID, c("minsplit", "cp"))
    list(svm_best = svm_best, rpart_best = rpart_best)
  })
  parallelStop()

  if (inherits(out, "try-error")) {
    cat(sprintf("!!! Dataset %d FAILED: %s\n", i, conditionMessage(attr(out, "condition"))))
    next
  }

  row <- data.frame(dataset = i,
                     svm_cost = out$svm_best$cost, svm_gamma = out$svm_best$gamma,
                     svm_median_f1 = out$svm_best$median_f1,
                     rpart_minsplit = out$rpart_best$minsplit, rpart_cp = out$rpart_best$cp,
                     rpart_median_f1 = out$rpart_best$median_f1)
  results <- rbind(results, row)
  write.csv(results, OUT_FILE, row.names = FALSE)
  cat(sprintf("Dataset %d: SVM cost=%s gamma=%s (F1=%.4f); RPART minsplit=%s cp=%s (F1=%.4f)\n",
              i, row$svm_cost, row$svm_gamma, row$svm_median_f1, row$rpart_minsplit, row$rpart_cp, row$rpart_median_f1))
}

cat("Done. Wrote", OUT_FILE, "\n")
