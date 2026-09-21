#!/usr/bin/env Rscript
# TDD seam for src/adapted/BaselineParamSearch.R's pure/deterministic part:
# pick_best_row() picks the argmax-F1 row out of a grid + score vector. The
# performanceEstimation() call around it is not unit-tested here (that's what
# scratch/debug_baseline_search_smoke.R's tiny real run is for) -- this test
# only pins down selection logic that a real run can't cheaply assert on
# (F1 outcomes are stochastic).
# Source only the definitions/grids, not the driver loop at the bottom (same
# convention as scratch/test_exps_capture.R uses for Exps.R) -- sourcing the
# whole file would kick off a real 24-dataset cluster-scale search.
lines <- readLines("src/adapted/BaselineParamSearch.R")
split_at <- grep("^#DRIVER LOOP", lines)
stopifnot(length(split_at) == 1)
funcs_file <- tempfile(fileext = ".R")
writeLines(lines[1:(split_at - 1)], funcs_file)
source(funcs_file)

assert <- function(cond, msg) if (!isTRUE(cond)) stop(msg, call. = FALSE)

test_pick_best_row_picks_the_max_score <- function() {
  grid <- expand.grid(cost = c(10, 150, 300), gamma = c(0.01, 0.001))
  scores <- c(0.1, 0.2, 0.9, 0.05, 0.3, 0.15)  # row 3 (cost=300, gamma=0.01) is max
  best <- pick_best_row(grid, scores)
  assert(best$cost == 300 && best$gamma == 0.01, "must pick the row matching the max score, not just the max score value")
  assert(best$median_f1 == 0.9, "median_f1 column must carry the winning score")
}

test_pick_best_row_ties_pick_first <- function() {
  grid <- expand.grid(minsplit = c(10, 20), cp = c(0.1))
  scores <- c(0.5, 0.5)
  best <- pick_best_row(grid, scores)
  assert(best$minsplit == 10, "a tie must resolve deterministically (first row), not error or randomize")
}

test_svm_grid_matches_paper_param_grids <- function() {
  # Cross-check against Results (Clean)/Eval_Metrics Code/Paper Comparison/
  # paper_reference.py's PARAM_GRIDS (the same published Annex 1 space
  # OPT_PARAMS_TABLE7 is already validated against) -- catches a typo in the
  # search grid rather than trusting it silently.
  assert(setequal(SVM_GRID$cost, c(10, 150, 300)), "SVM cost grid must match the paper's published search space")
  assert(setequal(SVM_GRID$gamma, c(0.01, 0.001)), "SVM gamma grid must match the paper's published search space")
  assert(nrow(SVM_GRID) == 6, "SVM grid must be the full 3x2 cross product")
}

test_rpart_grid_matches_paper_param_grids <- function() {
  assert(setequal(RPART_GRID$minsplit, c(10, 20, 30)), "RPART minsplit grid must match the paper's published search space")
  assert(setequal(RPART_GRID$cp, c(0.001, 0.01, 0.1)), "RPART cp grid must match the paper's published search space")
  assert(nrow(RPART_GRID) == 9, "RPART grid must be the full 3x3 cross product")
}

test_pick_best_row_picks_the_max_score()
test_pick_best_row_ties_pick_first()
test_svm_grid_matches_paper_param_grids()
test_rpart_grid_matches_paper_param_grids()
cat("All BaselineParamSearch tests passed.\n")
