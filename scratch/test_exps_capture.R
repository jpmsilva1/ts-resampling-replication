#!/usr/bin/env Rscript
# Seam-1 check (TDD): confirms the trues/preds/phi_ctrl capture added to
# src/adapted/Exps.R's workflow functions actually lands correctly in
# @iterationsInfo, on a tiny local run -- no cluster needed. Covers a plain
# workflow (mc.lm_OVERB) and an SVM workflow that goes through cap_svm_train
# (mc.svm_OVERB), since those two code paths differ.
suppressMessages(library(performanceEstimation))
suppressMessages(library(uba))
suppressMessages(library(UBL))
suppressMessages(library(DMwR))
suppressMessages(library(lubridate))
suppressMessages(library(xts))

# Run from the project root, as every other scratch/*.R script here does --
# no setwd() to an absolute path (self-contained-project-folder rule).

# Source only the function definitions from Exps.R, not its bottom driver
# loop (which would kick off the full 20-dataset run).
lines <- readLines("src/adapted/Exps.R")
split_at <- grep("^#DEFINITION OF VARIABLES FOR RESAMPLING", lines)
stopifnot(length(split_at) == 1)
funcs_file <- tempfile(fileext = ".R")
writeLines(lines[1:(split_at - 1)], funcs_file)
source(funcs_file)

load("src/original/Data/data_NM_PB_LT_DSAA2016.Rdata")
ds <- create.data(data[[1]], 10)
ds <- ds[complete.cases(ds), ]
form <- as.formula(V10 ~ .)

exp <- performanceEstimation(
  PredTask(form, ds),
  c(Workflow("mc.lm_OVERB"), Workflow("mc.svm_OVERB", cost = 150, gamma = 0.001)),
  EstimationTask("totTime", method = MonteCarlo(nReps = 2, szTrain = .5, szTest = .25))
)

task <- taskNames(exp)[1]
n_test_expected <- round(nrow(ds) * .25)

check_workflow <- function(wf) {
  info <- exp[[task]][[wf]]@iterationsInfo[[1]]
  stopifnot("trues" %in% names(info), "preds" %in% names(info), "phi_ctrl" %in% names(info))
  stopifnot(is.numeric(info$trues), is.numeric(info$preds))
  stopifnot(length(info$trues) == length(info$preds))
  stopifnot(abs(length(info$trues) - n_test_expected) <= 1)  # rounding
  # phi.control()$control.pts is a flat numeric vector (row-major (value,
  # phi, derivative) triples), not a matrix -- dim is stripped internally
  # by phi.setup()/phi.extremes(). Reshape in Python via .reshape(-1, 3).
  stopifnot(is.numeric(info$phi_ctrl))
  stopifnot(length(info$phi_ctrl) %% 3 == 0)
  cat(sprintf("PASS %-20s trues/preds len=%d, phi_ctrl len=%d (%d pts)\n",
              wf, length(info$trues), length(info$phi_ctrl), length(info$phi_ctrl) / 3))
}

check_workflow("mc.lm_OVERB")
check_workflow("mc.svm_OVERB")
cat("All seam-1 checks passed.\n")

# Save as a fixture so the CSV-extraction script can be validated end-to-end
# locally, without waiting for the real cluster rerun.
dir.create("scratch/fixtures", showWarnings = FALSE)
save(exp, file = "scratch/fixtures/results_dataset_1_mini.Rdata")
cat("Saved fixture: scratch/fixtures/results_dataset_1_mini.Rdata\n")
