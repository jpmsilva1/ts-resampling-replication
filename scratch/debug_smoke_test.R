#!/usr/bin/env Rscript
# Cluster debug-partition smoke test for Tier 2's new run_config.R knobs --
# run this under `srun --partition=debug` BEFORE submitting the real 2-day
# short-simple job, so a startup-level bug (package load, RNGkind, manifest
# write, the baseline-param override wiring) is caught in under debug's
# 30-min cap instead of burning queue time on a broken job. Exercises the
# exact code paths Tier 2 needs: EXP002_RUN_ID, EXP002_SAMPLE_KIND,
# RUN_MANIFEST.txt, and one real performanceEstimation() call touching the
# two baseline-override workflows (mc.svm, mc.rpart) plus one resampled
# variant of each.
#
# Run from the project root on the cluster (after `micromamba activate
# exp002_env`):
#   EXP002_RUN_ID=_debug_smoke EXP002_SAMPLE_KIND=Rounding Rscript scratch/debug_smoke_test.R
#
# Writes only under src/adapted/results/_debug_smoke/, deleted at the end --
# never touches a real run's directory or checkpoint files.
suppressMessages(library(performanceEstimation))
suppressMessages(library(uba))
suppressMessages(library(UBL))
suppressMessages(library(DMwR))
suppressMessages(library(lubridate))
suppressMessages(library(xts))
suppressMessages(library(e1071))
suppressMessages(library(earth))
suppressMessages(library(randomForest))
suppressMessages(library(rpart))
cat("[1/6] All Exps.R package dependencies load OK.\n")

source("src/adapted/run_config.R")
cat("[2/6] run_config.R sources OK.\n")

stopifnot(nzchar(Sys.getenv("EXP002_RUN_ID")), nzchar(Sys.getenv("EXP002_SAMPLE_KIND")))
results_dir <- resolve_results_dir()
stopifnot(results_dir == file.path("src/adapted/results", Sys.getenv("EXP002_RUN_ID")))
cat(sprintf("[3/6] resolve_results_dir() -> %s (matches EXP002_RUN_ID)\n", results_dir))

sample_kind <- resolve_sample_kind()
stopifnot(identical(sample_kind, Sys.getenv("EXP002_SAMPLE_KIND")))
RNGkind(sample.kind = sample_kind)
stopifnot(RNGkind()[3] == sample_kind)
cat(sprintf("[4/6] RNGkind override applied and confirmed: %s\n", paste(RNGkind(), collapse = ", ")))

dir.create(results_dir, recursive = TRUE, showWarnings = FALSE)
manifest_file <- file.path(results_dir, "RUN_MANIFEST.txt")
unlink(manifest_file)
ensure_manifest_header(manifest_file, run_id = Sys.getenv("EXP002_RUN_ID"), total_ds = 1,
                        sample_kind_override = sample_kind, baseline_params_file = "(none)",
                        exps_r_path = "src/adapted/Exps.R")
stopifnot(file.exists(manifest_file))
cat("[5/6] RUN_MANIFEST.txt written OK.\n")

# Source only Exps.R's function definitions (not its driver loop) -- same
# pattern as test_exps_capture.R -- and run one tiny slice on the cluster's
# actual R/package versions and CPU count, touching both baseline-override
# workflows plus one resampled variant of each.
lines <- readLines("src/adapted/Exps.R")
split_at <- grep("^#DEFINITION OF VARIABLES FOR RESAMPLING", lines)
stopifnot(length(split_at) == 1)
funcs_file <- tempfile(fileext = ".R")
writeLines(lines[1:(split_at - 1)], funcs_file)
source(funcs_file)

load("src/original/Data/data_NM_PB_LT_DSAA2016.Rdata")
ds <- create.data(data[[1]], 10)
ds <- ds[complete.cases(ds), ]
ds <- ds[1:min(300, nrow(ds)), ]  # trimmed hard -- this is a wiring smoke test, not a real fit
form <- as.formula(V10 ~ .)

library(parallelMap)
ncores <- min(as.numeric(Sys.getenv("SLURM_CPUS_PER_TASK", "2")), 2)
parallelStartMulticore(cpus = ncores)
exp <- performanceEstimation(
  PredTask(form, ds),
  c(Workflow("mc.svm", cost = 10, gamma = 0.01),
    Workflow("mc.svm_OVERB", cost = 300, gamma = 0.01),
    Workflow("mc.rpart", minsplit = 10, cp = 0.1),
    Workflow("mc.rpart_UNDERB", minsplit = 10, cp = 0.001)),
  EstimationTask("totTime", method = MonteCarlo(nReps = 2, szTrain = .5, szTest = .25)),
  cluster = TRUE
)
parallelStop()
stopifnot(all(c("mc.svm", "mc.svm_OVERB", "mc.rpart", "mc.rpart_UNDERB") %in% workflowNames(exp)))
append_manifest_line(manifest_file, dataset = 1, status = "ok", elapsed_min = 0.1)
cat("[6/6] Tiny performanceEstimation() slice (baseline + resampled, SVM + RPART) ran end to end OK.\n")

cat("\nSMOKE TEST PASSED. Safe to submit the real job.\n")
cat("Cleaning up:", results_dir, "\n")
unlink(results_dir, recursive = TRUE)
