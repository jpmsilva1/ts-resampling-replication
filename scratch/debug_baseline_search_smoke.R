#!/usr/bin/env Rscript
# Integration smoke test for src/adapted/BaselineParamSearch.R's real
# performanceEstimation() wiring (do.call(Workflow,...), cluster=TRUE,
# median_f1_of()'s @iterationsInfo extraction) -- scratch/
# test_baseline_param_search.R only covers the pure/deterministic
# pick_best_row() logic, not this. Run locally first (this file), then
# rerun the same idea on the cluster's debug partition before the real
# 24-dataset x 15-combo search job, same reasoning as
# scratch/debug_smoke_test.R for Exps.R itself.
#
# Trims the grid to 2 SVM configs and 2 RPART configs, 2 reps, 300 rows --
# a wiring check, not a real search. Writes nothing outside a temp OUT_FILE.
lines <- readLines("src/adapted/BaselineParamSearch.R")
split_at <- grep("^#DRIVER LOOP", lines)
stopifnot(length(split_at) == 1)
funcs_file <- tempfile(fileext = ".R")
writeLines(lines[1:(split_at - 1)], funcs_file)
source(funcs_file)

load("src/original/Data/data_NM_PB_LT_DSAA2016.Rdata")
ds <- create.data(data[[1]], 10)
ds <- ds[complete.cases(ds), ]
ds <- ds[1:min(300, nrow(ds)), ]
form <- as.formula(V10 ~ .)

tiny_svm_grid   <- expand.grid(cost = c(10, 300), gamma = c(0.01))
tiny_rpart_grid <- expand.grid(minsplit = c(10, 30), cp = c(0.1))

parallelStartMulticore(cpus = 2)
svm_best   <- search_grid(form, ds, .5, .25, "mc.svm",   tiny_svm_grid,   c("cost", "gamma"))
rpart_best <- search_grid(form, ds, .5, .25, "mc.rpart", tiny_rpart_grid, c("minsplit", "cp"))
parallelStop()

stopifnot(nrow(svm_best) == 1, nrow(rpart_best) == 1)
stopifnot(svm_best$cost %in% tiny_svm_grid$cost)
stopifnot(rpart_best$minsplit %in% tiny_rpart_grid$minsplit)
stopifnot(is.numeric(svm_best$median_f1), is.numeric(rpart_best$median_f1))

cat(sprintf("SVM best:   cost=%s gamma=%s median_f1=%.4f\n", svm_best$cost, svm_best$gamma, svm_best$median_f1))
cat(sprintf("RPART best: minsplit=%s cp=%s median_f1=%.4f\n", rpart_best$minsplit, rpart_best$cp, rpart_best$median_f1))
cat("BaselineParamSearch integration smoke test PASSED.\n")
