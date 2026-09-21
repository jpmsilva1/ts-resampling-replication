#!/usr/bin/env Rscript
# Extracts trues/preds/phi_ctrl (added to the new-format results_dataset_N.Rdata
# via src/adapted/Exps.R's capture change) into CSVs, one file PER
# (dataset, workflow) -- not one combined file per dataset, per the Apuana
# skill's R-12 finding (avoid one giant in-memory rbind() across all 52
# workflows x 50 iterations x test-fold rows).
#
# Usage: Rscript src/stage0/rdata_to_csv_predictions.R <rdata_dir> <out_dir> <manifest_csv>
# Defaults point at the real rerun's expected location; pass fixture paths
# to validate against the local mini-test's saved fixture instead.
suppressMessages(library(performanceEstimation))

args <- commandArgs(trailingOnly = TRUE)
root <- "."  # run from the project root, as the Usage line above documents
rdata_dir <- if (length(args) >= 1) args[1] else file.path(root, "src/adapted/results/data_v2")
out_dir   <- if (length(args) >= 2) args[2] else file.path(root, "Results (Clean)/Results Data/raw_predictions_by_dataset")
manifest_path <- if (length(args) >= 3) args[3] else file.path(root, "data/paper_datasets_csv/manifest.csv")

dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)
manifest <- if (file.exists(manifest_path)) read.csv(manifest_path, stringsAsFactors = FALSE) else NULL

rdata_files <- sort(list.files(rdata_dir, pattern = "^results_dataset_.*\\.Rdata$", full.names = TRUE))
if (length(rdata_files) == 0) stop("No results_dataset_*.Rdata files found in ", rdata_dir)

for (rdata_path in rdata_files) {
  ds_idx <- as.integer(gsub(".*results_dataset_([0-9]+)(_mini)?\\.Rdata", "\\1", basename(rdata_path)))
  ds_id  <- if (!is.null(manifest) && ds_idx <= nrow(manifest)) manifest$dataset_id[ds_idx] else sprintf("DS%02d", ds_idx)

  load(rdata_path)  # loads 'exp'
  task <- taskNames(exp)[1]
  wfs  <- workflowNames(exp)

  ds_out_dir <- file.path(out_dir, ds_id)
  dir.create(ds_out_dir, showWarnings = FALSE, recursive = TRUE)

  for (wf in wfs) {
    obj <- exp[[task]][[wf]]
    n_it <- length(obj@iterationsInfo)

    pred_rows <- vector("list", n_it)
    phi_rows  <- vector("list", n_it)

    for (it in seq_len(n_it)) {
      info <- obj@iterationsInfo[[it]]
      if (is.null(info$trues) || is.null(info$preds)) next  # old-format fold, skip

      n_obs <- length(info$trues)
      pred_rows[[it]] <- data.frame(
        iteration = it, obs_index = seq_len(n_obs),
        y_true = info$trues, y_pred = info$preds
      )
      ctrl <- matrix(info$phi_ctrl, ncol = 3, byrow = TRUE)
      phi_rows[[it]] <- data.frame(
        iteration = it, point_index = seq_len(nrow(ctrl)),
        ctrl_x = ctrl[, 1], ctrl_phi = ctrl[, 2], ctrl_deriv = ctrl[, 3]
      )
    }

    pred_df <- do.call(rbind, pred_rows)
    phi_df  <- do.call(rbind, phi_rows)
    if (is.null(pred_df)) next

    write.csv(pred_df, file.path(ds_out_dir, sprintf("%s_predictions.csv", wf)), row.names = FALSE)
    write.csv(phi_df, file.path(ds_out_dir, sprintf("%s_phi_ctrl.csv", wf)), row.names = FALSE)
  }
  rm(exp)
  cat(sprintf("%s: wrote %d workflows to %s\n", ds_id, length(wfs), ds_out_dir))
}
cat("Done.\n")
