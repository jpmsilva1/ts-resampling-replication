#!/usr/bin/env Rscript
# Converts the raw results_dataset_N.Rdata files (performanceEstimation
# ComparisonResults objects) into one tidy long-format CSV per dataset:
# columns dataset_id, dataset_name, workflow, iteration, prec, rec, F1.
#
# Usage: Rscript src/stage0/rdata_to_csv.R <rdata_dir> <out_dir> <manifest_csv>
# Same argv interface as src/stage0/rdata_to_csv_predictions.R -- one convention,
# not two. Defaults point at the canonical data_v2 run unless EXP002_RUN_ID
# is set (see src/adapted/run_config.R), so a Tier 2 variant run's manifest
# CSV lands under its own runs/<run_id>/ tree without touching data_v2's.
suppressMessages(library(performanceEstimation))

root <- "."  # run from the project root, as the Usage line above documents
args <- commandArgs(trailingOnly = TRUE)
run_id <- Sys.getenv("EXP002_RUN_ID", "data_v2")

data_dir <- if (length(args) >= 1) args[1] else file.path(root, "src/adapted/results", run_id)
#ponytail: reads data_v2 (post cap_svm_train fix) so F1 comes from the same
#run as RMSE/SERA; the old data/ extraction is left in raw_iterations_by_dataset/
#untouched so previously-published F1 tables stay reproducible.
out_dir <- if (length(args) >= 2) args[2] else file.path(root, "Results (Clean)/Results Data/raw_iterations_by_dataset_v2")
manifest_path <- if (length(args) >= 3) args[3] else file.path(root, "data/paper_datasets_csv/manifest.csv")
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

manifest <- read.csv(manifest_path, stringsAsFactors = FALSE)
available <- sort(as.integer(gsub(".*results_dataset_([0-9]+)\\.Rdata", "\\1",
                                   list.files(data_dir, pattern = "results_dataset_.*\\.Rdata$"))))
n_total <- length(available)

for (i in available) {
  rdata_path <- file.path(data_dir, sprintf("results_dataset_%d.Rdata", i))
  load(rdata_path)  # loads 'exp'

  ds_id   <- manifest$dataset_id[i]
  ds_name <- manifest$description[i]
  task    <- taskNames(exp)[1]
  wfs     <- workflowNames(exp)

  rows <- do.call(rbind, lapply(wfs, function(wf) {
    obj <- exp[[task]][[wf]]
    n_it <- length(obj@iterationsInfo)
    do.call(rbind, lapply(seq_len(n_it), function(it) {
      ev <- obj@iterationsInfo[[it]]$evaluation
      data.frame(
        dataset_id   = ds_id,
        dataset_name = ds_name,
        workflow     = wf,
        iteration    = it,
        prec         = unname(ev["prec"]),
        rec          = unname(ev["rec"]),
        F1           = unname(ev["F1"])
      )
    }))
  }))

  out_file <- file.path(out_dir, sprintf("%s_%s.csv", ds_id, ds_name |>
    tolower() |> gsub("[^a-z0-9]+", "_", x = _) |> gsub("^_|_$", "", x = _)))
  write.csv(rows, out_file, row.names = FALSE)
  cat(sprintf("[%2d/%d] %s -> %s (%d rows)\n", match(i, available), n_total, ds_id, basename(out_file), nrow(rows)))
}

cat("Done.\n")
