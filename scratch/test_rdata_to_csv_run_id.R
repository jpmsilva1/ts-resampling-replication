#!/usr/bin/env Rscript
# TDD seam: rdata_to_csv.R's root/data_dir/out_dir must be overridable the
# same way rdata_to_csv_predictions.R already is (argv, with EXP002_RUN_ID
# influencing the default out_dir) -- not a second, divergent interface.
# Uses the real mini fixture (scratch/fixtures/results_dataset_1_mini.Rdata,
# built by test_exps_capture.R) rather than a hand-rolled stub, so this
# exercises the actual performanceEstimation object shape.
#
# Run: Rscript scratch/test_rdata_to_csv_run_id.R
assert <- function(cond, msg) if (!isTRUE(cond)) stop(msg, call. = FALSE)

tmp_root <- tempfile()
dir.create(tmp_root)
on.exit(unlink(tmp_root, recursive = TRUE))

rdata_dir <- file.path(tmp_root, "rdata_in")
out_dir   <- file.path(tmp_root, "csv_out")
dir.create(rdata_dir)
file.copy("scratch/fixtures/results_dataset_1_mini.Rdata",
          file.path(rdata_dir, "results_dataset_1.Rdata"))

manifest_path <- file.path(tmp_root, "manifest.csv")
writeLines(c("dataset_id,description", "DS01,test_series"), manifest_path)

# New interface under test: Rscript rdata_to_csv.R <rdata_dir> <out_dir> <manifest_csv>
status <- system2("Rscript", c("src/stage0/rdata_to_csv.R", shQuote(rdata_dir), shQuote(out_dir), shQuote(manifest_path)),
                   stdout = TRUE, stderr = TRUE)
out_files <- list.files(out_dir, pattern = "\\.csv$")
assert(length(out_files) == 1,
       paste("expected exactly one output CSV from the 1-dataset fixture, got:",
             paste(out_files, collapse = ", "), "-- script output:", paste(status, collapse = "\n")))
assert(grepl("^DS01_", out_files[1]), "output filename must still use the manifest dataset_id/slug convention")

rows <- read.csv(file.path(out_dir, out_files[1]))
assert(all(c("dataset_id", "dataset_name", "workflow", "iteration", "prec", "rec", "F1") %in% names(rows)),
       "column contract must be unchanged by the interface change")

cat("All rdata_to_csv run-id/argv tests passed.\n")
