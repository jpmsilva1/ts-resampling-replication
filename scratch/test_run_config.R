#!/usr/bin/env Rscript
# TDD seam for the Tier 2 traceability knobs (~/.claude/plans/luminous-noodling-bear.md
# Phase 0). src/adapted/run_config.R is sourced by Exps.R and exposes these functions as
# its public interface -- tested here in isolation, without running the full
# performanceEstimation pipeline. Plain assert()s, matching this project's existing R test
# convention (scratch/test_exps_capture.R), not a framework.
#
# Run: Rscript scratch/test_run_config.R
source("src/adapted/run_config.R")

test_default_run_id_is_data_v3 <- function() {
  # Promoted 2026-09-21 (DIVERGENCE_ANALYSIS.md Sec 11): data_v3 is the RNG-pinned,
  # median-signed replication that matches the paper within +/-1 dataset on all 15
  # Table 3 cells. data_v2 stays on disk as the documented pre-fix predecessor --
  # still reachable via EXP002_RUN_ID=data_v2, just no longer the default.
  Sys.unsetenv("EXP002_RUN_ID")
  assert(resolve_results_dir() == "src/adapted/results/data_v3",
         "default (no env var) must resolve to the promoted data_v3 tree")
}

test_custom_run_id_changes_results_dir <- function() {
  Sys.setenv(EXP002_RUN_ID = "rng_rounding")
  assert(resolve_results_dir() == "src/adapted/results/rng_rounding",
         "EXP002_RUN_ID must redirect RESULTS_DIR, not just be read and ignored")
  Sys.unsetenv("EXP002_RUN_ID")
}

test_sample_kind_default_is_rounding <- function() {
  # Promoted alongside the run_id default: Experiment 1 confirmed the pre-2019
  # Rounding-based sample() is what reproduces the paper's Monte Carlo folds, so a
  # plain `Rscript Exps.R` now pins it by default instead of requiring the override.
  Sys.unsetenv("EXP002_SAMPLE_KIND")
  assert(identical(resolve_sample_kind(), "Rounding"),
         "no override set -> default must now be 'Rounding', the promoted behavior")
}

test_sample_kind_override_is_read <- function() {
  Sys.setenv(EXP002_SAMPLE_KIND = "Rounding")
  assert(identical(resolve_sample_kind(), "Rounding"),
         "EXP002_SAMPLE_KIND=Rounding must be readable as the literal string to pass to RNGkind(sample.kind=)")
  Sys.unsetenv("EXP002_SAMPLE_KIND")
}

test_baseline_params_default_is_null <- function() {
  Sys.unsetenv("EXP002_BASELINE_PARAMS")
  assert(is.null(resolve_baseline_params()),
         "no override file set -> NULL signals 'use OPT_PARAMS/Table 7 for baseline too', today's behavior")
}

test_baseline_params_csv_is_parsed_and_indexed_by_dataset <- function() {
  tmp <- tempfile(fileext = ".csv")
  on.exit(unlink(tmp))
  writeLines(c(
    "dataset,svm_cost,svm_gamma,rpart_minsplit,rpart_cp",
    "1,10,0.001,30,0.1",
    "2,300,0.01,10,0.001"
  ), tmp)
  Sys.setenv(EXP002_BASELINE_PARAMS = tmp)
  bp <- resolve_baseline_params()
  assert(!is.null(bp), "a real CSV path must parse to a non-NULL data.frame")
  row1 <- bp[bp$dataset == 1, ]
  assert(nrow(row1) == 1 && row1$svm_cost == 10 && row1$rpart_cp == 0.1,
         "row lookup by dataset index must return the exact tuned values from the CSV, not a default")
  Sys.unsetenv("EXP002_BASELINE_PARAMS")
}

test_manifest_header_contains_required_keys <- function() {
  tmp_dir <- tempfile()
  dir.create(tmp_dir)
  on.exit(unlink(tmp_dir, recursive = TRUE))
  manifest_file <- file.path(tmp_dir, "RUN_MANIFEST.txt")

  write_manifest_header(manifest_file, run_id = "test_run", total_ds = 24,
                         sample_kind_override = "Rounding",
                         baseline_params_file = "(none)",
                         exps_r_path = "src/adapted/Exps.R")

  assert(file.exists(manifest_file), "header write must create the file")
  txt <- readLines(manifest_file)
  required_keys <- c("run_id:", "started:", "r_version:", "rngkind:",
                      "sample_kind_override:", "baseline_params_file:",
                      "slurm_job_id:", "slurm_nodename:", "slurm_cpus_per_task:",
                      "exps_r_md5:", "total_ds:")
  for (key in required_keys) {
    assert(any(startsWith(txt, key)), sprintf("manifest must record '%s' -- that's the whole point of the traceability file", key))
  }
  assert(any(grepl("^run_id: test_run$", txt)), "run_id value must round-trip exactly")
  assert(any(grepl("^total_ds: 24$", txt)), "total_ds value must round-trip exactly")
}

test_manifest_append_adds_a_line_without_clobbering_header <- function() {
  tmp_dir <- tempfile()
  dir.create(tmp_dir)
  on.exit(unlink(tmp_dir, recursive = TRUE))
  manifest_file <- file.path(tmp_dir, "RUN_MANIFEST.txt")
  write_manifest_header(manifest_file, run_id = "test_run", total_ds = 24,
                         sample_kind_override = "(default)", baseline_params_file = "(none)",
                         exps_r_path = "src/adapted/Exps.R")
  n_before <- length(readLines(manifest_file))
  append_manifest_line(manifest_file, dataset = 1, status = "ok", elapsed_min = 1.5)
  txt <- readLines(manifest_file)
  assert(length(txt) == n_before + 1, "append must add exactly one line, not rewrite the file")
  assert(any(grepl("dataset=1 status=ok elapsed_min=1.5", txt)), "appended line must carry dataset, status and timing")
}

test_manifest_resume_appends_marker_instead_of_overwriting <- function() {
  tmp_dir <- tempfile()
  dir.create(tmp_dir)
  on.exit(unlink(tmp_dir, recursive = TRUE))
  manifest_file <- file.path(tmp_dir, "RUN_MANIFEST.txt")
  write_manifest_header(manifest_file, run_id = "test_run", total_ds = 24,
                         sample_kind_override = "(default)", baseline_params_file = "(none)",
                         exps_r_path = "src/adapted/Exps.R")
  append_manifest_line(manifest_file, dataset = 1, status = "ok", elapsed_min = 1.5)
  n_before <- length(readLines(manifest_file))
  ensure_manifest_header(manifest_file, run_id = "test_run", total_ds = 24,
                          sample_kind_override = "(default)", baseline_params_file = "(none)",
                          exps_r_path = "src/adapted/Exps.R")
  txt <- readLines(manifest_file)
  assert(length(txt) == n_before + 1, "ensure_manifest_header on an existing file must append a resumed marker, not rewrite")
  assert(any(grepl("^resumed run_id=test_run$", txt)), "resume marker must be readable")
  assert(sum(startsWith(txt, "run_id:")) == 1, "header must not be duplicated across resumes")
}

assert <- function(cond, msg) if (!isTRUE(cond)) stop(msg, call. = FALSE)

test_default_run_id_is_data_v3()
test_custom_run_id_changes_results_dir()
test_sample_kind_default_is_rounding()
test_sample_kind_override_is_read()
test_baseline_params_default_is_null()
test_baseline_params_csv_is_parsed_and_indexed_by_dataset()
test_manifest_header_contains_required_keys()
test_manifest_append_adds_a_line_without_clobbering_header()
test_manifest_resume_appends_marker_instead_of_overwriting()
cat("All run_config tests passed.\n")
