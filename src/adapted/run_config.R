# Tier 2 traceability knobs, sourced by Exps.R. Public interface (tested in
# isolation by scratch/test_run_config.R, no performanceEstimation run needed):
# resolve_results_dir(), resolve_sample_kind(), resolve_baseline_params(),
# ensure_manifest_header()/write_manifest_header(), append_manifest_line().
#
# Defaults were promoted 2026-09-21 (DIVERGENCE_ANALYSIS.md Sec 11): data_v3, run
# with RNGkind(sample.kind="Rounding"), matches the paper's Table 3 within +/-1
# dataset on all 15 cells -- a plain `Rscript src/adapted/Exps.R` now reproduces
# THAT experiment by default. data_v2 (the pre-promotion baseline) stays fully
# reachable via EXP002_RUN_ID=data_v2 / EXP002_SAMPLE_KIND=Rejection.

resolve_results_dir <- function() {
  run_id <- Sys.getenv("EXP002_RUN_ID", "data_v3")
  file.path("src/adapted/results", run_id)
}

resolve_sample_kind <- function() {
  v <- Sys.getenv("EXP002_SAMPLE_KIND", "Rounding")
  if (!nzchar(v)) NULL else v
}

resolve_baseline_params <- function() {
  path <- Sys.getenv("EXP002_BASELINE_PARAMS", "")
  if (!nzchar(path)) return(NULL)
  read.csv(path, stringsAsFactors = FALSE)
}

write_manifest_header <- function(manifest_file, run_id, total_ds, sample_kind_override,
                                   baseline_params_file, exps_r_path) {
  md5 <- tryCatch(unname(tools::md5sum(exps_r_path)), error = function(e) "(unavailable)")
  lines <- c(
    sprintf("run_id: %s", run_id),
    sprintf("started: %s", format(Sys.time(), "%Y-%m-%dT%H:%M:%S%z")),
    sprintf("r_version: %s", R.version.string),
    sprintf("rngkind: %s", paste(RNGkind(), collapse = ", ")),
    sprintf("sample_kind_override: %s", sample_kind_override),
    sprintf("baseline_params_file: %s", baseline_params_file),
    sprintf("slurm_job_id: %s", Sys.getenv("SLURM_JOB_ID", "(none)")),
    sprintf("slurm_nodename: %s", Sys.getenv("SLURMD_NODENAME", "(none)")),
    sprintf("slurm_cpus_per_task: %s", Sys.getenv("SLURM_CPUS_PER_TASK", "(none)")),
    sprintf("exps_r_md5: %s", md5),
    sprintf("total_ds: %d", total_ds),
    "---"
  )
  writeLines(lines, manifest_file)
}

# Checkpoint-consistent: if a manifest already exists (a resumed/requeued run,
# same pattern as the `if (file.exists(output_file)) next` .Rdata checkpoint),
# append a marker instead of clobbering the original header.
ensure_manifest_header <- function(manifest_file, run_id, total_ds, sample_kind_override,
                                    baseline_params_file, exps_r_path) {
  if (file.exists(manifest_file)) {
    cat(sprintf("resumed run_id=%s\n", run_id), file = manifest_file, append = TRUE)
  } else {
    write_manifest_header(manifest_file, run_id, total_ds, sample_kind_override,
                           baseline_params_file, exps_r_path)
  }
}

append_manifest_line <- function(manifest_file, dataset, status, elapsed_min = NA, msg = NULL) {
  line <- sprintf("[%s] dataset=%d status=%s elapsed_min=%s%s",
                   format(Sys.time(), "%H:%M:%S"), dataset, status,
                   ifelse(is.na(elapsed_min), "NA", sprintf("%.1f", elapsed_min)),
                   if (!is.null(msg)) sprintf(" msg=%s", msg) else "")
  cat(line, "\n", sep = "", file = manifest_file, append = TRUE)
}
