library(performanceEstimation)

# Load merged results
merged_file <- "results/merged_results.Rdata"
if (!file.exists(merged_file)) {
  stop("Merged results file not found. Please run merge_results.R first.")
}

load(merged_file)  # loads 'final_exp'

tasks <- taskNames(final_exp)
workflows <- workflowNames(final_exp)

cat("==========================================================\n")
cat(sprintf("Loaded merged results: %d tasks x %d workflows\n", length(tasks), length(workflows)))
cat("==========================================================\n\n")

# Extract a vector of F1 scores across 50 Monte Carlo iterations
# Metrics are stored in @iterationsInfo[[i]]$evaluation, NOT @iterationsScores
get_metric_vec <- function(obj, metric = "F1") {
  info <- obj@iterationsInfo
  vals <- sapply(info, function(x) {
    ev <- x$evaluation
    if (!is.null(ev) && metric %in% names(ev)) {
      return(as.numeric(ev[metric]))
    }
    return(NA_real_)
  })
  as.numeric(vals)
}

# Verify the extraction works on the first object
cat("--- Verification: First task/workflow F1 scores (first 6 iterations) ---\n")
sample_vec <- get_metric_vec(final_exp[[tasks[1]]][[workflows[1]]], "F1")
cat("Mean F1 for", tasks[1], "/", workflows[1], ":", round(mean(sample_vec, na.rm=TRUE), 4), "\n")
cat("Sample:", round(head(sample_vec), 6), "\n\n")

# --------------------------------------------------------------------------
# Win/Loss/Tie computation (paired Wilcoxon across 50 MC iterations per task)
# --------------------------------------------------------------------------
compute_WL <- function(res, base_wf, metric = "F1", sig = 0.05) {
  other_wfs <- setdiff(workflowNames(res), base_wf)
  WL <- matrix(0, ncol = 5, nrow = length(other_wfs),
               dimnames = list(other_wfs, c("Win", "sigWin", "Loss", "SigLoss", "Tie")))

  for (t in taskNames(res)) {
    base_vec <- get_metric_vec(res[[t]][[base_wf]], metric)
    # MEDIAN, not mean: the original WLdef() signs each cell from
    # performanceEstimation's DiffMedScores = median(base) - median(wf).
    # The means live on a separate t.test array the paper never reads.
    base_med <- median(base_vec, na.rm = TRUE)

    for (wf in other_wfs) {
      wf_vec  <- get_metric_vec(res[[t]][[wf]], metric)
      wf_med  <- median(wf_vec, na.rm = TRUE)
      diff    <- wf_med - base_med

      p_val <- tryCatch({
        wilcox.test(wf_vec, base_vec, paired = TRUE, exact = FALSE)$p.value
      }, error = function(e) 1.0)
      if (is.na(p_val)) p_val <- 1.0

      if (is.na(diff) || diff == 0) {   # WLdef()'s exact `== 0` Tie branch
        WL[wf, "Tie"] <- WL[wf, "Tie"] + 1
      } else if (diff > 0) {
        WL[wf, "Win"] <- WL[wf, "Win"] + 1
        if (p_val < sig) WL[wf, "sigWin"] <- WL[wf, "sigWin"] + 1
      } else {
        WL[wf, "Loss"] <- WL[wf, "Loss"] + 1
        if (p_val < sig) WL[wf, "SigLoss"] <- WL[wf, "SigLoss"] + 1
      }
    }
  }
  WL
}

# --------------------------------------------------------------------------
# Print Win/Loss/Tie vs each baseline
# --------------------------------------------------------------------------
cat("==========================================================\n")
cat("Win / sigWin / Loss / SigLoss / Tie  (Metric: F1, alpha=0.05)\n")
cat("==========================================================\n")

baselines <- c("mc.lm", "mc.svm", "mc.mars", "mc.rf", "mc.rpart")
for (b in baselines) {
  if (b %in% workflows) {
    cat(sprintf("\n>>> Baseline: %s <<<\n", b))
    print(compute_WL(final_exp, b, metric = "F1", sig = 0.05))
  }
}

# --------------------------------------------------------------------------
# Average F1 per workflow across all datasets — ranked table
# --------------------------------------------------------------------------
cat("\n==========================================================\n")
cat(sprintf("Average F1 Score per Workflow (%d Datasets)\n", length(tasks)))
cat("==========================================================\n")

mean_f1 <- sapply(workflows, function(wf) {
  per_task <- sapply(tasks, function(t) {
    mean(get_metric_vec(final_exp[[t]][[wf]], "F1"), na.rm = TRUE)
  })
  mean(per_task, na.rm = TRUE)
})

rank_df <- data.frame(Workflow  = names(mean_f1),
                      Avg_F1    = round(mean_f1, 4),
                      stringsAsFactors = FALSE)
rank_df <- rank_df[order(-rank_df$Avg_F1), ]
rownames(rank_df) <- NULL
print(head(rank_df, 30))

# Save
results_dir <- "results"
dir.create(results_dir, showWarnings = FALSE, recursive = TRUE)
out_path <- file.path(results_dir, "workflow_f1_rankings.csv")
write.csv(rank_df, out_path, row.names = FALSE)
cat(sprintf("\nRankings saved to %s\n", out_path))
cat("Evaluation Complete!\n")
