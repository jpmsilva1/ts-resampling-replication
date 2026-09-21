library(performanceEstimation)

# Run from the experiment root directory (002_ts_resampling_strategies/)
results_dir <- "src/adapted/results/data"
file_pattern <- "results_dataset_[0-9]+\\.Rdata$"
files <- list.files(path = results_dir, pattern = file_pattern, full.names = TRUE)

if (length(files) == 0) {
  stop("No result files found in ", results_dir)
}

# List to hold the individual EstimationResults objects
exp_list <- list()

for (f in files) {
  cat("Loading", f, "...\n")
  # load() will put an object named 'exp' into the environment
  load(f)
  exp_list[[f]] <- exp
}

# Merge all objects into a single ComparisonResults object
cat("Merging results...\n")
# Using do.call to pass the list of objects as arguments to mergeEstimationRes
final_exp <- do.call(mergeEstimationRes, exp_list)

# Save the final merged object so PairedComparisons.R can use it
save(final_exp, file = "results/merged_results.Rdata")
cat("Merged results saved to results/merged_results.Rdata\n")
