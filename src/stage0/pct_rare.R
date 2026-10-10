#!/usr/bin/env Rscript
# Descriptive %Rare per dataset (Table 2 / cmp_pct_rare), computed exactly the
# way Exps.R prepares data: authors' .Rdata -> create.data(.., 10) embed
# (V1..V9 -> V10) -> NA handling (DMwR::knnImputation per paper Sec. 4.1; falls
# back to complete.cases where it errors, DS13/DS23 -- same rule as Exps.R with
# EXP002_NA_METHOD=knn) -> uba::phi.control(method="extremes") on
# the full V10 column -> share with phi >= 0.9. Uses R uba, not a Python port:
# the iblr (Python) boxplot quantiles differ from R's fivenum and moved DS05
# from 3.54% to 1.06%. Exps.R itself refits phi per training fold; this is the
# whole-series descriptive figure, comparable to the paper's Table 1.
#
# Usage: Rscript src/stage0/pct_rare.R   (from the project root)
suppressMessages({library(uba); library(DMwR)})

load("src/original/Data/data_NM_PB_LT_DSAA2016.Rdata")
out <- "Results (Clean)/Results Data/pct_rare_canonical.csv"

rows <- lapply(seq_along(data), function(i) {
  e <- embed(as.numeric(data[[i]]), 10)[, 10:1]
  colnames(e) <- paste0("V", 1:10)
  if (anyNA(e)) {
    imp <- tryCatch(knnImputation(as.data.frame(e)), error = function(err) NULL)
    e <- if (is.null(imp)) e[complete.cases(e), , drop = FALSE] else as.matrix(imp)
  }
  y <- e[, 10]
  ph <- uba::phi.control(y, method = "extremes")
  cp <- matrix(ph$control.pts, ncol = 3, byrow = TRUE)
  ext <- if (cp[1, 2] == 1 && cp[nrow(cp), 2] == 1) "Both" else if (cp[nrow(cp), 2] == 1) "High" else "Low"
  data.frame(dataset_id = sprintf("DS%02d", i), raw_n = length(data[[i]]),
             clean_n = nrow(e), extremes = ext,
             pct_rare = 100 * mean(uba::phi(y, ph) >= 0.9))
})
write.csv(do.call(rbind, rows), out, row.names = FALSE)
cat("wrote", out, "\n")
