# Precompute BEAST-derived features for every CPC18 (problem, block) pair.
#
# This is a thin adapter around the canonical R implementation shipped by
# Plonsky et al. in the OSF demo (deps/BEAST_GB_demo/). We do not modify
# their files; we only orchestrate them with paths suitable for this repo
# and dump the resulting feature table to data/cpc18_features.csv.
#
# Run once (from the project root):
#   Rscript scripts/precompute_features.R
#
# Output columns (1 row per (problem, block); 5 blocks per problem):
#   Dataset, problem_id, Train_Test, block, Feedback, B_rate,
#   Ha, pHa, La, LotShapeA, LotNumA, Hb, pHb, Lb, LotShapeB, LotNumB,
#   Amb, Corr, diffEV, diffSDs, diffMins, diffMaxs, diffUV, RatioMin,
#   SignMax, pBbet_Unbiased1, pBbet_UnbiasedFB, pBbet_Uniform,
#   pBbet_Sign1, pBbet_SignFB, Dom, diffBEV0, diffBEVfb, diffSignEV,
#   BEASTpred
#
# Determinism: BEAST uses random mental sampling (6000 sims/problem) and
# pBetter uses 10k samples. We fix the RNG seed so re-runs are reproducible
# to bit precision.

set.seed(20260522)

# --- Resolve paths relative to this script ----------------------------
# Rscript exposes the script path via commandArgs(trailingOnly = FALSE).
script_args <- commandArgs(trailingOnly = FALSE)
file_arg <- script_args[grepl("^--file=", script_args)]
if (length(file_arg) == 0) {
  stop("Run this script with Rscript so commandArgs() exposes --file=...")
}
script_path <- normalizePath(sub("^--file=", "", file_arg[1]))
repo_root   <- normalizePath(file.path(dirname(script_path), ".."))

beast_dir  <- file.path(repo_root, "deps", "BEAST_GB_demo")
funcs_file <- file.path(beast_dir, "funcs for feature engineering.R")
data_in    <- file.path(beast_dir, "CPC18_data.csv")
data_out   <- file.path(repo_root, "data", "cpc18_features.csv")

if (!file.exists(funcs_file)) stop("Missing: ", funcs_file)
if (!file.exists(data_in))    stop("Missing: ", data_in)

source(funcs_file)

# --- Load competition data (canonical 210 train / 60 test split) ------
all_data <- read.csv(data_in, header = TRUE, stringsAsFactors = FALSE)
cat("Loaded", nrow(all_data), "problems from", data_in, "\n")
cat("Split:", sum(all_data$Train_Test == "Train"), "train /",
    sum(all_data$Train_Test == "Test"), "test\n")

# BEAST_GB.R skips problems 13 and 20 (duplicate parameter sets per
# Erev et al. 2017). Keep that behaviour for fidelity.
skip_idx <- c(13, 20)

# --- Compute features ------------------------------------------------
all_feats <- NULL
t0 <- Sys.time()

for (prob in seq_len(nrow(all_data))) {
  if (prob %in% skip_idx) next
  row <- all_data[prob, ]

  feats <- get_PF_Features(
    row$Ha, row$pHa, row$La, row$LotShapeA, row$LotNumA,
    row$Hb, row$pHb, row$Lb, row$LotShapeB, row$LotNumB,
    row$Amb, row$Corr,
    nBlocks = 5, nTrials = 25, firstFeedbackBlock = 2
  )

  feats$problem_id <- prob
  feats$Dataset    <- row$Dataset
  feats$Train_Test <- row$Train_Test
  # Attach the per-block B-rate target (B.1..B.5 in the wide CSV).
  for (b in 1:5) {
    feats$B_rate[feats$block == b] <- row[[paste0("B.", b)]]
  }

  if (is.null(all_feats)) {
    all_feats <- feats
  } else {
    all_feats <- rbind(all_feats, feats)
  }

  if (prob %% 10 == 0) {
    dt <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
    cat(sprintf("  %3d / %d  (%.1f s elapsed)\n", prob, nrow(all_data), dt))
  }
}

# Reorder columns: identifiers first, then features.
id_cols <- c("Dataset", "problem_id", "Train_Test", "block", "Feedback", "B_rate")
feat_cols <- setdiff(colnames(all_feats), id_cols)
all_feats <- all_feats[, c(id_cols, feat_cols)]

dir.create(dirname(data_out), showWarnings = FALSE, recursive = TRUE)
write.csv(all_feats, data_out, row.names = FALSE)

dt <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
cat(sprintf("\nWrote %d rows x %d cols to %s in %.1f s\n",
            nrow(all_feats), ncol(all_feats), data_out, dt))
