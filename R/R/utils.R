#' Compute the matrix of p-value.
#' 
#' See here for source: \url{http://www.sthda.com/english/wiki/visualize-correlation-matrix-using-correlogram}.
#'
#' @param x An object of class "DataFrame" or "matrix".
#' @param confidence_level confidence level for the returned confidence interval. Currently only used for the Pearson product moment correlation coefficient if there are at least 4 complete pairs of observations.
#' @param ... further arguments to be passed to or from \code{cor.test}.
#' @return A list containing three matrices; p_value, lower_bound, and upper bound.
#' @export
correlation_test <- function(x, confidence_level = 0.95, ...) {
  # Initialize matrices
  x <- as.matrix(x)
  n <- ncol(x)
  p_value <- lower_bound <- upper_bound <- matrix(NA, n, n)
  diag(p_value) <- 0
  diag(lower_bound) <- diag(upper_bound) <- 1
  
  # Loop through and test for correlation at some confidence_level
  for (i in 1:(n-1)) {
    for (j in (i+1):n) {
      result <- stats::cor.test(x[, i], x[, j], conf.level = confidence_level, ...)
      p_value[i, j] <- p_value[j, i] <- result$p.value
      lower_bound[i, j] <- lower_bound[j, i] <- result$conf.int[1]
      upper_bound[i, j] <- upper_bound[j, i] <- result$conf.int[2]
    }
  }
  
  # Return a list containing three matrices; p_value, lower_bound, and upper bound.
  list(p_value = p_value, 
       lower_bound = lower_bound, 
       upper_bound = upper_bound)
}

#' TO BE EDITED.
#' 
#' TO BE EDITED.
#'
#' @param coefs TO BE EDITED.
#' @param n_samples TO BE EDITED.
#' @param survival_rate_cutoff TO BE EDITED.
#' @return TO BE EDITED.
#' @export
process_coefficients <- function(coefs, column_names, survival_rate_cutoff = 0.05) {
  coefs <- coefs[-1, ]
  survived <- 1 * (abs(coefs) > 0)
  survival_rate <- apply(survived, 1, sum) / ncol(coefs)
  mask <- 1 * (survival_rate > survival_rate_cutoff)
  coefs_updated <- coefs * mask
  betas <- data.frame(predictor = factor(column_names, levels = column_names))
  betas[, "mean"] <- as.numeric(apply(coefs_updated, 1, mean))
  betas[, "lb"] <- as.numeric(apply(coefs_updated, 1, quantile, probs = 0.025))
  betas[, "ub"] <- as.numeric(apply(coefs_updated, 1, quantile, probs = 0.975))
  betas[, "survival"] <- mask
  betas[, "sig"] <- betas["survival"]
  betas[, "dotColor1"] <- as.numeric(1 * (betas["mean"] != 0))
  cond1 <- betas[, "dotColor1"] > 0
  cond2 <- betas[, "sig"] > 0
  betas[, "dotColor2"] <- (1 * (cond1 & cond2)) + 1
  betas[, "dotColor"] <- factor(betas[, "dotColor1"] * betas[, "dotColor2"])
  betas
}

#' TO BE EDITED.
#' 
#' TO BE EDITED.
#'
#' @return TO BE EDITED.
#' @export
identify_looper <- function(progress_bar = FALSE, n_core = 1) {
  # Reduce number of cores if argument is too high
  if (n_core > parallel::detectCores()) {
    n_core <- parallel::detectCores()
  }
  
  # Identify if parallel or not
  parallel <- FALSE
  if (n_core > 1) {
    parallel <- TRUE
    options(mc.cores = n_core)
  }
  
  # Handle settings
  if (progress_bar & parallel) {
    # Initialize progress bar and run in parallel (optional)
    l <- pbmcapply::pbmclapply
  } else if (parallel) {
    # Run in parallel (optional)
    l <- parallel::mclapply
  } else if (progress_bar) {
    # Initialize progress bar (optional)
    l <- pbapply::pblapply
  } else {
    # Default to base R lapply
    l <- lapply
  }
  l
}
