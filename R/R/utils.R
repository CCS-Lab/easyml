#' Reduce number of cores.
#'
#' @param n_core An integer vector of length one; specifies the number of cores to use for this analysis. Currenly only works on Mac OSx and Unix/Linux systems. Defaults to 1L.
#' @param cpu_count An integer vector of length one; specifies the number of cores potentially available to use for this analysis. Currenly only works on Mac OSx and Unix/Linux systems. Defaults to 1L.
#' @return TO BE EDITED.
#' @family utils
#' @export
reduce_cores <- function(n_core, cpu_count = NULL) {
  if (is.null(cpu_count)) {
    cpu_count <- parallel::detectCores()
  }
  n_core <- min(n_core, cpu_count)
  n_core
}

#' Remove variables from a dataset.
#'
#' @param .data A data.frame; the data to be analyzed.
#' @param exclude_variables A character vector; the variables from the data set to exclude. Defaults to NULL.
#' @return TO BE EDITED.
#' @family utils
#' @export
remove_variables <- function(.data = NULL, exclude_variables = NULL) {
  if (!is.null(exclude_variables)) {
    .data[, exclude_variables] <- NULL
  }
  .data
}

#' Compute the matrix of p-value.
#' 
#' See here for source: \url{http://www.sthda.com/english/wiki/visualize-correlation-matrix-using-correlogram}.
#'
#' @param x An object of class "DataFrame" or "matrix".
#' @param confidence_level confidence level for the returned confidence interval. Currently only used for the Pearson product moment correlation coefficient if there are at least 4 complete pairs of observations.
#' @param ... further arguments to be passed to or from \code{cor.test}.
#' @return A list containing three matrices; p_value, lower_bound, and upper bound.
#' @family utils
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

#' Process coefficients.
#'
#' @param coefs The replicated coefficients.
#' @param survival_rate_cutoff A numeric vector of length one; for \code{\link{easy_glmnet}}, specifies the minimal threshold (as a percentage) a coefficient must appear out of n_samples. Defaults to 0.05.
#' @return TO BE EDITED.
#' @family utils
#' @export
process_coefficients <- function(coefs, survival_rate_cutoff = 0.05) {
  coefs <- coefs[, -1]
  column_names <- colnames(coefs)
  coefs <- t(coefs) # TODO - this is a simple hack, need to clean up
  survived <- 1 * (abs(coefs) > 0)
  survival_rate <- apply(survived, 1, sum) / ncol(coefs)
  mask <- 1 * (survival_rate > survival_rate_cutoff)
  coefs_updated <- coefs * mask
  betas <- data.frame(predictor = factor(column_names, levels = column_names))
  betas[, "mean"] <- as.numeric(apply(coefs_updated, 1, mean))
  betas[, "lower_bound"] <- as.numeric(apply(coefs_updated, 1, stats::quantile, probs = 0.025))
  betas[, "upper_bound"] <- as.numeric(apply(coefs_updated, 1, stats::quantile, probs = 0.975))
  betas[, "survival"] <- mask
  betas[, "sig"] <- betas["survival"]
  betas[, "dot_color_1"] <- as.numeric(1 * (betas["mean"] != 0))
  cond1 <- betas[, "dot_color_1"] > 0
  cond2 <- betas[, "sig"] > 0
  betas[, "dot_color_2"] <- (1 * (cond1 & cond2)) + 1
  betas[, "dot_color"] <- factor(betas[, "dot_color_1"] * betas[, "dot_color_2"])
  betas
}
