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
#' @param data TO BE EDITED.
#' @param dependent_variable TO BE EDITED.
#' @param exclude_variables TO BE EDITED.
#' @return TO BE EDITED.
#' @export
process_data <- function(.df, dependent_variable = NULL, exclude_variables = NULL) {
  # Handle dependent variable
  if (!is.null(dependent_variable)) {
    y <- .df[, dependent_variable]
    .df[, dependent_variable] <- NULL
  } else {
    stop("Value error.")
  }
  
  # Possibly exclude columns
  if (!is.null(dependent_variable)) {
    .df[, exclude_variables] <- NULL
  }
  
  # Create X array
  X = .df
    
  list(X = X, y = y)
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
  survived <- 1 * (abs(coefs) > 0)
  survival_rate <- apply(survived, 1, sum) / 1000
  mask <- 1 * (survival_rate > survival_rate_cutoff)
  coefs_updated <- coefs * mask
  betas <- data.frame(predictor = column_names, stringsAsFactors = FALSE)
  betas['mean'] <- as.numeric(apply(coefs_updated, 1, mean))
  betas['lb'] <- as.numeric(apply(coefs_updated, 1, quantile, probs = 0.025))
  betas['ub'] <- as.numeric(apply(coefs_updated, 1, quantile, probs = 0.975))
  betas['survival'] <- mask
  betas['sig'] <- betas['survival']
  betas['dotColor1'] <- 1 * (betas['mean'] != 0)
  cond1 <- betas['dotColor1'] > 0
  cond2 <- betas['sig'] > 0
  betas['dotColor2'] <- (1 * (cond1 & cond2)) + 1
  betas['dotColor'] <- betas['dotColor1'] * betas['dotColor2']
  betas
}
