#' Reduce number of cores.
#' 
#' This function takes the number of cores specified by the user and
#' reduces it to the maximum number of cores supported by the computer.
#'
#' @param n_core An integer vector of length one; specifies the number of cores to use for this analysis. Currently only works on Mac OSx and Unix/Linux systems. Defaults to 1L.
#' @param cpu_count An integer vector of length one; specifies the number of cores potentially available to use for this analysis. Currently only works on Mac OSx and Unix/Linux systems. Defaults to 1L.
#' @return An integer vector of length one; specifies the number of cores to use for this analysis.
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
#' This utility function removes variables from a data.frame.
#'
#' @param .data A data.frame; the data to be analyzed.
#' @param exclude_variables A character vector; the variables from the data set to exclude. Defaults to NULL.
#' @return A data.frame; the data to be analyzed.
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
