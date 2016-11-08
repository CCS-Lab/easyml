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
process_data <- function(data, dependent_variable = NULL, exclude_variables = NULL) {
  # Handle dependent variable
  if (!is.null(dependent_variable)) {
    y <- data[, dependent_variable]
    data[, dependent_variable] <- NULL
  } else {
    stop("Value error.")
  }
  
  # Possibly exclude columns
  if (!is.null(dependent_variable)) {
    data[, exclude_variables] <- NULL
  }
  
  # Create X array
  X = data
    
  return(list(X = X, y = y))
}
