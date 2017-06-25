#' Process coefficients.
#' 
#' Takes the coefficients returned by the generate_coefficients function 
#' and prepares the coefficients for plotting.
#'
#' @param coefficients A data.frame, the replicated coefficients.
#' @param survival_rate_cutoff A numeric vector of length one; for \code{\link{easy_glmnet}}, specifies the minimal threshold (as a percentage) a coefficient must appear out of n_samples. Defaults to 0.05.
#' @return A data.frame; the replicated coefficients processed for easy plotting.
#' @family utils
#' @export
process_coefficients <- function(coefficients, survival_rate_cutoff = 0.05) {
  coefficients <- coefficients[, -1]
  column_names <- colnames(coefficients)
  coefficients <- t(coefficients) # TODO - this is a simple hack, need to clean up
  survived <- 1 * (abs(coefficients) > 0)
  survival_rate <- apply(survived, 1, sum) / ncol(coefficients)
  mask <- 1 * (survival_rate > survival_rate_cutoff)
  coefficients_updated <- coefficients * mask
  coefficients_processed <- data.frame(predictor = factor(column_names, levels = column_names))
  coefficients_processed[, "mean"] <- as.numeric(apply(coefficients_updated, 1, mean))
  coefficients_processed[, "sd"] <- as.numeric(apply(coefficients_updated, 1, stats::sd))
  coefficients_processed[, "lower_bound"] <- as.numeric(apply(coefficients_updated, 1, stats::quantile, probs = 0.025))
  coefficients_processed[, "upper_bound"] <- as.numeric(apply(coefficients_updated, 1, stats::quantile, probs = 0.975))
  coefficients_processed[, "survival"] <- mask
  coefficients_processed[, "sig"] <- coefficients_processed["survival"]
  coefficients_processed[, "dot_color_1"] <- as.numeric(1 * (coefficients_processed["mean"] != 0))
  cond1 <- coefficients_processed[, "dot_color_1"] > 0
  cond2 <- coefficients_processed[, "sig"] > 0
  coefficients_processed[, "dot_color_2"] <- (1 * (cond1 & cond2)) + 1
  coefficients_processed[, "dot_color"] <- factor(coefficients_processed[, "dot_color_1"] * coefficients_processed[, "dot_color_2"])
  coefficients_processed
}

#' Process variable importances.
#' 
#' Takes the variable importances returned by the 
#' generate_variable_importances function and prepares 
#' the variable importances for plotting.
#'
#' @param variable_importances A data.frame, the replicated coefficients.
#' @return A data.frame; the replicated variable importances processed for easy plotting.
#' @family utils
#' @export
process_variable_importances <- function(variable_importances) {
  column_names <- colnames(variable_importances)
  means <- vapply(variable_importances, mean, numeric(1))
  sds <- vapply(variable_importances, stats::sd, numeric(1))
  variable_importances_processed <- data.frame(predictor = column_names, 
                                               stringsAsFactors = FALSE)
  variable_importances_processed[, "mean"] <- means
  variable_importances_processed[, "sd"] <- sds
  variable_importances_processed[, "lower_bound"] <- means - sds
  variable_importances_processed[, "upper_bound"] <- means + sds
  rownames(variable_importances_processed) <- NULL
  variable_importances_processed
}
