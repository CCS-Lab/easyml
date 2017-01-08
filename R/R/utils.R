#' TO BE EDITED.
#' 
#' TO BE EDITED.
#'
#' @return TO BE EDITED.
#' @export
check_args <- function() {
  1L
}

#' TO BE EDITED.
#' 
#' TO BE EDITED.
#'
#' @return TO BE EDITED.
#' @export
identify_parallel <- function(n_core) {
  if (n_core == 1) {
    parallel <- FALSE
  } else if (n_core > 1) {
    parallel <- TRUE
  } else {
    stop("Value error.")
  }
  parallel
}

#' Reduce number of cores.
#' 
#' TO BE EDITED.
#'
#' @return TO BE EDITED.
#' @export
reduce_cores <- function(n_core) {
  if (n_core > parallel::detectCores()) {
    n_core <- parallel::detectCores()
  }
  n_core
}

#' Set cores.
#' 
#' Please note this affects global state.
#'
#' @return TO BE EDITED.
#' @export
set_cores <- function(n_core) {
  options(mc.cores = n_core)
}

#' TO BE EDITED.
#' 
#' Please note this affects global state.
#'
#' @return TO BE EDITED.
#' @export
identify_looper <- function(progress_bar = FALSE, n_core = 1) {
  # Identify if parallel or not
  parallel <- identify_parallel(n_core)
  
  # Set cores
  if (parallel) {
    n_core <- reduce_cores(n_core)
    set_cores(n_core)
  }

  # Decide which looper to use
  looper <- identify_looper_(progress_bar, parallel)
  looper
}

#' TO BE EDITED.
#' 
#' TO BE EDITED.
#'
#' @return TO BE EDITED.
#' @export
identify_looper_ <- function(progress_bar = FALSE, parallel = FALSE) {
  if (progress_bar & parallel) {
    # Initialize progress bar and run in parallel
    looper <- pbmcapply::pbmclapply
  } else if (parallel) {
    # Run in parallel
    looper <- parallel::mclapply
  } else if (progress_bar) {
    # Initialize progress bar
    pbapply::pboptions(char = "=", style = 1)
    looper <- pbapply::pblapply
  } else {
    # Default to base R lapply
    looper <- lapply
  }
  looper
}

#' TO BE EDITED.
#' 
#' TO BE EDITED.
#'
#' @return TO BE EDITED.
#' @export
set_column_names <- function(column_names, dependent_variable, 
                             exclude_variables = NULL, preprocessor = NULL, 
                             categorical_variables = NULL) {
  column_names <- setdiff(column_names, dependent_variable)
  column_names <- setdiff(column_names, exclude_variables)
  
  if (!is.null(categorical_variables) && !is.null(preprocessor)) {
    if (identical(preprocessor, preprocess_scaler)) {
      column_names <- setdiff(column_names, categorical_variables)
      column_names <- c(categorical_variables, column_names)
    }
  }
  column_names
}

#' TO BE EDITED.
#' 
#' TO BE EDITED.
#'
#' @return TO BE EDITED.
#' @export
set_categorical_variables <- function(column_names, categorical_variables = NULL) {
  if (!is.null(categorical_variables)) {
    categorical_variables <- column_names %in% categorical_variables
  }
  categorical_variables
}

#' TO BE EDITED.
#' 
#' TO BE EDITED.
#'
#' @return TO BE EDITED.
#' @export
set_preprocessor <- function(preprocessor = NULL) {
  if (is.null(preprocessor)) {
    preprocessor <- preprocess_identity
  }
  preprocessor
}

#' TO BE EDITED.
#' 
#' Please note this affects global state.
#'
#' @return TO BE EDITED.
#' @export
set_random_state <- function(random_state = NULL) {
  if (!is.null(random_state)) {
    set.seed(random_state)
  }
  invisible()
}

#' TO BE EDITED.
#' 
#' TO BE EDITED.
#'
#' @return TO BE EDITED.
#' @export
set_sampler <- function(sampler = NULL, family = NULL) {
  if (is.null(sampler)) {
    if (is.null(family)) {
      stop("Value error.")
    } else if (family == "gaussian") {
      sampler <- resample_simple_train_test_split
    } else if (family == "binomial") {
      sampler <- resample_stratified_train_test_split
    } else {
      stop("Value error.")
    }
  }
  sampler
}
#' TO BE EDITED.
#' 
#' TO BE EDITED.
#'
#' @return TO BE EDITED.
#' @export
isolate_dependent_variable <- function(.data = NULL, dependent_variable = NULL) {
  y <- as.vector(.data[, dependent_variable, drop = TRUE])
  y
}

#' TO BE EDITED.
#' 
#' TO BE EDITED.
#'
#' @return TO BE EDITED.
#' @export
isolate_independent_variables <- function(.data = NULL, dependent_variable = NULL) {
  .data <- remove_variables(.data, dependent_variable)
  .data
}

#' TO BE EDITED.
#' 
#' TO BE EDITED.
#'
#' @return TO BE EDITED.
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
