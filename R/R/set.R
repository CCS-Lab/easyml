#' TO BE EDITED.
#' 
#' TO BE EDITED.
#'
#' @return TO BE EDITED.
#' @export
set_coefficients_boolean <- function(algorithm) {
  # Add random_forest once feature importances are implemented?
  algorithms <- c("glmnet")
  boolean <- algorithm %in% algorithms
  boolean
}

#' TO BE EDITED.
#' 
#' TO BE EDITED.
#'
#' @return TO BE EDITED.
#' @export
set_predictions_boolean <- function(algorithm) {
  algorithms <- c("glmnet", "random_forest", "support_vector_machine")
  boolean <- algorithm %in% algorithms
  boolean
}

#' TO BE EDITED.
#' 
#' TO BE EDITED.
#'
#' @return TO BE EDITED.
#' @export
set_metrics_boolean <- function(algorithm) {
  algorithms <- c("glmnet", "random_forest", "support_vector_machine")
  boolean <- algorithm %in% algorithms
  boolean
}

#' TO BE EDITED.
#' 
#' TO BE EDITED.
#'
#' @return TO BE EDITED.
#' @export
set_parallel <- function(n_core) {
  if (n_core == 1) {
    parallel <- FALSE
  } else if (n_core > 1) {
    parallel <- TRUE
  } else {
    stop("Value error.")
  }
  parallel
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
set_looper <- function(progress_bar = FALSE, n_core = 1) {
  # Identify if parallel or not
  parallel <- set_parallel(n_core)
  
  # Set cores
  if (parallel) {
    n_core <- reduce_cores(n_core)
    set_cores(n_core)
  }
  
  # Decide which looper to use
  looper <- set_looper_(progress_bar, parallel)
  looper
}

#' TO BE EDITED.
#' 
#' TO BE EDITED.
#'
#' @return TO BE EDITED.
#' @export
set_looper_ <- function(progress_bar = FALSE, parallel = FALSE) {
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
                             exclude_variables = NULL, preprocess = NULL, 
                             categorical_variables = NULL) {
  column_names <- setdiff(column_names, dependent_variable)
  column_names <- setdiff(column_names, exclude_variables)
  
  if (!is.null(categorical_variables) && !is.null(preprocess)) {
    if (identical(preprocess, preprocess_scaler)) {
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
set_resample <- function(resample = NULL, family = NULL) {
  if (is.null(resample)) {
    if (family == "gaussian") {
      resample <- resample_simple_train_test_split
    } else if (family == "binomial") {
      resample <- resample_stratified_train_test_split
    }
  }
  resample
}

#' TO BE EDITED.
#' 
#' TO BE EDITED.
#'
#' @return TO BE EDITED.
#' @export
set_preprocess <- function(preprocess = NULL) {
  if (is.null(preprocess)) {
    preprocess <- preprocess_identity
  }
  preprocess
}

#' TO BE EDITED.
#' 
#' TO BE EDITED.
#'
#' @return TO BE EDITED.
#' @export
set_measure <- function(measure = NULL, algorithm, family) {
  if (is.null(measure)) {
    if (family == "gaussian") {
      if (algorithm == "glmnet") {
        measure <- measure_r2_score
      } else {
        measure <- measure_mean_squared_error
      }
    } else if (family == "binomial") {
      measure <- measure_area_under_curve
    }
  }

  measure
}

#' TO BE EDITED.
#' 
#' TO BE EDITED.
#'
#' @return TO BE EDITED.
#' @export
set_dependent_variable <- function(.data, dependent_variable) {
  y <- as.vector(.data[, dependent_variable, drop = TRUE])
  y
}

#' TO BE EDITED.
#' 
#' TO BE EDITED.
#'
#' @return TO BE EDITED.
#' @export
set_independent_variables <- function(.data, dependent_variable) {
  .data <- remove_variables(.data, dependent_variable)
  .data
}

#' TO BE EDITED.
#' 
#' TO BE EDITED.
#'
#' @return TO BE EDITED.
#' @export
set_fit_model <- function(algorithm, family) {
  if (algorithm == "glmnet") {
    if (family == "gaussian") {
      fit_model <- glmnet_fit_model_gaussian
    } else if (family == "binomial") {
      fit_model <- glmnet_fit_model_binomial
    }
  } else if (algorithm == "random_forest") {
    if (family == "gaussian") {
      fit_model <- random_forest_fit_model_gaussian
    } else if (family == "binomial") {
      fit_model <- random_forest_fit_model_binomial
    }
  } else if (algorithm == "support_vector_machine") {
    if (family == "gaussian") {
      fit_model <- support_vector_machine_fit_model_gaussian
    } else if (family == "binomial") {
      fit_model <-support_vector_machine_fit_model_binomial
    }
  }

  fit_model
}

#' TO BE EDITED.
#' 
#' TO BE EDITED.
#'
#' @return TO BE EDITED.
#' @export
set_extract_coefficients <- function(algorithm, family) {
  extract_coefficients <- NULL
  if (algorithm == "glmnet") {
    extract_coefficients <- glmnet_extract_coefficients
  }
  
  extract_coefficients
}

#' TO BE EDITED.
#' 
#' TO BE EDITED.
#'
#' @return TO BE EDITED.
#' @export
set_predict_model <- function(algorithm, family) {
  if (algorithm == "glmnet") {
    predict_model <- glmnet_predict_model
  } else if (algorithm == "random_forest") {
    predict_model <- random_forest_predict_model
  } else if (algorithm == "support_vector_machine") {
    predict_model <- support_vector_machine_predict_model
  }
  
  predict_model
}

#' TO BE EDITED.
#' 
#' TO BE EDITED.
#'
#' @return TO BE EDITED.
#' @export
set_plot_predictions <- function(algorithm, family) {
  plot_predictions <- NULL
  if (family == "gaussian") {
    plot_predictions <- plot_predictions_gaussian
  } else if (family == "binomial") {
    plot_predictions <- plot_predictions_binomial
  }
  
  if (is.null(plot_predictions)) 
    stop("Value error!")
  
  plot_predictions
}

#' TO BE EDITED.
#' 
#' TO BE EDITED.
#'
#' @return TO BE EDITED.
#' @export
set_plot_metrics <- function(measure) {
  plot_metrics <- NULL
  if (identical(measure, measure_r2_score)) {
    plot_metrics <- plot_metrics_gaussian_r2_score
  } else if (identical(measure, measure_mean_squared_error)) {
    plot_metrics <- plot_metrics_gaussian_mean_squared_error
  } else if (identical(measure, measure_area_under_curve)) {
    plot_metrics <- plot_metrics_binomial_area_under_curve
  }
  
  if (is.null(plot_metrics)) 
    stop("Value error!")
  
  plot_metrics
}
