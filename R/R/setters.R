#' Set random state.
#' 
#' Sets the random state to a specific seed. Please note this function affects global state.
#'
#' @param random_state An integer vector of length one; specifies the seed to be used for the analysis. Defaults to NULL.
#' @return NULL.
#' @family setters
#' @export
set_random_state <- function(random_state = NULL) {
  if (!is.null(random_state)) {
    set.seed(random_state)
  }
  invisible()
}

#' Set coefficients boolean.
#'
#' @param algorithm A character vector of length one; the algorithm to run on the data. Choices are one of c("glmnet", "random_forest", "support_vector_machine").
#' @return A logical vector of length one; whether coefficients should be replicated for this analyis or not.
#' @family setters
#' @export
set_coefficients_boolean <- function(algorithm) {
  algorithms <- c("glmnet")
  boolean <- algorithm %in% algorithms
  boolean
}

#' Set variable importances boolean.
#'
#' @param algorithm A character vector of length one; the algorithm to run on the data. Choices are one of c("glmnet", "random_forest", "support_vector_machine").
#' @return A logical vector of length one; whether variable importances should be replicated for this analyis or not.
#' @family setters
#' @export
set_variable_importances_boolean <- function(algorithm) {
  algorithms <- c("random_forest")
  boolean <- algorithm %in% algorithms
  boolean
}

#' Set predictions boolean.
#'
#' @param algorithm A character vector of length one; the algorithm to run on the data. Choices are one of c("glmnet", "random_forest", "support_vector_machine").
#' @return A logical vector of length one; whether predictions should be replicated for this analyis or not.
#' @family setters
#' @export
set_predictions_boolean <- function(algorithm) {
  algorithms <- c("glmnet", "random_forest", "support_vector_machine")
  boolean <- algorithm %in% algorithms
  boolean
}

#' Set metrics boolean.
#'
#' @param algorithm A character vector of length one; the algorithm to run on the data. Choices are one of c("glmnet", "random_forest", "support_vector_machine").
#' @return A logical vector of length one; whether metrics should be replicated for this analyis or not.
#' @family setters
#' @export
set_metrics_boolean <- function(algorithm) {
  algorithms <- c("glmnet", "random_forest", "support_vector_machine")
  boolean <- algorithm %in% algorithms
  boolean
}

#' Set parallel.
#' 
#' @param n_core An integer vector of length one; specifies the number of cores to use for this analysis. Currenly only works on Mac OSx and Unix/Linux systems. Defaults to 1L.
#' @return A logical vector of length one; whether analyis should be run in parallel or not.
#' @family setters
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
#' Please note this affects global state and sets the number of cores by running \code{options(mc.cores = n_core)}.
#'
#' @param n_core An integer vector of length one; specifies the number of cores to use for this analysis. Currenly only works on Mac OSx and Unix/Linux systems. Defaults to 1L.
#' @return NULL.
#' @family setters
#' @export
set_cores <- function(n_core) {
  options(mc.cores = n_core)
  invisible()
}

#' Set looper.
#' 
#' This function decides which looper (a functional like lapply) to run. Please note this affects global state and sets the number of cores by ultimately running \code{options(mc.cores = n_core)}.
#'
#' @param progress_bar A logical vector of length one; specifies whether to display a progress bar during calculations. Defaults to TRUE.
#' @param n_core An integer vector of length one; specifies the number of cores to use for this analysis. Currenly only works on Mac OSx and Unix/Linux systems. Defaults to 1L.
#' @return The looper to use depending on progress bar and whether to run in parallel or not.
#' @family setters
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

#' Set looper.
#' 
#' This function decides which looper (a functional like lapply) to run. This function does not affect global state.
#'
#' @param progress_bar A logical vector of length one; specifies whether to display a progress bar during calculations. Defaults to FALSE.
#' @param parallel A logical vector of length one; specifies whether to run calculations in parallel. Defaults to FALSE.
#' @return The looper to use depending on progress bar and whether to run in parallel or not.
#' @family setters
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

#' Set column names.
#' 
#' @param column_names A character vector; the column names of the data for this analysis.
#' @param dependent_variable A character vector of length one; the dependent variable for this analysis.
#' @param preprocess A function; the function for preprocessing the data. Defaults to NULL.
#' @param exclude_variables A character vector; the variables from the data set to exclude. Defaults to NULL.
#' @param categorical_variables A character vector; the variables that are categorical. Defaults to NULL.
#' @return The updated columns, in the correct order for preprocessing.
#' @family setters
#' @export
set_column_names <- function(column_names, dependent_variable, 
                             preprocess = NULL, exclude_variables = NULL, 
                             categorical_variables = NULL) {
  column_names <- setdiff(column_names, dependent_variable)
  column_names <- setdiff(column_names, exclude_variables)
  
  if (!is.null(categorical_variables) && !is.null(preprocess)) {
    if (identical(preprocess, preprocess_scale)) {
      column_names <- setdiff(column_names, categorical_variables)
      column_names <- c(categorical_variables, column_names)
    }
  }
  column_names
}

#' Set categorical variables.
#'
#' @param column_names A character vector; the column names of the data for this analysis.
#' @param categorical_variables A character vector; the variables that are categorical. Defaults to NULL.
#' @return NULL, or if \code{categorical_variables} is not NULL, then a logical vector of length \code{length(column_names} where TRUE represents that column is a categorical variable.
#' @family setters
#' @export
set_categorical_variables <- function(column_names, categorical_variables = NULL) {
  if (!is.null(categorical_variables)) {
    categorical_variables <- column_names %in% categorical_variables
  }
  categorical_variables
}

#' Set dependent variable.
#'
#' @param .data A data.frame; the data to be analyzed.
#' @param dependent_variable A character vector of length one; the dependent variable for this analysis.
#' @return A vector, the dependent variable of the analysis.
#' @family setters
#' @export
set_dependent_variable <- function(.data, dependent_variable) {
  y <- as.vector(.data[, dependent_variable, drop = TRUE])
  y
}

#' Set independent variables.
#'
#' @param .data A data.frame; the data to be analyzed.
#' @param dependent_variable A character vector of length one; the dependent variable for this analysis.
#' @return A data.frame, the independent variables of the analysis.
#' @family setters
#' @export
set_independent_variables <- function(.data, dependent_variable) {
  .data <- remove_variables(.data, dependent_variable)
  .data
}

#' Set resample function.
#' 
#' Sets the function responsible for resampling the data.
#'
#' @param resample A function; the function for resampling the data. Defaults to NULL.
#' @param family A character vector of length one; the type of regression to run on the data. Choices are one of c("gaussian", "binomial"). Defaults to "gaussian".
#' @return A function; the function for resampling the data.
#' @family setters
#' @export
set_resample <- function(resample = NULL, family = NULL) {
  if (is.null(resample)) {
    if (family == "gaussian") {
      resample <- resample_simple_train_test_split
    } else if (family == "binomial") {
      resample <- resample_stratified_class_train_test_split
    }
  }
  resample
}

#' Set preprocess function.
#' 
#' Sets the function responsible for preprocessing the data.
#'
#' @param preprocess A function; the function for preprocessing the data. Defaults to NULL.
#' @param algorithm A character vector of length one; the algorithm to run on the data. Choices are one of c("glmnet", "random_forest", "support_vector_machine").
#' @return A function; the function for preprocessing the data.
#' @family setters
#' @export
set_preprocess <- function(preprocess = NULL, algorithm) {
  if (is.null(preprocess)) {
    if (algorithm == "glmnet") {
      preprocess <- preprocess_scale
    } else if (algorithm == "random_forest") {
      preprocess <- preprocess_identity
    } else if (algorithm == "support_vector_machine") {
      preprocess <- preprocess_scale
    }
  }
  
  preprocess
}

#' Set measure function.
#' 
#' Sets the function responsible for measuring the results.
#'
#' @param measure A function; the function for measuring the results. Defaults to NULL.
#' @param algorithm A character vector of length one; the algorithm to run on the data. Choices are one of c("glmnet", "random_forest", "support_vector_machine").
#' @param family A character vector of length one; the type of regression to run on the data. Choices are one of c("gaussian", "binomial"). Defaults to "gaussian".
#' @return A function; the function for measuring the results.
#' @family setters
#' @export
set_measure <- function(measure = NULL, algorithm, family) {
  if (is.null(measure)) {
    if (family == "gaussian") {
      if (algorithm %in% c("glmnet", "random_forest")) {
        measure <- measure_cor_score
      } else {
        measure <- measure_mean_squared_error
      }
    } else if (family == "binomial") {
      measure <- measure_area_under_curve
    }
  }

  measure
}

#' Set fit model function.
#' 
#' Sets the function responsible for fitting a model to the data.
#'
#' @param algorithm A character vector of length one; the algorithm to run on the data. Choices are one of c("glmnet", "random_forest", "support_vector_machine").
#' @param family A character vector of length one; the type of regression to run on the data. Choices are one of c("gaussian", "binomial"). Defaults to "gaussian".
#' @return A function; the function for fitting a model to the data.
#' @family setters
#' @export
set_fit_model <- function(algorithm, family) {
  fit_model <- NULL
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
  
  if (is.null(fit_model))
    stop("Value error!")

  fit_model
}

#' Set extract coefficients function.
#' 
#' Sets the function responsible for extracting coefficients from a model.
#'
#' @param algorithm A character vector of length one; the algorithm to run on the data. Choices are one of c("glmnet", "random_forest", "support_vector_machine").
#' @param family A character vector of length one; the type of regression to run on the data. Choices are one of c("gaussian", "binomial"). Defaults to "gaussian".
#' @return A function; the function for extracting coefficients from a model.
#' @family setters
#' @export
set_extract_coefficients <- function(algorithm, family) {
  extract_coefficients <- NULL
  if (algorithm == "glmnet") {
    extract_coefficients <- glmnet_extract_coefficients
  }
  
  extract_coefficients
}

#' Set predict model function.
#' 
#' Sets the function responsible for generating predictions from a fitted model.
#'
#' @param algorithm A character vector of length one; the algorithm to run on the data. Choices are one of c("glmnet", "random_forest", "support_vector_machine").
#' @param family A character vector of length one; the type of regression to run on the data. Choices are one of c("gaussian", "binomial"). Defaults to "gaussian".
#' @return A function; the function for generating predictions from a fitted model.
#' @family setters
#' @export
set_predict_model <- function(algorithm, family) {
  predict_model <- NULL
  if (algorithm == "glmnet") {
    predict_model <- glmnet_predict_model
  } else if (algorithm == "random_forest") {
    predict_model <- random_forest_predict_model
  } else if (algorithm == "support_vector_machine") {
    predict_model <- support_vector_machine_predict_model
  }
  
  if (is.null(predict_model)) 
    stop("Value error!")
  
  predict_model
}

#' Set plot predictions function.
#' 
#' Sets the function responsible for plotting the predictions generated from a fitted model.
#' 
#' @param algorithm A character vector of length one; the algorithm to run on the data. Choices are one of c("glmnet", "random_forest", "support_vector_machine").
#' @param family A character vector of length one; the type of regression to run on the data. Choices are one of c("gaussian", "binomial"). Defaults to "gaussian".
#' @return A function; the function for plotting the predictions generated from a fitted model.
#' @family setters
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

#' Set plot metrics function.
#' 
#' Sets the function responsible for plotting the metrics generated from the predictions generated from a fitted model.
#'
#' @param measure A function; the function for measuring the results. Defaults to NULL.
#' @return TA function; the function for plotting the metrics generated from the predictions generated from a fitted model.
#' @family setters
#' @export
set_plot_metrics <- function(measure) {
  plot_metrics <- NULL
  if (identical(measure, measure_r2_score)) {
    plot_metrics <- plot_metrics_gaussian_r2_score
  } else if (identical(measure, measure_mean_squared_error)) {
    plot_metrics <- plot_metrics_gaussian_mean_squared_error
  } else if (identical(measure, measure_area_under_curve)) {
    plot_metrics <- plot_metrics_binomial_area_under_curve
  } else if (identical(measure, measure_cor_score)) {
    plot_metrics <- plot_metrics_gaussian_cor_score
  }
  
  if (is.null(plot_metrics)) 
    stop("Value error!")
  
  plot_metrics
}
