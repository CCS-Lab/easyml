#' Fit model.
#' 
#' @param X input matrix, of dimension nobs x nvars; each row is an observation vector. Can be in sparse matrix format (inherit from class "sparseMatrix" as in package Matrix; not yet available for family="cox")
#' @param y response variable. Quantitative for family="gaussian", or family="poisson" (non-negative counts). For family="binomial" should be either a factor with two levels, or a two-column matrix of counts or proportions (the second column is treated as the target class; for a factor, the last level in alphabetical order is the target class). For family="multinomial", can be a nc>=2 level factor, or a matrix with nc columns of counts or proportions. For either "binomial" or "multinomial", if y is presented as a vector, it will be coerced into a factor. For family="cox", y should be a two-column matrix with columns named 'time' and 'status'. The latter is a binary variable, with '1' indicating death, and '0' indicating right censored. The function Surv() in package survival produces such a matrix. For family="mgaussian", y is a matrix of quantitative responses.
#' @param ... Arguments to be passed to \code{\link[glmnet]{glmnet}}. See that function's documentation for more details.
#' @return A list, the model and the cross validated model.
#' @export
fit_model <- function(X, y, ...) {
  UseMethod("fit_model")
}

fit_model.default <- function(X, y, ...) {
  msg <- paste0("Error: fit_model not implemented for class ", class(foo))
  stop(msg)
}

extract_coefficients <- function(foo) {
  UseMethod("extract_coefficients")
}

extract_coefficients.default <- function(foo) {
  msg <- paste0("Error: extract_coefficients not implemented for class ", class(foo))
  stop(msg)
}

extract_variable_importances <- function(foo) {
  UseMethod("extract_variable_importances")
}

extract_variable_importances.default <- function(foo) {
  msg <- paste0("Error: extract_variable_importances not implemented for class ", class(foo))
  stop(msg)
}

predict_model <- function(foo) {
  UseMethod("predict_model")
}

predict_model.default <- function(foo) {
  msg <- paste0("Error: predict_model not implemented for class ", class(foo))
  stop(msg)
}


#' The core reciple of easyml.
#' 
#' This recipe is the workhorse behind all of the easy_* functions. 
#'
#' @param .data A data.frame; the data to be analyzed.
#' @param dependent_variable A character vector of length one; the dependent variable for this analysis.
#' @param algorithm A character vector of length one; the algorithm to run on the data. Choices are one of c("glmnet", "random_forest", "support_vector_machine").
#' @param family A character vector of length one; the type of regression to run on the data. Choices are one of c("gaussian", "binomial"). Defaults to "gaussian".
#' @param resample A function; the function for resampling the data. Defaults to NULL.
#' @param preprocess A function; the function for preprocessing the data. Defaults to NULL.
#' @param measure A function; the function for measuring the results. Defaults to NULL.
#' @param exclude_variables A character vector; the variables from the data set to exclude. Defaults to NULL.
#' @param categorical_variables A character vector; the variables that are categorical. Defaults to NULL.
#' @param train_size A numeric vector of length one; specifies what proportion of the data should be used for the training data set. Defaults to 0.667.
#' @param foldid A vector with length equal to \code{length(y)} which identifies cases belonging to the same fold. 
#' @param survival_rate_cutoff A numeric vector of length one; for \code{\link{easy_glmnet}}, specifies the minimal threshold (as a percentage) a coefficient must appear out of n_samples. Defaults to 0.05.
#' @param n_samples An integer vector of length one; specifies the number of times the coefficients and predictions should be replicated. Defaults to 1000. 
#' @param n_divisions An integer vector of length one; specifies the number of times the data should be divided when replicating the error metrics. Defaults to 1000.
#' @param n_iterations An integer vector of length one; during each division, specifies the number of times the predictions should be replicated. Defaults to 10.
#' @param random_state An integer vector of length one; specifies the seed to be used for the analysis. Defaults to NULL.
#' @param progress_bar A logical vector of length one; specifies whether to display a progress bar during calculations. Defaults to TRUE.
#' @param n_core An integer vector of length one; specifies the number of cores to use for this analysis. Currenly only works on Mac OSx and Unix/Linux systems. Defaults to 1.
#' @param ... The arguments to be passed to the algorithm specified.
#' @return A list with the following values:
#' \describe{
#' \item{resample}{A function; the function for resampling the data.}
#' \item{preprocess}{A function; the function for preprocessing the data.}
#' \item{measure}{A function; the function for measuring the results.}
#' \item{fit_model}{A function; the function for fitting the model to the data.}
#' \item{extract_coefficients}{A function; the function for extracting coefficients from the model.}
#' \item{predict_model}{A function; the function for generating predictions on new data from the model.}
#' \item{plot_predictions}{A function; the function for plotting predictions generated by the model.}
#' \item{plot_metrics}{A function; the function for plotting metrics generated by scoring the model.}
#' \item{data}{A data.frame; the original data.}
#' \item{X}{A data.frame; the full dataset to be used for modeling.}
#' \item{y}{A vector; the full response variable to be used for modeling.}
#' \item{X_train}{A data.frame; the train dataset to be used for modeling.}
#' \item{X_test}{A data.frame; the test dataset to be used for modeling.}
#' \item{y_train}{A vector; the train response variable to be used for modeling.}
#' \item{y_test}{A vector; the test response variable to be used for modeling.}
#' \item{coefficients}{A (n_variables, n_samples) matrix; the replicated coefficients.}
#' \item{coefficients_processed}{A data.frame; the coefficients after being processed.}
#' \item{plot_coefficients_processed}{A ggplot object; the plot of the processed coefficients.}
#' \item{predictions_train}{A (nrow(X_train), n_samples) matrix; the train predictions.}
#' \item{predictions_test}{A (nrow(X_test), n_samples) matrix; the test predictions.}
#' \item{predictions_train_mean}{A vector; the mean train predictions.}
#' \item{predictions_test_mean}{A vector; the mean test predictions.}
#' \item{plot_predictions_train_mean}{A ggplot object; the plot of the mean train predictions.}
#' \item{plot_predictions_test_mean}{A ggplot object; the plot of the mean test predictions.}
#' \item{metrics_train_mean}{A vector of length n_divisions; the mean train metrics.}
#' \item{metrics_test_mean}{A vector of length n_divisions; the mean test metrics.}
#' \item{plot_metrics_train_mean}{A ggplot object; the plot of the mean train metrics.}
#' \item{plot_metrics_test_mean}{A ggplot object; the plot of the mean test metrics.}
#' }
#' @family recipes
#' @export
easy_analysis <- function(.data, dependent_variable, algorithm, 
                          family = "gaussian", resample = NULL, 
                          preprocess = NULL, measure = NULL, 
                          exclude_variables = NULL, 
                          categorical_variables = NULL, train_size = 0.667, 
                          foldid = NULL, survival_rate_cutoff = 0.05, 
                          n_samples = 1000, n_divisions = 1000, n_iterations = 10, 
                          random_state = NULL, progress_bar = TRUE, n_core = 1, 
                          coefficients = NULL, variable_importances = NULL, 
                          predictions = NULL, metrics = NULL, ...) {
  # Check positional arguments for validity
  check_arguments(.data, dependent_variable, algorithm)
  
  # Instantiate object
  object <- list()
  
  # Capture data
  object[["data"]] <- .data
  
  # Capture dependent variable
  object[["dependent_variable"]] <- dependent_variable
  
  # Capture algorithm
  object[["algorithm"]] <- algorithm
  
  # Capture class
  .class <- paste0("easy_", algorithm)
  object[["class"]] <- .class
  
  # Capture family
  object[["family"]] <- family
  
  # Capture resample
  object[["resample"]] <- resample
  
  # Capture preprocess
  object[["preprocess"]] <- preprocess
  
  # Capture measure
  object[["measure"]] <- measure
  
  # Capture exclude variables
  object[["exclude_variables"]] <- exclude_variables
  
  # Capture categorical variables
  object[["categorical_variables"]] <- categorical_variables
  
  # # Capture 
  # object[[""]] <- ""
  # 
  # # Capture 
  # object[[""]] <- ""
  # 
  # # Capture 
  # object[[""]] <- ""
  # 
  # # Capture 
  # object[[""]] <- ""
  # 
  # # Capture 
  # object[[""]] <- ""
  # 
  # # Capture 
  # object[[""]] <- ""
  # 
  # # Capture 
  # object[[""]] <- ""
  # 
  # # Capture 
  # object[[""]] <- ""
  # 
  # # Capture 
  # object[[""]] <- ""
  
  # Capture random state
  set_random_state(random_state)
  object[["random_state"]] <- random_state
  
  # Capture progress bar
  object[["progress_bar"]] <- progress_bar
  
  # Capture number of cores
  object[["n_core"]] <- n_core
  
  # Capture coefficients
  object[["coefficients"]] <- coefficients
  
  # Capture variable importances
  object[["variable_importances"]] <- variable_importances
  
  # Capture predictions
  object[["predictions"]] <- predictions
  
  # Capture metrics
  object[["metrics"]] <- metrics
  
  # Capture
  object[[""]] <- ""

  # Capture keyword arguments
  kwargs <- list(...)
  object[["kwargs"]] <- kwargs
  
  # Set column names
  column_names <- set_column_names(colnames(.data), dependent_variable, 
                                   preprocess = preprocess, 
                                   exclude_variables = exclude_variables, 
                                   categorical_variables = categorical_variables)
  
  # Set categorical variables
  categorical_variables <- set_categorical_variables(column_names, categorical_variables)
  
  # Remove variables
  .data <- remove_variables(.data, exclude_variables)
  object[["data"]] <- .data
  
  # Set dependent variable
  y <- set_dependent_variable(.data, dependent_variable)
  object[["y"]] <- y
  
  # Set independent variables
  X <- set_independent_variables(.data, dependent_variable)
  X <- X[, column_names]
  object[["X"]] <- X
  
  # Set class of the object
  object <- structure(object, class = .class)
  
  # Assess if coefficients should be replicated for this algorithm
  if (coefficients) {
    # Replicate coefficients
    coefs <- replicate_coefficients(object)
    object[["coefficients"]] <- coefs
    
    # Process coefficients
    coefs_processed <- process_coefficients(coefs, survival_rate_cutoff)
    object[["coefficients_processed"]] <- coefs_processed
    
    # Save coefficients plots
    g <- plot_coefficients_processed(coefs_processed)
    object[["plot_coefficients_processed"]] <- g
  }
  
  # Assess if variable importances should be replicated for this algorithm
  if (variable_importances) {
    # Replicate variable importances
    variable_imps <- replicate_variable_importances(object)
    object[["variable_importances"]] <- variable_imps
    
    # Process variable_importances
    variable_imps_processed <- process_variable_importances(variable_imps)
    object[["variable_importances_processed"]] <- variable_imps_processed
    
    # Save variable importances plot
    g <- plot_variable_importances_processed(variable_imps_processed)
    object[["plot_variable_importances_processed"]] <- g
  }
  
  # Assess if predictions should be replicated for this algorithm
  if (prediction) {
    # Resample data
    split_data <- resample(X, y, train_size = train_size, foldid = foldid)
    object <- c(object, split_data)
    X_train <- split_data[["X_train"]]
    X_test <- split_data[["X_test"]]
    y_train <- split_data[["y_train"]]
    y_test <- split_data[["y_test"]]
    
    # Replicate predictions
    predictions <- replicate_predictions(object)
    object <- c(object, predictions)
    predictions_train <- predictions[["predictions_train"]]
    predictions_test <- predictions[["predictions_test"]]
    
    # Process predictions
    predictions_train_mean <- apply(predictions_train, 1, mean)
    predictions_test_mean <- apply(predictions_test, 1, mean)
    object[["predictions_train_mean"]] <- predictions_train_mean
    object[["predictions_test_mean"]] <- predictions_test_mean
    
    # Set plot_predictions function
    plot_predictions <- set_plot_predictions(algorithm, family)
    object[["plot_predictions"]] <- plot_predictions
    
    # Save predictions plots
    plot_predictions_train_mean <- 
      plot_predictions(y_train, predictions_train_mean) + 
      ggplot2::labs(subtitle = "Train Predictions")
    object[["plot_predictions_train_mean"]] <- plot_predictions_train_mean
    plot_predictions_test_mean <- 
      plot_predictions(y_test, predictions_test_mean) + 
      ggplot2::labs(subtitle = "Test Predictions")
    object[["plot_predictions_test_mean"]] <- plot_predictions_test_mean
  }
  
  # Assess if metrics should be replicated for this algorithm
  if (metrics_boolean) {
    # Replicate metrics
    metrics <- replicate_metrics(object)
    object <- c(object, metrics)
    metrics_train_mean <- metrics[["metrics_train_mean"]]
    metrics_test_mean <- metrics[["metrics_test_mean"]]
    
    # Set plot_metrics function
    plot_metrics <- set_plot_metrics(measure)
    object[["plot_metrics"]] <- plot_metrics
    
    # Save metrics plots
    plot_metrics_train_mean <- 
      plot_metrics(metrics_train_mean) + 
      ggplot2::labs(subtitle = "Train Metrics")
    object[["plot_metrics_train_mean"]] <- plot_metrics_train_mean
    plot_metrics_test_mean <- 
      plot_metrics(metrics_test_mean) + 
      ggplot2::labs(subtitle = "Test Metrics")
    object[["plot_metrics_test_mean"]] <- plot_metrics_test_mean
  }
  
  # Return object
  object
}

# # Set coefficients boolean
# coefficients_boolean <- set_coefficients_boolean(algorithm)
# 
# # Set variable importances boolean
# variable_importances_boolean <- set_variable_importances_boolean(algorithm)
# 
# # Set predictions boolean
# predictions_boolean <- set_predictions_boolean(algorithm)
# 
# # Set metrics boolean
# metrics_boolean <- set_metrics_boolean(algorithm)
