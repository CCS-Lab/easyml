#' Fit model.
#' 
#' @param object A list of class \code{easy_*}, where * is the name of the algorithm.
#' @return A list of class \code{easy_*}, where * is the name of the algorithm.
#' @export
fit_model <- function(object) {
  UseMethod("fit_model")
}

#' Extract coefficients.
#' 
#' @param object A list of class \code{easy_*}, where * is the name of the algorithm.
#' @return A data.frame, the generated coefficients.
#' @export
extract_coefficients <- function(object) {
  UseMethod("extract_coefficients")
}

#' Extract variable importances.
#' 
#' @param object A list of class \code{easy_*}, where * is the name of the algorithm.
#' @return A data.frame, the generated random forest variable importance scores.
#' @export
extract_variable_importances <- function(object) {
  UseMethod("extract_variable_importances")
}

#' Predict model.
#' 
#' @param object A list of class \code{easy_*}, where * is the name of the algorithm.
#' @param newx A data.frame, the new data to use for predictions.
#' @return A vector, the predicted values using the new data.
#' @export
predict_model <- function(object, newx = NULL) {
  UseMethod("predict_model")
}

#' The core recipe of easyml.
#' 
#' This recipe is the workhorse behind all of the easy_* functions. 
#'
#' @param .data A data.frame; the data to be analyzed.
#' @param dependent_variable A character vector of length one; the dependent variable for this analysis.
#' @param algorithm A character vector of length one; the algorithm to run on the data. Choices are currently one of c("deep_neural_network", "glinternet", "glmnet", "neural_network", "random_forest", "support_vector_machine").
#' @param family A character vector of length one; the type of regression to run on the data. Choices are one of c("gaussian", "binomial"). Defaults to "gaussian".
#' @param resample A function; the function for resampling the data. Defaults to NULL.
#' @param preprocess A function; the function for preprocessing the data. Defaults to NULL.
#' @param measure A function; the function for measuring the results. Defaults to NULL.
#' @param exclude_variables A character vector; the variables from the data set to exclude. Defaults to NULL.
#' @param categorical_variables A character vector; the variables that are categorical. Defaults to NULL.
#' @param train_size A numeric vector of length one; specifies what proportion of the data should be used for the training data set. Defaults to 0.667.
#' @param foldid A vector with length equal to \code{length(y)} which identifies cases belonging to the same fold. 
#' @param survival_rate_cutoff A numeric vector of length one; for \code{\link{easy_glmnet}}, specifies the minimal threshold (as a percentage) a coefficient must appear out of n_samples. Defaults to 0.05.
#' @param n_samples An integer vector of length one; specifies the number of times the coefficients and predictions should be generated. Defaults to 1000. 
#' @param n_divisions An integer vector of length one; specifies the number of times the data should be divided when replicating the error metrics. Defaults to 1000.
#' @param n_iterations An integer vector of length one; during each division, specifies the number of times the predictions should be generated. Defaults to 10.
#' @param random_state An integer vector of length one; specifies the seed to be used for the analysis. Defaults to NULL.
#' @param progress_bar A logical vector of length one; specifies whether to display a progress bar during calculations. Defaults to TRUE.
#' @param n_core An integer vector of length one; specifies the number of cores to use for this analysis. Currenly only works on Mac OSx and Unix/Linux systems. Defaults to 1.
#' @param coefficients A logical vector of length one; whether or not to generate coefficients for this analysis.
#' @param variable_importances A logical vector of length one; whether or not to generate variable importances for this analysis.
#' @param predictions A logical vector of length one; whether or not to generate predictions for this analysis.
#' @param metrics A logical vector of length one; whether or not to generate metrics for this analysis.
#' @param model_args A list; the arguments to be passed to the algorithm specified.
#' @return A list of class \code{easy_*}, where * is the name of the algorithm.
#' \describe{
#' \item{call}{An object of class \code{call}; the original function call.}
#' \item{data}{A data.frame; the original data.}
#' \item{dependent_variable}{A character vector of length one; the dependent variable for this analysis.}
#' \item{algorithm}{A character vector of length one; the algorithm to run on the data.}
#' \item{class}{A character vector of length one; the class of the object.}
#' \item{family}{A character vector of length one; the type of regression to run on the data. Choices are one of c("gaussian", "binomial"). Defaults to "gaussian".}
#' \item{resample}{A function; the function for resampling the data.}
#' \item{preprocess}{A function; the function for preprocessing the data.}
#' \item{measure}{A function; the function for measuring the results.}
#' \item{exclude_variables}{A character vector; the variables from the data set to exclude.}
#' \item{train_size}{A numeric vector of length one; specifies what proportion of the data should be used for the training data set.}
#' \item{survival_rate_cutoff}{A numeric vector of length one; for \code{\link{easy_glmnet}}, specifies the minimal threshold (as a percentage) a coefficient must appear out of n_samples.}
#' \item{n_samples}{An integer vector of length one; specifies the number of times the coefficients and predictions should be generated.}
#' \item{n_divisions}{An integer vector of length one; specifies the number of times the data should be divided when replicating the error metrics.}
#' \item{n_iterations}{An integer vector of length one; during each division, specifies the number of times the predictions should be generated.}
#' \item{random_state}{An integer vector of length one; specifies the seed to be used for the analysis.}
#' \item{progress_bar}{A logical vector of length one; specifies whether to display a progress bar during calculations.}
#' \item{n_core}{An integer vector of length one; specifies the number of cores to use for this analysis.}
#' \item{generate_coefficients}{A logical vector of length one; whether or not to generate coefficients for this analysis.}
#' \item{generate_variable_importances}{A logical vector of length one; whether or not to generate variable importances for this analysis.}
#' \item{generate_predictions}{A logical vector of length one; whether or not to generate predictions for this analysis.}
#' \item{generate_metrics}{A logical vector of length one; whether or not to generate metrics for this analysis.}
#' \item{model_args}{A list; the arguments to be passed to the algorithm specified.}
#' \item{column_names}{A character vector; the column names.}
#' \item{categorical_variables}{A logical vector; the variables that are categorical.}
#' \item{X}{A data.frame; the full dataset to be used for modeling.}
#' \item{y}{A vector; the full response variable to be used for modeling.}
#' \item{coefficients}{A (n_variables, n_samples) matrix; the generated coefficients.}
#' \item{coefficients_processed}{A data.frame; the coefficients after being processed.}
#' \item{plot_coefficients_processed}{A ggplot object; the plot of the processed coefficients.}
#' \item{X_train}{A data.frame; the train dataset to be used for modeling.}
#' \item{X_test}{A data.frame; the test dataset to be used for modeling.}
#' \item{y_train}{A vector; the train response variable to be used for modeling.}
#' \item{y_test}{A vector; the test response variable to be used for modeling.}
#' \item{predictions_train}{A (nrow(X_train), n_samples) matrix; the train predictions.}
#' \item{predictions_test}{A (nrow(X_test), n_samples) matrix; the test predictions.}
#' \item{predictions_train_mean}{A vector; the mean train predictions.}
#' \item{predictions_test_mean}{A vector; the mean test predictions.}
#' \item{plot_predictions}{A function; the function for plotting predictions generated by the model.}
#' \item{plot_predictions_train_mean}{A ggplot object; the plot of the mean train predictions.}
#' \item{plot_predictions_test_mean}{A ggplot object; the plot of the mean test predictions.}
#' \item{metrics_train_mean}{A vector of length n_divisions; the mean train metrics.}
#' \item{metrics_test_mean}{A vector of length n_divisions; the mean test metrics.}
#' \item{plot_metrics}{A function; the function for plotting metrics generated by scoring the model.}
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
                          n_samples = 1000, n_divisions = 1000, 
                          n_iterations = 10, random_state = NULL, 
                          progress_bar = TRUE, n_core = 1, 
                          coefficients = NULL, variable_importances = NULL, 
                          predictions = NULL, metrics = NULL, 
                          model_args = list()) {
  # Check positional arguments for validity
  check_arguments(.data, dependent_variable, algorithm)
  
  # Instantiate object
  object <- list()
  
  # Capture call
  object[["call"]] <- match.call(call = sys.call(sys.parent(1L)))
  
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
  
  # Set and capture resample
  resample <- set_resample(resample, family)
  object[["resample"]] <- resample
  
  # Set and capture preprocess
  preprocess <- set_preprocess(preprocess, algorithm)
  object[["preprocess"]] <- preprocess
  
  # Set and capture measure
  measure <- set_measure(measure, algorithm, family)
  object[["measure"]] <- measure
  
  # Capture exclude variables
  object[["exclude_variables"]] <- exclude_variables
  
  # Capture train size
  object[["train_size"]] <- train_size
  
  # Capture foldid
  object[["foldid"]] <- foldid

  # Capture survival rate cutoff
  object[["survival_rate_cutoff"]] <- survival_rate_cutoff

  # Capture n samples
  object[["n_samples"]] <- n_samples

  # Capture n divisions
  object[["n_divisions"]] <- n_divisions

  # Capture n iterations
  object[["n_iterations"]] <- n_iterations

  # Capture random state
  set_random_state(random_state)
  object[["random_state"]] <- random_state
  
  # Capture progress bar
  object[["progress_bar"]] <- progress_bar
  
  # Capture number of cores
  object[["n_core"]] <- n_core
  
  # Capture coefficients
  object[["generate_coefficients"]] <- coefficients
  
  # Capture variable importances
  object[["generate_variable_importances"]] <- variable_importances
  
  # Capture predictions
  object[["generate_predictions"]] <- predictions
  
  # Capture metrics
  object[["generate_metrics"]] <- metrics

  # Capture model arguments
  object[["model_args"]] <- model_args
  
  # Set and capture column names
  column_names <- colnames(.data)
  column_names <- set_column_names(column_names, dependent_variable, 
                                   preprocess, exclude_variables, 
                                   categorical_variables)
  object[["column_names"]] <- column_names
  
  # Set and capture categorical variables
  cat_vars <- set_categorical_variables(column_names, categorical_variables)
  object[["categorical_variables"]] <- cat_vars
  
  # Remove variables and capture data
  .data <- remove_variables(.data, exclude_variables)
  object[["data"]] <- .data
  
  # Set and capture dependent variable
  y <- set_dependent_variable(.data, dependent_variable)
  object[["y"]] <- y
  
  # Set and capture independent variables
  X <- set_independent_variables(.data, dependent_variable)
  X <- X[, column_names]
  object[["X"]] <- X
  
  # Set class of the object
  object <- structure(object, class = .class)
  
  # Assess if coefficients should be generated for this algorithm
  if (coefficients) {
    # Replicate coefficients
    coefs <- generate_coefficients(object)
    object[["coefficients"]] <- coefs
    
    # Process and capture coefficients
    coefs_processed <- process_coefficients(coefs, survival_rate_cutoff)
    object[["coefficients_processed"]] <- coefs_processed
    
    # Create and capture coefficients plots
    g <- plot_coefficients_processed(coefs_processed)
    object[["plot_coefficients_processed"]] <- g
  }
  
  # Assess if variable importances should be generated for this algorithm
  if (variable_importances) {
    # Replicate variable importances
    variable_imps <- generate_variable_importances(object)
    object[["variable_importances"]] <- variable_imps
    
    # Process and capture variable_importances
    variable_imps_processed <- process_variable_importances(variable_imps)
    object[["variable_importances_processed"]] <- variable_imps_processed
    
    # Create and capture variable importances plot
    g <- plot_variable_importances_processed(variable_imps_processed)
    object[["plot_variable_importances_processed"]] <- g
  }
  
  # Assess if predictions should be generated for this algorithm
  if (predictions) {
    # Resample and capture data
    split_data <- resample(X, y, train_size = train_size, foldid = foldid)
    object[["X_train"]] <- split_data[["X_train"]]
    object[["X_test"]] <- split_data[["X_test"]]
    y_train <- object[["y_train"]] <- split_data[["y_train"]]
    y_test <- object[["y_test"]] <- split_data[["y_test"]]
    
    # Replicate predictions
    preds <- generate_predictions(object)
    predictions_train <- preds[["predictions_train"]]
    object[["predictions_train"]] <- predictions_train
    predictions_test <- preds[["predictions_test"]]
    object[["predictions_test"]] <- predictions_test
    
    # Process and capture predictions
    predictions_train_mean <- apply(predictions_train, 1, mean)
    predictions_test_mean <- apply(predictions_test, 1, mean)
    object[["predictions_train_mean"]] <- predictions_train_mean
    object[["predictions_test_mean"]] <- predictions_test_mean
    
    # Set and capture plot_predictions function
    plot_predictions <- set_plot_predictions(algorithm, family)
    object[["plot_predictions"]] <- plot_predictions
    
    # Create and capture predictions plots
    plot_predictions_train_mean <- 
      plot_predictions(y_train, predictions_train_mean) + 
      ggplot2::labs(subtitle = "Train Predictions")
    object[["plot_predictions_train_mean"]] <- plot_predictions_train_mean
    plot_predictions_test_mean <- 
      plot_predictions(y_test, predictions_test_mean) + 
      ggplot2::labs(subtitle = "Test Predictions")
    object[["plot_predictions_test_mean"]] <- plot_predictions_test_mean
  }
  
  # Assess if metrics should be generated for this algorithm
  if (metrics) {
    # Replicate and capture metrics
    mets <- generate_metrics(object)
    metrics_train_mean <- mets[["metrics_train_mean"]]
    object[["metrics_train_mean"]] <- metrics_train_mean
    metrics_test_mean <- mets[["metrics_test_mean"]]
    object[["metrics_test_mean"]] <- metrics_test_mean
    
    # Set and capture plot_metrics function
    plot_metrics <- set_plot_metrics(measure)
    object[["plot_metrics"]] <- plot_metrics
    
    # Create and capture metrics plots
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
