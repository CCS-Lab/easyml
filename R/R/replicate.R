#' Replicate coefficients.
#'
#' @param fit_model A function; the function for fitting a model to the data.
#' @param extract_coefficients A function; the function for extracting coefficients from a model.
#' @param preprocess A function; the function for preprocessing the data. Defaults to NULL.
#' @param X A matrix; the independent variables.
#' @param y A vector; the dependent variable.
#' @param categorical_variables A logical vector; each value TRUE indicates that column in the data.frame is a categorical variable. Defaults to NULL.
#' @param n_samples An integer vector of length one; specifies the number of times the coefficients and predictions should be replicated. Defaults to 1000L. 
#' @param progress_bar A logical vector of length one; specifies whether to display a progress bar during calculations. Defaults to TRUE.
#' @param n_core An integer vector of length one; specifies the number of cores to use for this analysis. Currenly only works on Mac OSx and Unix/Linux systems. Defaults to 1L.
#' @param ... The arguments to be passed to the algorithm specified.
#' @return A data.frame, the replicated penalized regression model coefficients.
#' @family replicate
#' @export
replicate_coefficients <- function(fit_model, extract_coefficients, 
                                   preprocess, X, y, 
                                   categorical_variables = NULL, 
                                   n_samples = 1000, progress_bar = TRUE, 
                                   n_core = 1, ...) {
  # Print an informative message
  if (progress_bar) {
    parallel_string <- ifelse(n_core > 1, " in parallel:", ":")
    print(paste0("Replicating coefficients", parallel_string))
  }
  
  # Preprocess data
  result <- preprocess(list(X = X), categorical_variables)
  X <- result[["X"]]

  # Define closure
  replicate_coefficient <- function(i) {
    model <- fit_model(X, y, ...)
    coefficient <- extract_coefficients(model)
    coefficient
  }
  
  # Set which looping mechanism to use
  looper <- set_looper(progress_bar, n_core)
  
  # Loop over number of iterations
  coefficients <- looper(1:n_samples, replicate_coefficient)
  
  # Combine list of data.frames into one data.frame; 
  # structure should be a data.frame of n_samples by ncol(X)
  coefficients <- do.call(rbind, coefficients)
  coefficients
}

#' Replicate variable importances.
#'
#' @param fit_model A function; the function for fitting a model to the data.
#' @param extract_variable_importances A function; the function for extracting variable importances from a model.
#' @param preprocess A function; the function for preprocessing the data. Defaults to NULL.
#' @param X A matrix; the independent variables.
#' @param y A vector; the dependent variable.
#' @param categorical_variables A logical vector; each value TRUE indicates that column in the data.frame is a categorical variable. Defaults to NULL.
#' @param n_samples An integer vector of length one; specifies the number of times the coefficients and predictions should be replicated. Defaults to 1000L. 
#' @param progress_bar A logical vector of length one; specifies whether to display a progress bar during calculations. Defaults to TRUE.
#' @param n_core An integer vector of length one; specifies the number of cores to use for this analysis. Currenly only works on Mac OSx and Unix/Linux systems. Defaults to 1L.
#' @param ... The arguments to be passed to the algorithm specified.
#' @return A data.frame, the replicated variable importance scores.
#' @family replicate
#' @export
replicate_variable_importances <- function(fit_model, extract_variable_importances, 
                                           preprocess, X, y, 
                                           categorical_variables = NULL, 
                                           n_samples = 1000, progress_bar = TRUE, 
                                           n_core = 1, ...) {
  # Print an informative message
  if (progress_bar) {
    parallel_string <- ifelse(n_core > 1, " in parallel:", ":")
    print(paste0("Replicating variable importances", parallel_string))
  }
  
  # Preprocess data
  result <- preprocess(list(X = X), categorical_variables)
  X <- result[["X"]]
  
  # Define closure
  replicate_variable_importance <- function(i) {
    model <- fit_model(X, y, ...)
    variable_importance <- extract_variable_importances(model)
    variable_importance
  }
  
  # Set which looping mechanism to use
  looper <- set_looper(progress_bar, n_core)
  
  # Loop over number of iterations
  variable_importances <- looper(1:n_samples, replicate_variable_importance)
  
  # Combine list of data.frames into one data.frame; 
  # structure should be a data.frame of n_samples by ncol(X)
  variable_importances <- do.call(rbind, variable_importances)
  variable_importances
}

#' Replicate predictions.
#'
#' @param fit_model A function; the function for fitting a model to the data.
#' @param predict_model A function; the function for generating predictions from a fitted model.
#' @param preprocess A function; the function for preprocessing the data. Defaults to NULL.
#' @param X_train A matrix; the independent variables sampled to a training set.
#' @param y_train A vector; the dependent variable sampled to a training set.
#' @param X_test A matrix; the independent variables sampled to a testing set.
#' @param categorical_variables A logical vector; each value TRUE indicates that column in the data.frame is a categorical variable. Defaults to NULL.
#' @param n_samples An integer vector of length one; specifies the number of times the coefficients and predictions should be replicated. Defaults to 1000L. 
#' @param progress_bar A logical vector of length one; specifies whether to display a progress bar during calculations. Defaults to TRUE.
#' @param n_core An integer vector of length one; specifies the number of cores to use for this analysis. Currenly only works on Mac OSx and Unix/Linux systems. Defaults to 1L.
#' @param ... The arguments to be passed to the algorithm specified.
#' @return A list of matrixes, the replicated predictions.
#' @family replicate
#' @export
replicate_predictions <- function(fit_model, predict_model, preprocess, 
                                  X_train, y_train, X_test, 
                                  categorical_variables = NULL, 
                                  n_samples = 1000, progress_bar = TRUE, 
                                  n_core = 1, ...) {
  # Print an informative message
  if (progress_bar) {
    print(paste0("Replicating predictions", ifelse(n_core > 1, " in parallel:", ":")))
  }
  
  # Preprocess data
  result <- preprocess(list(X_train = X_train, X_test = X_test), 
                         categorical_variables = categorical_variables)
  X_train <- result[["X_train"]]
  X_test <- result[["X_test"]]
  
  # Define closure
  replicate_prediction <- function(i) {
    # Fit model with the training set
    results <- fit_model(X_train, y_train, ...)
    
    # Train data set to NULL for training predictions (except glmnet)
    if (identical(predict_model, glmnet_predict_model)) {
      tmp_train <- X_train
    } else {
      tmp_train <- NULL
    }
    
    # Save predictions 
    list(prediction_train = predict_model(results, newx = tmp_train), 
         prediction_test = predict_model(results, newx = X_test))
  }
  
  # Set which looping mechanism to use
  looper <- set_looper(progress_bar, n_core)
  
  # Loop over number of iterations
  output <- looper(1:n_samples, replicate_prediction)
  
  predictions_train <- lapply(output, function(x) x$prediction_train)
  predictions_test <- lapply(output, function(x) x$prediction_test)
  
  predictions_train <- t(matrix(unlist(predictions_train), 
                                ncol = nrow(X_train), byrow = TRUE))
  predictions_test <- t(matrix(unlist(predictions_test), 
                               ncol = nrow(X_test), byrow = TRUE))
  
  list(predictions_train = predictions_train, 
       predictions_test = predictions_test)
}

#' Replicate metrics.
#'
#' @param fit_model A function; the function for fitting a model to the data.
#' @param predict_model A function; the function for generating predictions from a fitted model.
#' @param resample A function; the function for resampling the data. Defaults to NULL.
#' @param preprocess A function; the function for preprocessing the data. Defaults to NULL.
#' @param measure A function; the function for measuring the results. Defaults to NULL.
#' @param X A matrix; the independent variables.
#' @param y A vector; the dependent variable.
#' @param train_size A numeric vector of length one; specifies what proportion of the data should be used for the training data set. Defaults to 0.667.
#' @param categorical_variables A logical vector; each value TRUE indicates that column in the data.frame is a categorical variable. Defaults to NULL.
#' @param n_divisions An integer vector of length one; specifies the number of times the data should be divided when replicating the error metrics. Defaults to 1000L.
#' @param n_iterations An integer vector of length one; during each division, specifies the number of times the predictions should be replicated. Defaults to 10L.
#' @param progress_bar A logical vector of length one; specifies whether to display a progress bar during calculations. Defaults to TRUE.
#' @param n_core An integer vector of length one; specifies the number of cores to use for this analysis. Currenly only works on Mac OSx and Unix/Linux systems. Defaults to 1L.
#' @param foldid A vector with length equal to \code{length(y)} which identifies cases belonging to the same fold.
#' @param ... The arguments to be passed to the algorithm specified.
#' @return A list of matrixes, the replicated metrics.
#' @family replicate
#' @export
replicate_metrics <- function(fit_model, predict_model, resample, preprocess, 
                              measure, X, y, train_size = train_size, 
                              categorical_variables = NULL, 
                              n_divisions = 1000, n_iterations = 100, 
                              progress_bar = TRUE, n_core = 1, foldid = NULL, ...) {
  # Print an informative message
  if (progress_bar) {
    print(paste0("Replicating metrics", ifelse(n_core > 1, " in parallel:", ":")))
  }
  
  # Set which looping mechanism to use
  looper <- set_looper(progress_bar, n_core)
  
  # Define closure
  replicate_metric <- function(i) {
    # Split data
    split_data <- resample(X, y, foldid = foldid, train_size = train_size)
    X_train <- split_data[["X_train"]]
    X_test <- split_data[["X_test"]]
    y_train <- split_data[["y_train"]]
    y_test <- split_data[["y_test"]]
    
    # Preprocess data
    result <- preprocess(list(X_train = X_train, X_test = X_test), 
                           categorical_variables = categorical_variables)
    X_train <- result[["X_train"]]
    X_test <- result[["X_test"]]
    
    # Create temporary containers
    metric_train <- numeric()
    test_metrics <- numeric()
    
    # Loop over number of iterations
    output_iterations <- lapply(1:n_iterations, function(i) {
      # Fit estimator with the training set
      results <- fit_model(X_train, y_train, ...)
      
      # Train data set to NULL for training predictions (except glmnet)
      if (identical(predict_model, glmnet_predict_model)) {
        tmp_train <- X_train
      } else {
        tmp_train <- NULL
      }
      
      predictions_train <- predict_model(results, newx = tmp_train)
      predictions_test <- predict_model(results, newx = X_test)
      
      # Save metrics
      metric_train <- measure(y_train, predictions_train)
      metric_test <- measure(y_test, predictions_test)
      list(metric_train = metric_train, metric_test = metric_test)
    })
    
    # Take average of metrics
    metrics_train <- unlist(lapply(output_iterations, function(x) x$metric_train))
    metrics_test <- unlist(lapply(output_iterations, function(x) x$metric_test))
    
    # Save mean of metrics
    list(metrics_train_mean = mean(metrics_train), 
         metrics_test_mean = mean(metrics_test))
  }
  
  # Loop over number of divisions
  output_divisions <- looper(1:n_divisions, replicate_metric)
  
  metrics_train_mean <- unlist(lapply(output_divisions, function(x) x$metrics_train_mean))
  metrics_test_mean <- unlist(lapply(output_divisions, function(x) x$metrics_test_mean))
  
  list(metrics_train_mean = metrics_train_mean, 
       metrics_test_mean = metrics_test_mean)
}
