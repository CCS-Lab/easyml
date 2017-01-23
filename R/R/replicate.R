#' TO BE EDITED.
#' 
#' TO BE EDITED.
#'
#' @param fit_model TO BE EDITED.
#' @param extract_coefficients TO BE EDITED.
#' @param X TO BE EDITED.
#' @param y TO BE EDITED.
#' @param n_samples TO BE EDITED.
#' @param progress_bar TO BE EDITED.
#' @param parallel TO BE EDITED.
#' @return TO BE EDITED.
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

#' TO BE EDITED.
#' 
#' TO BE EDITED.
#'
#' @param fit_model TO BE EDITED.
#' @param predict_model TO BE EDITED.
#' @param X_train TO BE EDITED.
#' @param y_train TO BE EDITED.
#' @param X_test TO BE EDITED.
#' @param n_samples TO BE EDITED.
#' @param progress_bar TO BE EDITED.
#' @param parallel TO BE EDITED.
#' @return TO BE EDITED.
#' @export
replicate_predictions <- function(fit_model, predict_model, preprocess, 
                                  X_train, y_train, X_test, 
                                  categorical_variables = NULL, 
                                  n_samples = 1000, progress_bar = TRUE, 
                                  n_core = 1, ...) {
  # Handle progress bar
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
    
    # Save predictions
    list(prediction_train = predict_model(results, X_train), 
         prediction_test = predict_model(results, X_test))
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

#' TO BE EDITED.
#' 
#' TO BE EDITED.
#'
#' @param fit_model TO BE EDITED.
#' @param predict_model TO BE EDITED.
#' @param predict_model TO BE EDITED.
#' @param X TO BE EDITED.
#' @param y TO BE EDITED.
#' @param n_divisions TO BE EDITED.
#' @param n_iterations TO BE EDITED.
#' @param progress_bar TO BE EDITED.
#' @param parallel TO BE EDITED.
#' @return TO BE EDITED.
#' @export
replicate_metrics <- function(fit_model, predict_model, resample, preprocess, 
                              measure, X, y, categorical_variables = NULL, 
                              n_divisions = 1000, n_iterations = 100, 
                              progress_bar = TRUE, n_core = 1, ...) {
  # Handle progress bar
  if (progress_bar) {
    print(paste0("Replicating metrics", ifelse(n_core > 1, " in parallel:", ":")))
  }
  
  # Set which looping mechanism to use
  looper <- set_looper(progress_bar, n_core)
  
  # Define closure
  replicate_metric <- function(i) {
    # Split data
    split_data <- resample(X, y)
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
      
      # Generate scores for training and test sets
      predictions_train <- predict_model(results, X_train)
      predictions_test <- predict_model(results, X_test)
      
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
