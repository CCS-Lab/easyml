#' Generate coefficients for a model (if applicable).
#'
#' @param object A list of class \code{easy_*}, where * is the name of the algorithm.
#' @return A data.frame, the generated penalized regression model coefficients.
#' @family generate
#' @export
generate_coefficients <- function(object) {
  # Extract attributes from object
  X <- object[["X"]]
  y <- object[["y"]]
  categorical_variables <- object[["categorical_variables"]]
  preprocess <- object[["preprocess"]]
  n_samples <- object[["n_samples"]]
  progress_bar <- object[["progress_bar"]]
  n_core <- object[["n_core"]]
  
  # Print an informative message
  if (progress_bar) {
    msg <- "Generating coefficients from multiple model builds"
    parallel_string <- ifelse(n_core > 1, " in parallel:", ":")
    print(paste0(msg, parallel_string))
  }
  
  # Preprocess data
  result <- preprocess(list(X = X), categorical_variables)
  object[["X"]] <- result[["X"]]
  
  # Define closure
  generate_coefficient <- function(i) {
    results <- fit_model(object)
    coef <- extract_coefficients(results)
    coef
  }
  
  # Set which looping mechanism to use
  looper <- set_looper(progress_bar, n_core)
  
  # Loop over number of iterations
  coefs <- looper(1:n_samples, generate_coefficient)
  
  # Combine list of data.frames into one data.frame; 
  # structure should be a data.frame of n_samples by ncol(X)
  coefs <- do.call(rbind, coefs)
  coefs
}

#' Generate variable importances for a model (if applicable).
#'
#' @param object A list of class \code{easy_*}, where * is the name of the algorithm.
#' @return A data.frame, the generated variable importance scores.
#' @family generate
#' @export
generate_variable_importances <- function(object) {
  # Extract attributes from object
  X <- object[["X"]]
  y <- object[["y"]]
  categorical_variables <- object[["categorical_variables"]]
  preprocess <- object[["preprocess"]]
  n_samples <- object[["n_samples"]]
  progress_bar <- object[["progress_bar"]]
  n_core <- object[["n_core"]]
  
  if (progress_bar) {
    msg <- "Generating variable importances from multiple model builds"
    parallel_string <- ifelse(n_core > 1, " in parallel:", ":")
    print(paste0(msg, parallel_string))
  }

  # Preprocess data
  result <- preprocess(list(X = X), categorical_variables)
  object[["X"]] <- result[["X"]]
  
  # Define closure
  generate_variable_importance <- function(i) {
    model <- fit_model(object)
    variable_importance <- extract_variable_importances(model)
    variable_importance
  }
  
  # Set which looping mechanism to use
  looper <- set_looper(progress_bar, n_core)
  
  # Loop over number of iterations
  variable_importances <- looper(1:n_samples, generate_variable_importance)
  
  # Combine list of data.frames into one data.frame; 
  # structure should be a data.frame of n_samples by ncol(X)
  variable_importances <- do.call(rbind, variable_importances)
  variable_importances
}

#' Generate predictions for a model.
#'
#' @param object A list of class \code{easy_*}, where * is the name of the algorithm.
#' @return A list of matrixes, the generated predictions.
#' @family generate
#' @export
generate_predictions <- function(object) {
  # Extract attributes from object
  X_train <- object[["X_train"]]
  X_test <- object[["X_test"]]
  y_train <- object[["y_train"]]
  categorical_variables <- object[["categorical_variables"]]
  preprocess <- object[["preprocess"]]
  n_samples <- object[["n_samples"]]
  progress_bar <- object[["progress_bar"]]
  n_core <- object[["n_core"]]

  # Print an informative message
  if (progress_bar) {
    msg <- "Generating predictions for a single train test split"
    parallel_string <- ifelse(n_core > 1, " in parallel:", ":")
    print(paste0(msg, parallel_string))
  }
  
  # Preprocess data
  result <- preprocess(list(X_train = X_train, X_test = X_test), 
                       categorical_variables = categorical_variables)
  object[["X"]] <- result[["X_train"]]
  object[["y"]] <- y_train

  # Define closure
  generate_prediction <- function(i) {
    # Fit model with the training set
    results <- fit_model(object)
    
    # Generate predictions
    prediction_train = predict_model(results, newx = result[["X_train"]])
    prediction_test = predict_model(results, newx = result[["X_test"]])
    
    # Save predictions 
    list(prediction_train = prediction_train, 
         prediction_test = prediction_test)
  }
  
  # Set which looping mechanism to use
  looper <- set_looper(progress_bar, n_core)
  
  # Loop over number of iterations
  output <- looper(1:n_samples, generate_prediction)
  
  predictions_train <- lapply(output, function(x) x$prediction_train)
  predictions_test <- lapply(output, function(x) x$prediction_test)
  
  predictions_train <- t(matrix(unlist(predictions_train), 
                                ncol = nrow(X_train), byrow = TRUE))
  predictions_test <- t(matrix(unlist(predictions_test), 
                               ncol = nrow(X_test), byrow = TRUE))
  
  list(predictions_train = predictions_train, 
       predictions_test = predictions_test)
}

#' Generate measures of model performance for a model.
#'
#' @param object A list of class \code{easy_*}, where * is the name of the algorithm.
#' @return A list of matrixes, the generated measures of model performance.
#' @family generate
#' @export
generate_model_performance <- function(object) {
  # Extract attributes from object
  X <- object[["X"]]
  y <- object[["y"]]
  categorical_variables <- object[["categorical_variables"]]
  train_size <- object[["train_size"]]
  foldid <- object[["foldid"]]
  resample <- object[["resample"]]
  preprocess <- object[["preprocess"]]
  measure <- object[["measure"]]
  n_divisions <- object[["n_divisions"]]
  n_iterations <- object[["n_iterations"]]
  progress_bar <- object[["progress_bar"]]
  n_core <- object[["n_core"]]

  # Print an informative message
  if (progress_bar) {
    msg1 <- "Generating measures of model performance"
    msg2 <- " over multiple train test splits"
    msg <- paste0(msg1, msg2)
    parallel_string <- ifelse(n_core > 1, " in parallel:", ":")
    print(paste0(msg, parallel_string))
  }
  
  # Set which looping mechanism to use
  looper <- set_looper(progress_bar, n_core)
  
  # Define closure
  generate_model_performance_ <- function(i) {
    # Split data
    split_data <- resample(X, y, foldid = foldid, train_size = train_size)
    X_train <- split_data[["X_train"]]
    X_test <- split_data[["X_test"]]
    y_train <- object[["y"]] <- split_data[["y_train"]]
    y_test <- split_data[["y_test"]]
    
    # Preprocess data
    result <- preprocess(list(X_train = X_train, X_test = X_test), 
                         categorical_variables = categorical_variables)
    X_train <- object[["X"]] <- result[["X_train"]]
    X_test <- result[["X_test"]]

    # Create temporary containers
    model_performance_train <- numeric()
    model_performance_test <- numeric()
    
    # Loop over number of iterations
    output_iterations <- lapply(1:n_iterations, function(i) {
      # Fit estimator with the training set
      results <- fit_model(object)
      
      # Generate predictions
      predictions_train <- predict_model(results, newx = X_train)
      predictions_test <- predict_model(results, newx = X_test)
      
      # Save measures of model performance
      list(predictions_train = predictions_train, 
           predictions_test = predictions_test)
    })
    
    # Take average of predicitons
    predictions_train <- lapply(output_iterations, 
                                function(x) x$predictions_train)
    predictions_train <- matrix(unlist(predictions_train), ncol = n_iterations)
    predictions_test <- lapply(output_iterations, 
                               function(x) x$predictions_test)
    predictions_test <- matrix(unlist(predictions_test), ncol = n_iterations)
    
    # Save mean of predictions
    predictions_train <- apply(predictions_train, 1, mean)
    predictions_test <- apply(predictions_test, 1, mean)
    
    # Create measures of model performance
    model_performance_train <- measure(y_train, predictions_train)
    model_performance_test <- measure(y_test, predictions_test)
    
    list(model_performance_train = model_performance_train, 
         model_performance_test = model_performance_test)
  }

  # Loop over number of divisions
  output_divisions <- looper(1:n_divisions, generate_model_performance_)
  model_performance_train <- lapply(output_divisions, 
                                    function(x) x$model_performance_train)
  model_performance_train <- unlist(model_performance_train)
  model_performance_test <- lapply(output_divisions, 
                                   function(x) x$model_performance_test)
  model_performance_test <- unlist(model_performance_test)
  
  list(model_performance_train = model_performance_train, 
       model_performance_test = model_performance_test)
}
