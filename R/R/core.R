#' TO BE EDITED.
#' 
#' TO BE EDITED.
#'
#' @return TO BE EDITED.
#' @export
easy_analysis <- function(.data, dependent_variable, algorithm = NULL, 
                          family = "gaussian", resample = NULL, 
                          preprocess = NULL, measure = NULL, 
                          exclude_variables = NULL, 
                          categorical_variables = NULL, train_size = 0.667, 
                          survival_rate_cutoff = 0.05, n_samples = 1000, 
                          n_divisions = 1000, n_iterations = 10, 
                          random_state = NULL, progress_bar = TRUE, 
                          n_core = 1, ...) {
  # Instantiate output
  output <- list()
  
  # Set random state
  set_random_state(random_state)
  
  # Set coefficients boolean
  coefficients_boolean <- set_coefficients_boolean(algorithm)
  
  # Set predictions boolean
  predictions_boolean <- set_predictions_boolean(algorithm)
  
  # Set metrics boolean
  metrics_boolean <- set_metrics_boolean(algorithm)
  
  # Set resample function
  resample <- set_resample(resample, family)
  output[["resample"]] <- resample
  
  # Set preprocess function
  preprocess <- set_preprocess(preprocess)
  output[["preprocess"]] <- preprocess
  
  # Set measure function
  measure <- set_measure(measure, algorithm, family)
  output[["measure"]] <- measure
  
  # Set fit_model function
  fit_model <- set_fit_model(algorithm, family)
  output[["fit_model"]] <- fit_model
  
  # Set extract_coefficients function
  extract_coefficients <- set_extract_coefficients(algorithm, family)
  output[["extract_coefficients"]] <- extract_coefficients
  
  # Set predict_model function
  predict_model <- set_predict_model(algorithm, family)
  output[["predict_model"]] <- predict_model
  
  # Set plot_predictions function
  plot_predictions <- set_plot_predictions(algorithm, family)
  output[["plot_predictions"]] <- plot_predictions
  
  # Set plot_metrics function
  plot_metrics <- set_plot_metrics(measure)
  output[["plot_metrics"]] <- plot_metrics
  
  # Set column names
  column_names <- set_column_names(colnames(.data), dependent_variable, 
                                   exclude_variables = exclude_variables, 
                                   preprocess = preprocess, 
                                   categorical_variables = categorical_variables)
  
  # Set categorical variables
  categorical_variables <- set_categorical_variables(column_names, categorical_variables)
  
  # Remove variables
  .data <- remove_variables(.data, exclude_variables)
  output[["data"]] <- .data
  
  # Set dependent variable
  y <- set_dependent_variable(.data, dependent_variable)
  output[["y"]] <- y
  
  # Set independent variables
  X <- set_independent_variables(.data, dependent_variable)
  output[["X"]] <- X
  
  # Resample data
  split_data <- resample(X, y, train_size = train_size)
  output <- c(output, split_data)
  X_train <- split_data[["X_train"]]
  X_test <- split_data[["X_test"]]
  y_train <- split_data[["y_train"]]
  y_test <- split_data[["y_test"]]
  
  # Assess if coefficients should be replicated for this algorithm
  if (coefficients_boolean) {
    # Replicate coefficients
    coefficients <- replicate_coefficients(fit_model, extract_coefficients, 
                                           preprocess, X, y, 
                                           categorical_variables = categorical_variables, 
                                           n_samples = n_samples, 
                                           progress_bar = progress_bar, 
                                           n_core = n_core, ...)
    output[["coefficients"]] <- coefficients
    
    # Process coefficients
    coefficients_processed <- process_coefficients(coefficients, survival_rate_cutoff)
    output[["coefficients_processed"]] <- coefficients_processed
    
    # Save coefficients plots
    output[["plot_coefficients_processed"]] <- plot_coefficients_processed(coefficients_processed)
  }
  
  # Assess if predictions should be replicated for this algorithm
  if (predictions_boolean) {
    # Replicate predictions
    predictions <- replicate_predictions(fit_model, predict_model, 
                                         preprocess, 
                                         X_train, y_train, X_test, 
                                         categorical_variables = categorical_variables, 
                                         n_samples = n_samples, 
                                         progress_bar = progress_bar, 
                                         n_core = n_core, ...)
    output <- c(output, predictions)
    predictions_train <- predictions[["predictions_train"]]
    predictions_test <- predictions[["predictions_test"]]
    
    # Process predictions
    predictions_train_mean <- apply(predictions_train, 1, mean)
    predictions_test_mean <- apply(predictions_test, 1, mean)
    output[["predictions_train_mean"]] <- predictions_train_mean
    output[["predictions_test_mean"]] <- predictions_test_mean
    
    # Save predictions plots
    output[["plot_predictions_train"]] <- plot_predictions(y_train, predictions_train_mean)
    output[["plot_predictions_test"]] <- plot_predictions(y_test, predictions_test_mean)
  }
  
  # Assess if metrics should be replicated for this algorithm
  if (metrics_boolean) {
    # Replicate metrics
    metrics <- replicate_metrics(fit_model, predict_model, 
                                 resample, preprocess, measure, X, y, 
                                 categorical_variables = categorical_variables, 
                                 n_divisions = n_divisions, n_iterations = n_iterations, 
                                 progress_bar = progress_bar, n_core = n_core, ...)
    output <- c(output, metrics)
    metrics_train_mean <- metrics[["metrics_train_mean"]]
    metrics_test_mean <- metrics[["metrics_test_mean"]]
    
    # Save metrics plots
    output[["plot_metrics_train_mean"]] <- plot_metrics(metrics_train_mean)
    output[["plot_metrics_test_mean"]] <- plot_metrics(metrics_test_mean)
  }
  
  # Return output
  output
}
