#' TO BE EDITED.
#' 
#' TO BE EDITED.
#'
#' @return TO BE EDITED.
#' @export
easy_glmnet <- function(.data, dependent_variable, family = "gaussian", 
                        sampler = NULL, preprocessor = NULL, 
                        exclude_variables = NULL, categorical_variables = NULL, 
                        train_size = 0.667, survival_rate_cutoff = 0.05, 
                        n_samples = 1000, n_divisions = 1000, 
                        n_iterations = 10, random_state = NULL, 
                        progress_bar = TRUE, n_core = 1, ...) {
  # Instantiate output
  output <- list()
  
  # Set sampler function
  sampler <- set_sampler(sampler, family)
  output[["sampler"]] <- sampler
  
  # Set preprocessor function
  preprocessor <- set_preprocessor(preprocessor)
  output[["preprocessor"]] <- preprocessor
  
  # Set fit_model function
  fit_model <- set_fit_model(algorithm, family)
  output[["fit_model"]] <- fit_model
  
  # Set extract_coefficients function
  extract_coefficients <- set_extract_coefficients(algorithm, family)
  
  # Set predict_model function
  predict_model <- set_predict_model(algorithm, family)
  
  # Set plot_predictions function
  plot_predictions <- set_plot_predictions(algorithm, family)
  
  # Set replicate_metrics function
  replicate_metrics <- replicate_mses
  
  # Set plot_metrics function
  replicate_metrics <- replicate_mses
  
  # Set random state
  set_random_state(random_state)
  
  # Set column names
  column_names <- set_column_names(colnames(.data), dependent_variable, 
                                   exclude_variables = exclude_variables, 
                                   preprocessor = preprocessor, 
                                   categorical_variables = categorical_variables)
  
  # Set categorical variables
  categorical_variables <- set_categorical_variables(column_names, categorical_variables)
  
  # Remove variables
  .data <- remove_variables(.data, exclude_variables)
  
  # Isolate dependent variable
  y <- isolate_dependent_variable(.data, dependent_variable)
  output[["y"]] <- y
  
  # Isolate independent variables
  X <- isolate_independent_variables(.data, dependent_variable)
  output[["X"]] <- X
  
  # Split data
  split_data <- sampler(X, y, train_size = train_size)
  output <- c(output, split_data)
  X_train <- split_data[["X_train"]]
  X_test <- split_data[["X_test"]]
  y_train <- split_data[["y_train"]]
  y_test <- split_data[["y_test"]]
  
  # Replicate coefficients
  coefficients <- replicate_coefficients(fit_model, extract_coefficients, 
                                         preprocessor, X, y, 
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
  
  # Replicate predictions
  predictions <- replicate_predictions(fit_model, predict_model, 
                                       preprocessor, 
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
  
  # Replicate metrics
  metrics <- replicate_metrics(fit_model, predict_model, 
                               sampler, preprocessor, X, y, 
                               categorical_variables = categorical_variables, 
                               n_divisions = n_divisions, n_iterations = n_iterations, 
                               progress_bar = progress_bar, n_core = n_core, ...)
  output <- c(output, metrics)
  metrics_train_mean <- metrics[["metrics_train_mean"]]
  metrics_test_mean <- metrics[["metrics_test_mean"]]
  
  # Save metrics plots
  output[["plot_metrics_train_mean"]] <- plot_metrics(metrics_train_mean)
  output[["plot_metrics_test_mean"]] <- plot_metrics(metrics_test_mean)
  
  # Return output
  output
}
