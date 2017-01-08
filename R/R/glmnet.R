#' TO BE EDITED.
#' 
#' TO BE EDITED.
#'
#' @return TO BE EDITED.
#' @export
glmnet_fit_model_gaussian <- function(X, y, ...) {
  X <- as.matrix(X)
  model <- glmnet::glmnet(X, y, family = "gaussian", ...)
  cv_model <- glmnet::cv.glmnet(X, y, family = "gaussian", ...)
  list(model = model, cv_model = cv_model)
}

#' TO BE EDITED.
#' 
#' TO BE EDITED.
#'
#' @return TO BE EDITED.
#' @export
glmnet_fit_model_binomial <- function(X, y, ...) {
  X <- as.matrix(X)
  model <- glmnet::glmnet(X, y, family = "binomial", ...)
  cv_model <- glmnet::cv.glmnet(X, y, family = "binomial", ...)
  list(model = model, cv_model = cv_model)
}

#' TO BE EDITED.
#' 
#' TO BE EDITED.
#'
#' @return TO BE EDITED.
#' @export
glmnet_extract_coefficients <- function(results) {
  model <- results[["model"]]
  cv_model <- results[["cv_model"]]
  coefs <- coef(model, s = cv_model$lambda.min)
  .data <- data.frame(t(as.matrix(as.numeric(coefs), nrow = 1)))
  colnames(.data) <- rownames(coefs)
  .data
}

#' TO BE EDITED.
#' 
#' TO BE EDITED.
#'
#' @return TO BE EDITED.
#' @export
glmnet_predict_model <- function(results, newx) {
  newx <- as.matrix(newx)
  model <- results[["model"]]
  cv_model <- results[["cv_model"]]
  predict(model, newx = newx, s = cv_model$lambda.min, type = "response")
}

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
  # Set sampler function
  sampler <- set_sampler(sampler, family)
  
  # Set preprocessor function
  preprocessor <- set_preprocessor(preprocessor)
  
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
  
  # Isolate independent variables
  X <- isolate_independent_variables(.data, dependent_variable)
  
  # Instantiate output
  output <- list()
  
  # assess family of regression
  if (family == "gaussian") {

    # Replicate coefficients
    coefs <- replicate_coefficients(glmnet_fit_model_gaussian, 
                                    glmnet_extract_coefficients, 
                                    preprocessor, X, y, 
                                    categorical_variables = categorical_variables, 
                                    n_samples = n_samples, 
                                    progress_bar = progress_bar, 
                                    n_core = n_core, ...)
    output[["coefs"]] <- coefs
    
    # Process coefficients
    betas <- process_coefficients(coefs, survival_rate_cutoff)
    output[["betas"]] <- betas
    output[["betas_plot"]] <- plot_betas(betas)
    
    # Split data
    split_data <- sampler(X, y, train_size = train_size)
    output <- c(output, split_data)
    X_train <- split_data[["X_train"]]
    X_test <- split_data[["X_test"]]
    y_train <- split_data[["y_train"]]
    y_test <- split_data[["y_test"]]
    
    # Replicate predictions
    predictions <- replicate_predictions(glmnet_fit_model_gaussian, 
                                         glmnet_predict_model, 
                                         preprocessor, 
                                         X_train, y_train, X_test, 
                                         categorical_variables = categorical_variables, 
                                         n_samples = n_samples, 
                                         progress_bar = progress_bar, 
                                         n_core = n_core, ...)
    output <- c(output, predictions)
    y_train_predictions <- predictions[["y_train_predictions"]]
    y_test_predictions <- predictions[["y_test_predictions"]]
    
    # Take average of predictions for training and test sets
    y_train_predictions_mean <- apply(y_train_predictions, 1, mean)
    y_test_predictions_mean <- apply(y_test_predictions, 1, mean)
    
    # Save plots
    output[["predictions_train_plot"]] <- plot_gaussian_predictions(y_train, y_train_predictions_mean)
    output[["predictions_test_plot"]] <- plot_gaussian_predictions(y_test, y_test_predictions_mean)
    
    # Replicate training and test metrics
    mses <- replicate_mses(glmnet_fit_model_gaussian, glmnet_predict_model, 
                           sampler, preprocessor, X, y, 
                           categorical_variables = categorical_variables, 
                           n_divisions = n_divisions, 
                           n_iterations = n_iterations, 
                           progress_bar = progress_bar, n_core = n_core, ...)
    output <- c(output, mses)
    train_mses <- mses[["mean_train_metrics"]]
    test_mses <- mses[["mean_test_metrics"]]
    
    # Save plots
    output[["metrics_train_plot"]] <- plot_mse_histogram(train_mses)
    output[["metrics_test_plot"]] <- plot_mse_histogram(test_mses)

  } else if (family == "binomial") {

    # Replicate coefficients
    coefs <- replicate_coefficients(glmnet_fit_model_binomial, glmnet_extract_coefficients, 
                                    preprocessor, X, y, 
                                    categorical_variables = categorical_variables, 
                                    n_samples = n_samples, 
                                    progress_bar = progress_bar, 
                                    n_core = n_core, ...)
    output[["coefs"]] <- coefs
    
    # Process coefficients
    betas <- process_coefficients(coefs, survival_rate_cutoff)
    output[["betas"]] <- betas
    output[["betas_plot"]] <- plot_betas(betas)
    
    # Split data
    split_data <- sampler(X, y, train_size = train_size)
    output <- c(output, split_data)
    X_train <- split_data[["X_train"]]
    X_test <- split_data[["X_test"]]
    y_train <- split_data[["y_train"]]
    y_test <- split_data[["y_test"]]
    
    # Replicate predictions
    predictions <- replicate_predictions(glmnet_fit_model_binomial, glmnet_predict_model, 
                                         preprocessor, 
                                         X_train, y_train, X_test, 
                                         categorical_variables = categorical_variables, 
                                         n_samples = n_samples, 
                                         progress_bar = progress_bar, 
                                         n_core = n_core, ...)
    output <- c(output, predictions)
    y_train_predictions <- predictions[["y_train_predictions"]]
    y_test_predictions <- predictions[["y_test_predictions"]]
    
    # Generate scores for training and test sets
    y_train_predictions_mean <- apply(y_train_predictions, 1, mean)
    y_test_predictions_mean <- apply(y_test_predictions, 1, mean)
    
    # Save plots
    output[["predictions_train_plot"]] <- plot_roc_curve(y_train, y_train_predictions_mean)
    output[["predictions_test_plot"]] <- plot_roc_curve(y_test, y_test_predictions_mean)
    
    # Replicate training and test AUCs
    aucs <- replicate_aucs(glmnet_fit_model_binomial, glmnet_predict_model, 
                           sampler, preprocessor, X, y, 
                           categorical_variables = categorical_variables, 
                           n_divisions = n_divisions, n_iterations = n_iterations, 
                           progress_bar = progress_bar, n_core = n_core, ...)
    output <- c(output, aucs)
    train_aucs <- aucs[["mean_train_metrics"]]
    test_aucs <- aucs[["mean_test_metrics"]]
    
    # Save plots
    output[["metrics_train_plot"]] <- plot_auc_histogram(train_aucs)
    output[["metrics_test_plot"]] <- plot_auc_histogram(test_aucs)

  } else {
    stop("Value error!")
  }
  
  output
}
