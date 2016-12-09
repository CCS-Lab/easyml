#' TO BE EDITED.
#' 
#' TO BE EDITED.
#'
#' @return TO BE EDITED.
#' @export
glmnet_analysis <- function(.data, dependent_variable, family = "gaussian", 
                            sampler = NULL, preprocessor = NULL, 
                            exclude_variables = NULL, categorical_variables = NULL, 
                            train_size = 0.667, survival_rate_cutoff = 0.05, 
                            n_samples = 1000, n_divisions = 1000, 
                            n_iterations = 10, out_directory = ".", 
                            random_state = NULL, progress_bar = TRUE, 
                            n_core = 1, ...) {
  # Set random state
  set_random_state(random_state)
  
  # Set column names
  column_names <- colnames(.data)
  
  # Exclude certain variables
  if (!is.null(exclude_variables)) {
    .data[, exclude_variables] <- NULL
    column_names <- setdiff(column_names, exclude_variables)
  }
  
  # Move categorical names to the front when there are categorical variables
  if (!is.null(categorical_variables) && !is.null(preprocessor)) {
    column_names <- setdiff(column_names, categorical_variables)
    column_names <- c(categorical_variables, column_names)
  }
  
  # Isolate y
  y <- .data[, dependent_variable]
  
  # Remove y column name from column names
  column_names <- setdiff(column_names, dependent_variable)
  .data[, dependent_variable] <- NULL
  
  # Isolate X
  X <- .data
  
  # Set preprocessor function
  if (is.null(preprocessor)) {
    preprocessor <- preprocess_identity
  }
  
  # assess family of regression
  if (family == "gaussian") {
    # Set sampler function
    if (is.null(sampler)) {
      sampler <- train_test_split
    }
    
    # Bootstrap coefficients
    coefs <- bootstrap_coefficients(glmnet_fit_model_gaussian, glmnet_extract_coefficients, 
                                    preprocessor, X, y, 
                                    categorical_variables = categorical_variables, 
                                    n_samples = n_samples, 
                                    progress_bar = progress_bar, 
                                    n_core = n_core, ...)
    
    # Process coefficients
    betas <- process_coefficients(coefs, column_names, 
                                  survival_rate_cutoff = survival_rate_cutoff)
    plot_betas(betas)
    ggplot2::ggsave(file.path(out_directory, "betas.png"))
    
    # Split data
    split_data <- sampler(X, y, train_size = train_size)
    X_train <- split_data[["X_train"]]
    X_test <- split_data[["X_test"]]
    y_train <- split_data[["y_train"]]
    y_test <- split_data[["y_test"]]
    
    # Bootstrap predictions
    predictions <- bootstrap_predictions(glmnet_fit_model_gaussian, glmnet_predict_model, 
                                         preprocessor, 
                                         X_train, y_train, X_test, 
                                         categorical_variables = categorical_variables, 
                                         n_samples = n_samples, 
                                         progress_bar = progress_bar, 
                                         n_core = n_core, ...)
    y_train_predictions <- predictions[["y_train_predictions"]]
    y_test_predictions <- predictions[["y_test_predictions"]]
    
    # Take average of predictions for training and test sets
    y_train_predictions_mean <- apply(y_train_predictions, 1, mean)
    y_test_predictions_mean <- apply(y_test_predictions, 1, mean)
    
    # Plot the gaussian predictions for training
    plot_gaussian_predictions(y_train, y_train_predictions_mean)
    ggplot2::ggsave(file.path(out_directory, "train_gaussian_predictions.png"))
    
    # Plot the gaussian predictions for test
    plot_gaussian_predictions(y_test, y_test_predictions_mean)
    ggplot2::ggsave(file.path(out_directory, "test_gaussian_predictions.png"))
    
    # Bootstrap training and test MSEs
    mses <- bootstrap_mses(glmnet_fit_model_gaussian, glmnet_predict_model, 
                           sampler, preprocessor, X, y, 
                           categorical_variables = categorical_variables, 
                           n_divisions = n_divisions, 
                           n_iterations = n_iterations, 
                           progress_bar = progress_bar, n_core = n_core, ...)
    train_mses <- mses[["mean_train_metrics"]]
    test_mses <- mses[["mean_test_metrics"]]
    
    # Plot histogram of training MSEs
    plot_mse_histogram(train_mses)
    ggplot2::ggsave(file.path(out_directory, "train_mse_distribution.png"))
    
    # Plot histogram of test MSEs
    plot_mse_histogram(test_mses)
    ggplot2::ggsave(file.path(out_directory, "test_mse_distribution.png"))
    
  } else if (family == "binomial") {
    # Set sample
    if (is.null(sampler)) {
      sampler <- sample_equal_proportion
    }
    
    # Bootstrap coefficients
    coefs <- bootstrap_coefficients(glmnet_fit_model_binomial, glmnet_extract_coefficients, 
                                    preprocessor, X, y, 
                                    categorical_variables = categorical_variables, 
                                    n_samples = n_samples, 
                                    progress_bar = progress_bar, 
                                    n_core = n_core, ...)
    
    # Process coefficients
    betas <- process_coefficients(coefs, column_names, 
                                  survival_rate_cutoff = survival_rate_cutoff)
    plot_betas(betas)
    ggplot2::ggsave(file.path(out_directory, "betas.png"))
    
    # Split data
    split_data <- sampler(X, y, train_size = train_size)
    X_train <- split_data[["X_train"]]
    X_test <- split_data[["X_test"]]
    y_train <- split_data[["y_train"]]
    y_test <- split_data[["y_test"]]
    
    # Bootstrap predictions
    predictions <- bootstrap_predictions(glmnet_fit_model_binomial, glmnet_predict_model, 
                                         preprocessor, 
                                         X_train, y_train, X_test, 
                                         categorical_variables = categorical_variables, 
                                         n_samples = n_samples, 
                                         progress_bar = progress_bar, 
                                         n_core = n_core, ...)
    y_train_predictions <- predictions[["y_train_predictions"]]
    y_test_predictions <- predictions[["y_test_predictions"]]
    
    # Generate scores for training and test sets
    y_train_predictions_mean <- apply(y_train_predictions, 1, mean)
    y_test_predictions_mean <- apply(y_test_predictions, 1, mean)
    
    # Compute ROC curve and ROC area for training
    plot_roc_curve(y_train, y_train_predictions_mean)
    ggplot2::ggsave(file.path(out_directory, "train_roc_curve.png"))
    
    # Compute ROC curve and ROC area for test
    plot_roc_curve(y_test, y_test_predictions_mean)
    ggplot2::ggsave(file.path(out_directory, "test_roc_curve.png"))
    
    # Bootstrap training and test AUCs
    aucs <- bootstrap_aucs(glmnet_fit_model_binomial, glmnet_predict_model, 
                           sampler, preprocessor, X, y, 
                           categorical_variables = categorical_variables, 
                           n_divisions = n_divisions, n_iterations = n_iterations, 
                           progress_bar = progress_bar, n_core = n_core, ...)
    train_aucs <- aucs[["mean_train_metrics"]]
    test_aucs <- aucs[["mean_test_metrics"]]
    
    # Plot histogram of training AUCs
    plot_auc_histogram(train_aucs)
    ggplot2::ggsave(file.path(out_directory, "train_auc_distribution.png"))
    
    # Plot histogram of test AUCs
    plot_auc_histogram(test_aucs)
    ggplot2::ggsave(file.path(out_directory, "test_auc_distribution.png"))
    
  } else {
    stop("Value error!")
  }
  
  invisible()
}

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
  as.numeric(coef(model, s = cv_model$lambda.min))
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
