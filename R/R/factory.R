#' TO BE EDITED.
#' 
#' TO BE EDITED.
#'
#' @return TO BE EDITED.
#' @export
easy_glmnet <- function(.data, dependent_variable, family = "gaussian", 
                        sampler = NULL, exclude_variables = NULL, 
                        categorical_variables = NULL, standardize_data = TRUE, 
                        train_size = 0.667, survival_rate_cutoff = 0.05, 
                        n_samples = 1000, n_divisions = 1000, 
                        n_iterations = 10, out_directory = ".", 
                        random_state = NULL, progress_bar = FALSE, 
                        parallel = FALSE, ...) {
  # Handle random state
  if (!is.null(random_state)) {
    set.seed(random_state)
  }
  
  # Handle columns
  column_names <- colnames(.data)
  column_names <- column_names[column_names != dependent_variable]
  if (!is.null(exclude_variables)) {
    column_names <- column_names[!(column_names %in% exclude_variables)]
  }
  
  # Exclude certain variables and y
  y <- .data[, dependent_variable]
  .data[, dependent_variable] <- NULL
  
  if (!is.null(exclude_variables)) {
    .data[, exclude_variables] <- NULL
  }
  
  # If True, standardize the data
  if (standardize_data) {
    if (is.null(categorical_variables)) {
      X_data_frame <- data.frame(scale(.data))
      X <- as.matrix(X_data_frame)
    } else {
      mask <- colnames(.data) %in% categorical_variables
      X_categorical <- .data[, mask, drop = FALSE]
      X_numeric <- .data[, !mask]
      X_std <- data.frame(scale(X_numeric))
      X_data_frame <- cbind(X_categorical, X_std)
      X <- as.matrix(X_data_frame)
      column_names <- c(categorical_variables, setdiff(column_names, categorical_variables))
    }
  }
  
  # Create fit, extract, and predict wrapper functions
  fit_model <- function(X, y, ...) {
    model <- glmnet::glmnet(X, y, family = family, ...)
    cv_model <- glmnet::cv.glmnet(X, y, family = family, ...)
    list(model = model, cv_model = cv_model)
  }
  
  extract_coefficients <- function(results) {
    model <- results[["model"]]
    cv_model <- results[["cv_model"]]
    as.numeric(coef(model, s = cv_model$lambda.min))
  }
  
  predict_model <- function(results, newx) {
    model <- results[["model"]]
    cv_model <- results[["cv_model"]]
    predict(model, newx = newx, s = cv_model$lambda.min, type = "response")
  }
  
  if (family == "gaussian") {
    # Set sample
    if (is.null(sampler)) {
      sampler <- train_test_split
    }
    
    # Bootstrap coefficients
    coefs <- bootstrap_coefficients(fit_model, extract_coefficients, X, y, 
                                    n_samples = n_samples, 
                                    progress_bar = progress_bar, 
                                    parallel = parallel)
    
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
    predictions <- bootstrap_predictions(fit_model, predict_model, 
                                         X_train, y_train, X_test, 
                                         n_samples = n_samples, 
                                         progress_bar = progress_bar, 
                                         parallel = parallel)
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
    mses <- bootstrap_mses(fit_model, predict_model, sampler, X, y, 
                           n_divisions = n_divisions, n_iterations = n_iterations, 
                           progress_bar = progress_bar, parallel = parallel)
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
    coefs <- bootstrap_coefficients(fit_model, extract_coefficients, 
                                    X, y, n_samples = n_samples, 
                                    progress_bar = progress_bar, 
                                    parallel = parallel)
    
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
    predictions <- bootstrap_predictions(fit_model, predict_model, 
                                         X_train, y_train, X_test, 
                                         n_samples = n_samples, 
                                         progress_bar = progress_bar, 
                                         parallel = parallel)
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
    aucs <- bootstrap_aucs(fit_model, predict_model, sampler, X, y, 
                           n_divisions = n_divisions, n_iterations = n_iterations, 
                           progress_bar = progress_bar, parallel = parallel)
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
