#' TO BE EDITED.
#' 
#' TO BE EDITED.
#'
#' @return TO BE EDITED.
#' @export
easy_glmnet <- function(.data, dependent_variable = NULL, family = "gaussian", 
                        exclude_variables = NULL, train_size = 0.667,
                        n_divisions = 1000, n_iterations = 10, 
                        n_samples = 1000, out_directory = '.', random_state = NULL,
                        ...) {
  # Process the data
  .processed_data <- process_data(.data, dependent_variable = dependent_variable, exclude_variables = exclude_variables)
  X_data_frame <- .processed_data[["X"]]
  X <- as.matrix(X_data_frame)
  y <- .processed_data[["y"]]
  
  # Create fit, extract, and predict wrapper functions
  fit_model <- function(X, y) {
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
    # Bootstrap coefficients
    coefs <- bootstrap_coefficients(fit_model, extract_coefficients, 
                                    X, y, n_samples = n_samples)
    
    # Process coefficients
    betas <- process_coefficients(coefs, n_samples, c("Intercept", colnames(X_data_frame)))
    plot_betas(betas)
    ggplot2::ggsave("betas.png")

  } else if (family == "binomial") {
    # Set sampler
    sampler <- sample_equal_proportion
    
    # Bootstrap coefficients
    coefs <- bootstrap_coefficients(fit_model, extract_coefficients, 
                                    X, y, n_samples = n_samples)
    
    # # Process coefficients
    betas <- process_coefficients(coefs, n_samples, c("Intercept", colnames(X_data_frame)))
    plot_betas(betas)
    ggplot2::ggsave("betas.png")

    # Split data
    split_data <- sampler(X, y)
    X_train <- split_data[["X_train"]]
    X_test <- split_data[["X_test"]]
    y_train <- split_data[["y_train"]]
    y_test <- split_data[["y_test"]]
    
    # # Bootstrap predictions
    predictions <- bootstrap_predictions(fit_model, predict_model, 
                                         X_train, y_train, X_test, 
                                         n_samples = n_samples)
    y_train_predictions <- predictions[["y_train_predictions"]]
    y_test_predictions <- predictions[["y_test_predictions"]]

    # Generate scores for training and test sets
    y_train_predictions_mean <- apply(y_train_predictions, 1, mean)
    y_test_predictions_mean <- apply(y_test_predictions, 1, mean)

    # Compute ROC curve and ROC area for training
    plot_roc_curve(y_train, y_train_predictions_mean)
    ggplot2::ggsave("train_roc_curve.png")

    # Compute ROC curve and ROC area for test
    plot_roc_curve(y_test, y_test_predictions_mean)
    ggplot2::ggsave("test_roc_curve.png")

    # Bootstrap training and test AUCS
    aucs <- bootstrap_aucs(fit_model, predict_model, sampler, X, y, 
                           n_divisions = n_divisions, n_iterations = n_iterations)
    train_aucs <- aucs[["train_aucs"]]
    test_aucs <- aucs[["test_aucs"]]

    # Plot histogram of training AUCS
    plot_auc_histogram(train_aucs)
    ggplot2::ggsave("train_auc_distribution.png")

    # Plot histogram of test AUCS
    plot_auc_histogram(test_aucs)
    ggplot2::ggsave("test_auc_distribution.png")

  } else {
    stop("Value error!")
  }

  invisible()
}
