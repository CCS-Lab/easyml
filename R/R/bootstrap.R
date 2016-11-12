#' TO BE EDITED.
#' 
#' TO BE EDITED.
#'
#' @param fit_model TO BE EDITED.
#' @param predict_model TO BE EDITED.
#' @param X TO BE EDITED.
#' @param y TO BE EDITED.
#' @param n_divisions TO BE EDITED.
#' @param n_iterations TO BE EDITED.
#' @return TO BE EDITED.
#' @export
bootstrap_aucs <- function(fit_model, predict_model, sampler, X, y, n_divisions = 1000, n_iterations = 100) {
  # Create temporary containers
  all_train_aucs <- numeric()
  all_test_aucs <- numeric()

  # Loop over number of divisions
  for (i in 1:n_divisions) {
    # Split data
    split_data <- sampler(X, y)
    X_train <- split_data[["X_train"]]
    X_test <- split_data[["X_test"]]
    y_train <- split_data[["y_train"]]
    y_test <- split_data[["y_test"]]
    
    # Create temporary containers
    train_aucs <- numeric()
    test_aucs <- numeric()
    
    # Loop over number of iterations
    for (j in 1:n_iterations) {
      # Fit estimator with the training set
      results <- fit_model(X_train, y_train)
      
      # Generate scores for training and test sets
      y_train_predictions <- predict_model(results, X_train)
      y_test_predictions <- predict_model(results, X_test)
      
      # Save AUCs
      train_auc <- pROC::roc(y_train, y_train_predictions)
      train_auc <- as.numeric(train_auc$auc)
      test_auc <- pROC::roc(y_test, y_test_predictions)
      test_auc <- as.numeric(test_auc$auc)
      train_aucs <- c(train_aucs, train_auc)
      test_aucs <- c(test_aucs, test_auc)
    }
    
    # Save mean of AUCS
    all_train_aucs <- c(all_train_aucs, mean(train_aucs))
    all_test_aucs <- c(all_test_aucs, mean(test_aucs))
    
  }
  
  list(train_aucs = all_train_aucs, test_aucs = all_test_aucs)
}

#' TO BE EDITED.
#' 
#' TO BE EDITED.
#'
#' @param fit_model TO BE EDITED.
#' @param extract_coefficients TO BE EDITED.
#' @param X TO BE EDITED.
#' @param y TO BE EDITED.
#' @param n_samples TO BE EDITED.
#' @return TO BE EDITED.
#' @export
bootstrap_coefficients <- function(fit_model, extract_coefficients, X, y, n_samples = 1000) {
  # Initialize containers
  coefs <- array(NA, dim = c(ncol(X) + 1, n_samples))

  # Loop over number of iterations
  for (i in 1:n_samples) {
    # Fit estimator with the training set
    results <- fit_model(X, y)

    # Extract and save coefficients
    coefs[, i] <- extract_coefficients(results)
  }
  
  coefs
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
#' @return TO BE EDITED.
#' @export
bootstrap_predictions <- function(fit_model, predict_model, X_train, y_train, X_test, n_samples = 1000) {
  # Initialize containers
  y_train_predictions <- array(NA, dim = c(nrow(X_train), n_samples))
  y_test_predictions <- array(NA, dim = c(nrow(X_test), n_samples))

  # Loop over number of iterations
  for (i in 1:n_samples) {
    # Fit model with the training set
    results <- fit_model(X_train, y_train)
    
    # Save predictions
    y_train_predictions[, i] <- predict_model(results, X_train)
    y_test_predictions[, i] <- predict_model(results, X_test)
  }
  
  list(y_train_predictions = y_train_predictions, 
       y_test_predictions = y_test_predictions)
}
