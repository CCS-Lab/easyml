#' TO BE EDITED.
#' 
#' TO BE EDITED.
#'
#' @param fit TO BE EDITED.
#' @param predict TO BE EDITED.
#' @param X TO BE EDITED.
#' @param y TO BE EDITED.
#' @param n_divisions TO BE EDITED.
#' @param n_iterations TO BE EDITED.
#' @return TO BE EDITED.
#' @export
bootstrap_aucs <- function(fit, predict, X, y, n_divisions = 1000, n_iterations = 100) {
  # Create temporary containers
  train_aucs = numeric()
  test_aucs = numeric()

  # Loop over number of divisions
  for (i in range(n_divisions)) {
    # Split data
    mask = easyml::sample_equal_proportion(y, random_state = i)
    y_train = y[mask]
    y_test = y[!mask]
    X_train = X[mask, ]
    X_test = X[!mask, ]
    
    # Create temporary containers
    train_aucs = numeric()
    test_aucs = numeric()
    
    # Loop over number of iterations
    for (j in range(n_iterations)) {
      # Fit estimator with the training set
      model <- fit(X_train, y_train)
      
      # Generate scores for training and test sets
      y_train_scores <- predict(model, X_train)
      y_test_scores <- predict(model, X_test)
      
      # Save predictions
      y_train_predictions[i, ] <- predict(model, X_train)
      y_test_predictions[i, ] <- predict(model, X_test)
      
      # Save AUCs
      train_aucs <- c(train_aucs, auc(y_train, y_train_scores))
      test_aucs <- c(test_aucs, auc(y_test, y_test_scores))
    }
  }
  
  return(list(train_aucs = train_aucs, test_aucs = test_aucs))
}

#' TO BE EDITED.
#' 
#' TO BE EDITED.
#'
#' @param fit TO BE EDITED.
#' @param extract TO BE EDITED.
#' @param X TO BE EDITED.
#' @param y TO BE EDITED.
#' @param n_samples TO BE EDITED.
#' @return TO BE EDITED.
#' @export
bootstrap_coefficients <- function(fit, extract, X, y, n_samples = 1000) {
  # Initialize containers
  coefs <- array(NA, dim = c(nrow(X), n_samples))

  # Loop over number of iterations
  for (i in 1:n_samples) {
    # Fit estimator with the training set
    model <- fit(X_train, y_train)

    # Extract and save coefficients
    coefs[i, ] <- extract(model)
  }
  
  return(coefs)
}

#' TO BE EDITED.
#' 
#' TO BE EDITED.
#'
#' @param fit TO BE EDITED.
#' @param predict TO BE EDITED.
#' @param X_train TO BE EDITED.
#' @param y_train TO BE EDITED.
#' @param X_test TO BE EDITED.
#' @param n_samples TO BE EDITED.
#' @return TO BE EDITED.
#' @export
bootstrap_predictions <- function(fit, predict, X_train, y_train, X_test, n_samples = 1000) {
  # Initialize containers
  y_train_predictions <- array(NA, dim = c(nrow(X_train), n_samples))
  y_test_predictions <- array(NA, dim = c(nrow(X_train), n_samples))

  # Loop over number of iterations
  for (i in 1:n_samples) {
    # Fit model with the training set
    model <- fit(X_train, y_train)
    
    # Save predictions
    y_train_predictions[i, ] <- predict(model, X_train)
    y_test_predictions[i, ] <- predict(model, X_test)
  }
  
  return(list(y_train_predictions = y_train_predictions, 
               y_test_predictions = y_test_predictions))
}
