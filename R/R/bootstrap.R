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
bootstrap_coefficients <- function(fit_model, extract_coefficients, X, y, 
                                   n_samples = 1000, progress_bar = FALSE, 
                                   parallel = FALSE) {
  if (progress_bar) {
    print(paste0("Bootstrapping coefficients", ifelse(parallel, " in parallel:", ":")))
  }
  
  # Handle settings
  loop <- identify_looper(progress_bar = progress_bar, parallel = parallel)

  # Loop over number of iterations
  output <- loop(1:n_samples, function(i) {
    extract_coefficients(fit_model(X, y))
  })
  
  t(matrix(unlist(output), ncol = ncol(X) + 1, byrow = TRUE))
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
bootstrap_predictions <- function(fit_model, predict_model, X_train, y_train, X_test, 
                                  n_samples = 1000, progress_bar = FALSE, 
                                  parallel = FALSE) {
  if (progress_bar) {
    print(paste0("Bootstrapping predictions", ifelse(parallel, " in parallel:", ":")))
  }
  
  # Handle settings
  loop <- identify_looper(progress_bar = progress_bar, parallel = parallel)
  
  # Loop over number of iterations
  output <- loop(1:n_samples, function(i) {
    # Fit model with the training set
    results <- fit_model(X_train, y_train)
    
    # Save predictions
    list(y_train_predictions = predict_model(results, X_train), 
         y_test_predictions = predict_model(results, X_test))
  })
  
  y_train_predictions <- lapply(output, function(x) x$y_train_predictions)
  y_test_predictions <- lapply(output, function(x) x$y_test_predictions)
  
  y_train_predictions <- t(matrix(unlist(y_train_predictions), 
                                  ncol = nrow(X_train), byrow = TRUE))
  y_test_predictions <- t(matrix(unlist(y_test_predictions), 
                                 ncol = nrow(X_test), byrow = TRUE))
  
  list(y_train_predictions = y_train_predictions, 
       y_test_predictions = y_test_predictions)
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
bootstrap_metrics <- function(fit_model, predict_model, sampler, measure, X, y, 
                           n_divisions = 1000, n_iterations = 100, 
                           progress_bar = FALSE, parallel = FALSE) {
  if (progress_bar) {
    print(paste0("Bootstrapping metrics", ifelse(parallel, " in parallel:", ":")))
  }
  
  # Handle settings
  loop <- identify_looper(progress_bar = progress_bar, parallel = parallel)
  
  # Loop over number of divisions
  output_divisions <- loop(1:n_divisions, function(i) {
    # Split data
    split_data <- sampler(X, y)
    X_train <- split_data[["X_train"]]
    X_test <- split_data[["X_test"]]
    y_train <- split_data[["y_train"]]
    y_test <- split_data[["y_test"]]
    
    # Create temporary containers
    train_metrics <- numeric()
    test_metrics <- numeric()
    
    # Loop over number of iterations
    output_iterations <- lapply(1:n_iterations, function(j) {
      # Fit estimator with the training set
      results <- fit_model(X_train, y_train)
      
      # Generate scores for training and test sets
      y_train_predictions <- predict_model(results, X_train)
      y_test_predictions <- predict_model(results, X_test)
      
      # Save metrics
      train_metric <- measure(y_train, y_train_predictions)
      test_metric <- measure(y_test, y_test_predictions)
      list(train_metric = train_metric, test_metric = test_metric)
    })
    
    # Take average of metrics
    train_metrics <- unlist(lapply(output_iterations, function(x) x$train_metric))
    test_metrics <- unlist(lapply(output_iterations, function(x) x$test_metric))
    
    # Save mean of metrics
    list(mean_train_metric = mean(train_metrics), 
         mean_test_metric = mean(test_metrics))
  })
  
  mean_train_metrics <- unlist(lapply(output_divisions, function(x) x$mean_train_metric))
  mean_test_metrics <- unlist(lapply(output_divisions, function(x) x$mean_test_metric))
  
  list(mean_train_metrics = mean_train_metrics, 
       mean_test_metrics = mean_test_metrics)
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
bootstrap_aucs <- function(fit_model, predict_model, sampler, X, y, 
                           n_divisions = 1000, n_iterations = 100, 
                           progress_bar = FALSE, parallel = FALSE) {
  area_under_roc_curve <- function(y_true, y_pred) {
    as.numeric(pROC::roc(y_true, y_pred)$auc)
  }
  bootstrap_metrics(fit_model = fit_model, predict_model = predict_model, 
                    sampler = sampler, measure = area_under_roc_curve, 
                    X = X, y = y, n_divisions = n_divisions, 
                    n_iterations = n_iterations, progress_bar = progress_bar, 
                    parallel = parallel)
}

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
#' @param progress_bar TO BE EDITED.
#' @param parallel TO BE EDITED.
#' @return TO BE EDITED.
#' @export
bootstrap_mses <- function(fit_model, predict_model, sampler, X, y, 
                           n_divisions = 1000, n_iterations = 100, 
                           progress_bar = FALSE, parallel = FALSE) {
  bootstrap_metrics(fit_model = fit_model, predict_model = predict_model, 
                    sampler = sampler, measure = scorer::mean_squared_error, 
                    X = X, y = y, n_divisions = n_divisions, 
                    n_iterations = n_iterations, progress_bar = progress_bar, 
                    parallel = parallel)
}
