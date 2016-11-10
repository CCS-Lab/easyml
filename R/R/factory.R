#' TO BE EDITED.
#' 
#' TO BE EDITED.
#'
#' @return TO BE EDITED.
#' @export
easy_glmnet <- function(data, dependent_variable = None, family = "gaussian", 
                        exclude_variables = NULL, train_size = 0.667,
                        n_divisions = 1000, n_iterations = 10, 
                        n_samples = 1000, out_directory = '.', random_state = None,
                        ...) {
  # args <- list(...)
  # Create fit, extract, and predict functions
  fit <- function(X, y) {
    model <- glmnet::glmnet(X, y, family = family)
    cv_model <- glmnet::glmnet(X, y, family = family)
    list(model = model, cv_model = cv_model)
  }
  
  extract <- function(results) {
    model <- results[["model"]]
    cv_model <- results[["cv_model"]]
    coef(model, s = cv_model$lambda.min, type="coefficient")
  }
  
  predict <- function(results) {
    model <- results[["model"]]
    cv_model <- results[["cv_model"]]
    predict(model, s = cv_model$lambda.1se, type="coefficient", type="response")
  }
  
  # Process the data
  data <- process_data(data, dependent_variable = dependent_variable, exclude_variables = exclude_variables)
  X <- as.matrix(data[["X"]])
  y <- data[["y"]]

  # Bootstrap coefficients
  coefs <- bootstrap_coefficients(fit, extract, X, y)

  # Process coefficients
  betas <- process_coefficients(coefs)
  plot_betas(betas)
  ggsave("betas.png")

  # Split data
  mask <- sample_equal_proportion(y, proportion=train_size, random_state=random_state)
  y_train <- y[mask]
  y_test <- y[!mask]
  X_train <- X[mask, ]
  X_test <- X[!mask, ]

  # Bootstrap predictions
  predictions <- bootstrap_predictions(fit, predict, X_train, y_train, X_test, n_samples = n_samples)
  y_train_scores <- predictions[["y_train_scores"]]
  y_test_scores <- predictions[["y_test_scores"]]

  # Generate scores for training and test sets
  y_train_scores_mean <- apply(y_train_scores, 2, mean)
  y_test_scores_mean <- apply(y_test_scores, 2, mean)

  # Compute ROC curve and ROC area for training
  plot_roc_curve(y_train, y_train_scores_mean)
  ggsave("train_roc_curve.png")

  # Compute ROC curve and ROC area for test
  plot_roc_curve(y_test, y_test_scores_mean)
  ggsave("test_roc_curve.png")

  # Bootstrap training and test AUCS
  aucs <- bootstrap_aucs(model, X, y, n_divisions = n_divisions, n_iterations = n_iterations)
  train_aucs <- aucs[["train_aucs"]]
  test_aucs <- aucs[["test_aucs"]]

  # Plot histogram of training AUCS
  plot_auc_histogram(train_aucs)
  ggsave("train_auc_distribution.png")

  # Plot histogram of test AUCS
  plot_auc_histogram(test_aucs)
  ggsave("test_auc_distribution.png")

  invisible()
}
