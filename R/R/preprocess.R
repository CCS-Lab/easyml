#' TO BE EDITED.
#' 
#' TO BE EDITED.
#'
#' @return TO BE EDITED.
#' @export
preprocess_identity <- function(.data, categorical_variables = NULL) {
  .data
}

#' TO BE EDITED.
#' 
#' TO BE EDITED.
#'
#' @return TO BE EDITED.
#' @export
preprocess_scaler <- function(.data, categorical_variables = NULL) {
  if (length(.data) == 1) {
    # Handle case of list(X)
    X <- .data[["X"]]
    if (is.null(categorical_variables)) {
      # No categorical variables
      X_output <- data.frame(scale(X))
      output <- list(X = X_output)
    } else {
      # Categorical variables
      mask <- (colnames(X) == categorical_variables)
      X_categorical <- X[, mask, drop = FALSE]
      X_numerical <- X[, !mask, drop = FALSE]
      X_standardized <- data.frame(scale(X_numerical))
      X_output <- cbind(X_categorical, X_numerical)
      output <- list(X = X_output)
    }
  } else if (length(.data) == 2) {
    # Handle case of list(X_train, X_test)
    X_train <- .data[["X_train"]]
    X_test <- .data[["X_test"]]
    if (is.null(categorical_variables)) {
      # No categorical variables
      # scale train
      X_train_scaled <- scale(X_train)
      scaled_center <- attr(X_train_scaled, "scaled:center")
      scaled_scale <- attr(X_train_scaled, "scaled:scale")
      
      # scale test
      X_test_scaled <- scale(X_test, scaled_center, scaled_scale)
      
      # output data
      X_train_output <- data.frame(X_train_scaled)
      X_test_output <- data.frame(X_test_scaled)
      output <- list(X_train = X_train_output, X_test = X_test_output)
    } else {
      # Categorical variables
      mask <- (colnames(X_train) == categorical_variables)
      X_train_categorical <- X_train[, mask, drop = FALSE]
      X_train_numerical <- X_train[, !mask, drop = FALSE]
      X_test_categorical <- X_test[, mask, drop = FALSE]
      X_test_numerical <- X_test[, !mask, drop = FALSE]
      
      # scale train
      X_train_scaled <- scale(X_train_numerical)
      scaled_center <- attr(X_train_scaled, "scaled:center")
      scaled_scale <- attr(X_train_scaled, "scaled:scale")
      
      # scale test
      X_test_scaled <- scale(X_test_numerical, scaled_center, scaled_scale)
      
      # output data
      X_train_output <- cbind(X_train_categorical, data.frame(X_train_scaled))
      X_test_output <- cbind(X_test_categorical, data.frame(X_test_scaled))
      output <- list(X_train = X_train_output, X_test = X_test_output)
    }
  } else {
    stop("Value error: length(.data) is not 1 or 2.")
  }
  
  output
}
