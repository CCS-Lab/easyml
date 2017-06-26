#' Preprocess data by leaving it exactly the way it is.
#' 
#' This function is the same as the identity function. Anything passed
#' to it is returned untouched.
#'
#' @param .data A data.frame; the data to be analyzed.
#' @param categorical_variables A logical vector; each value TRUE indicates that column in the data.frame is a categorical variable. Defaults to NULL.
#' @return A list, containing one or two data.frames.
#' @family preprocess
#' @export
preprocess_identity <- function(.data, categorical_variables = NULL) {
  .data
}

#' Preprocess data by scaling it.
#' 
#' This function takes either a data.frame or a list of data.frames. In 
#' the event of the first, this function takes the dataset and will
#' scale each column that is not categorical such that that column
#' has zero mean and unit variance. In the event of the second, this 
#' function takes the training dataset and will identify the parameters 
#' needed to scale the training dataset such that each column that 
#' is not categorical has zero mean and unit variance, and then will 
#' apply those parameters to each column in the testing dataset 
#' that is not categorical.
#'
#' @param .data A data.frame; the data to be analyzed.
#' @param categorical_variables A logical vector; each value TRUE indicates that column in the data.frame is a categorical variable. Defaults to NULL.
#' @return A list, containing one or two data.frames.
#' @family preprocess
#' @export
preprocess_scale <- function(.data, categorical_variables = NULL) {
  mask <- categorical_variables
  if (length(.data) == 1) {
    # Handle case of list(X)
    X <- .data[["X"]]
    if (is.null(mask)) {
      # No categorical variables
      X_output <- data.frame(scale(X))
      output <- list(X = X_output)
    } else {
      # Categorical variables
      X_categorical <- X[, mask, drop = FALSE]
      X_numerical <- X[, !mask, drop = FALSE]
      X_standardized <- data.frame(scale(X_numerical))
      X_output <- cbind(X_categorical, X_standardized)
      output <- list(X = X_output)
    }
  } else if (length(.data) == 2) {
    # Handle case of list(X_train, X_test)
    X_train <- .data[["X_train"]]
    X_test <- .data[["X_test"]]
    if (is.null(mask)) {
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
