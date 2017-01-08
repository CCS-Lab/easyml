#' Train test split.
#'
#' This will split the data into train and test.
#'
#' @param X TO BE EDITED.
#' @param y A numeric vector with two classes, 0 and 1.
#' @param train_size The proportoin of data in training set.
#' @param random_state An integer to seed the random number generator.
#' @return A boolean vector of length n_obs where TRUE represents that observation should be in the train set.
#' @export
resample_simple_train_test_split <- function(X, y, train_size = 0.667, random_state = NULL) {
  # Set random state
  set_random_state(random_state)

  # return a boolean vector of length n_obs where TRUE represents
  # that observation should be in the train set
  n_obs <- length(y)
  index <- sample(1:n_obs, floor(n_obs * train_size))
  mask <- 1:n_obs %in% index
  
  # Create splits
  X_train <- X[mask, ]
  X_test <- X[!mask, ]
  y_train <- y[mask]
  y_test <- y[!mask]

  list(X_train = X_train, X_test = X_test, y_train = y_train, y_test = y_test)
}

#' Sample in equal proportion.
#'
#' This will sample in equal proportion.
#'
#' @param X TO BE EDITED.
#' @param y A numeric vector with two classes, 0 and 1.
#' @param train_size The proportoin of data in training set.
#' @param random_state An integer to seed the random number generator.
#' @return A boolean vector of length n_obs where TRUE represents that observation should be in the train set.
#' @export
resample_stratified_train_test_split <- function(X, y, train_size = 0.667, random_state = NULL) {
  # Set random state
  set_random_state(random_state)
  
  # calculate number of observations
  n_obs <- length(y)
  
  # identify index number for class1 and class2
  index_class1 <- which(y == 0)
  index_class2 <- which(y == 1)
  
  # calculate  number of class1 and class2 observations
  n_class1 <- length(index_class1)
  n_class2 <- length(index_class2)
  
  # calculate number of class1 and class2 observations in the train set
  n_class1_train <- round(n_class1 * train_size)
  n_class2_train <- round(n_class2 * train_size)
  
  # generate indices for class1 and class2 observations in the train set
  index_class1_train <- sample(index_class1, n_class1_train, replace = FALSE)
  index_class2_train <- sample(index_class2, n_class2_train, replace = FALSE)
  index_train <- c(index_class1_train, index_class2_train)
  
  # return a boolean vector of length n_obs where TRUE represents
  # that observation should be in the train set
  mask <- 1:n_obs %in% index_train
  
  # Create splits
  X_train <- X[mask, ]
  X_test <- X[!mask, ]
  y_train <- y[mask]
  y_test <- y[!mask]
  
  list(X_train = X_train, X_test = X_test, y_train = y_train, y_test = y_test)
}
