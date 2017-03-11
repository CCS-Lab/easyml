#' Train test split.
#'
#' This will split the data into train and test.
#'
#' @param X A data.frame, the data to be resampled.
#' @param y A numeric vector with two classes, 0 and 1.
#' @param train_size A numeric vector of length one; specifies what proportion of the data should be used for the training data set. Defaults to 0.667.
#' @param foldid Not currently supported in this function.
#' @param random_state An integer vector of length one; specifies the seed to be used for the analysis. Defaults to NULL.
#' @return A boolean vector of length n_obs where TRUE represents that observation should be in the train set.
#' @family resample
#' @export
resample_simple_train_test_split <- function(X, y, train_size = 0.667, foldid = NULL, random_state = NULL) {
  # Set random state
  set_random_state(random_state)
  
  # Calculate number of observations
  n_obs <- length(y)
  
  # return a boolean vector of length n_obs where TRUE represents
  # that observation should be in the train set
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
#' @param X A data.frame, the data to be resampled.
#' @param y A numeric vector with two classes, 0 and 1.
#' @param train_size A numeric vector of length one; specifies what proportion of the data should be used for the training data set. Defaults to 0.667.
#' @param foldid A vector with length equal to \code{length(y)} which identifies cases belonging to the same fold. 
#' @param random_state An integer vector of length one; specifies the seed to be used for the analysis. Defaults to NULL.
#' @return A boolean vector of length n_obs where TRUE represents that observation should be in the train set.
#' @family resample
#' @export
resample_stratified_simple_train_test_split <- function(X, y, train_size = 0.667, foldid = NULL, random_state = NULL) {
  # Set random state
  set_random_state(random_state)
  
  # Calculate number of observations
  n_obs <- length(y)
  
  # Split data into list of X_train, X_test, y_train, y_test by stratum/foldid
  foldid <- factor(foldid)
  X_split_list <- split(X, foldid)
  y_split_list <- split(y, foldid)
  train_test_split_list <- Map(resample_simple_train_test_split, 
                               X_split_list, y_split_list)
  
  # Create X_train and X_test
  X_train <- lapply(train_test_split_list, function(x) x$X_train)
  X_train <- data.frame(do.call(rbind, X_train))
  X_test <- lapply(train_test_split_list, function(x) x$X_test)
  X_test <- data.frame(do.call(rbind, X_test))
  
  # Create y_train and y_test
  y_train <- Reduce(c, lapply(train_test_split_list, function(x) x$y_train))
  y_test <- Reduce(c, lapply(train_test_split_list, function(x) x$y_test))
  
  list(X_train = X_train, X_test = X_test, y_train = y_train, y_test = y_test)
}

#' Sample in equal proportion.
#'
#' This will sample in equal proportion.
#'
#' @param X A data.frame, the data to be resampled.
#' @param y A numeric vector with two classes, 0 and 1.
#' @param train_size A numeric vector of length one; specifies what proportion of the data should be used for the training data set. Defaults to 0.667.
#' @param foldid Not currently supported in this function.
#' @param random_state An integer vector of length one; specifies the seed to be used for the analysis. Defaults to NULL.
#' @return A boolean vector of length n_obs where TRUE represents that observation should be in the train set.
#' @family resample
#' @export
resample_stratified_class_train_test_split <- function(X, y, train_size = 0.667, foldid = NULL, random_state = NULL) {
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

#' Sample with respect to an identification vector
#'
#' This will sample the training and test sets so that case identifiers (e.g. subject ID's) are not shared across training and test sets.
#'
#' @param X A data.frame, the data to be resampled.
#' @param y A numeric vector with two classes, 0 and 1.
#' @param train_size A numeric vector of length one; specifies what proportion of the data should be used for the training data set. Defaults to 0.667.
#' @param foldid A vector with length equal to \code{length(y)} which identifies cases belonging to the same fold. 
#' @param random_state An integer vector of length one; specifies the seed to be used for the analysis. Defaults to NULL.
#' @return A boolean vector of length n_obs where TRUE represents that observation should be in the train set.
#' @family resample
#' @export
resample_fold_train_test_split <- function(X, y, train_size = 0.667, foldid = NULL, random_state = NULL) {
  # Catch error if foldid is not specified
  if (is.null(foldid))
    stop("'foldid' must be specified when calling 'resample_fold_train_test_split'!")
  
  # Set random state
  set_random_state(random_state)
  
  # Shuffle the unique values in the foldid vector
  unique_foldid <- sample(unique(foldid), replace = FALSE)
  
  # Transform unique_foldid indices to [0, 1] space
  foldid_idx <- seq_along(unique_foldid) / (length(unique_foldid))
  
  # Select unique foldid values with indices less than or equal to train_size
  train_ids <- unique_foldid[foldid_idx <= train_size]
  
  # Create mask with train_ids used to subset training samples
  mask <- foldid %in% train_ids
  
  # Separate training and test sets, return y's separately
  X_train <- X[mask,]
  X_test  <- X[!mask,]
  y_train <- y[mask]
  y_test  <- y[!mask]
  
  # Return the split data
  list(X_train = X_train, X_test = X_test, y_train = y_train, y_test = y_test)
}
