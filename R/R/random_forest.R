#' Fit random forest gaussian regression model.
#' 
#' @param X a data frame or a matrix of predictors, or a formula describing the model to be fitted (for the print method, an randomForest object).
#' @param y A response vector. If a factor, classification is assumed, otherwise regression is assumed. If omitted, randomForest will run in unsupervised mode.
#' @param ... Arguments to be passed to \code{\link[randomForest]{randomForest}}. See that function's documentation for more details.
#' @return TO BE EDITED.
#' @export
random_forest_fit_model_gaussian <- function(X, y, ...) {
  X <- as.matrix(X)
  randomForest::randomForest(X, y, ...)
}

#' Fit random forest binomial regression model.
#' 
#' @param X a data frame or a matrix of predictors, or a formula describing the model to be fitted (for the print method, an randomForest object).
#' @param y A response vector. If a factor, classification is assumed, otherwise regression is assumed. If omitted, randomForest will run in unsupervised mode.
#' @param ... Arguments to be passed to \code{\link[randomForest]{randomForest}}. See that function's documentation for more details.
#' @return TO BE EDITED.
#' @export
random_forest_fit_model_binomial <- function(X, y, ...) {
  X <- as.matrix(X)
  y <- factor(y)
  randomForest::randomForest(X, y, ...)
}

#' Predict values for a random forest regression model.
#' 
#' @param results TO BE EDITED.
#' @param newx TO BE EDITED.
#' @return TO BE EDITED.
#' @export
random_forest_predict_model <- function(results, newx) {
  as.numeric(stats::predict(results, newdata = newx))
}

#' Easily build and evaluate a random forest regression model.
#' 
#' @param ... Arguments to be passed to \code{\link[randomForest]{randomForest}}. See that function's documentation for more details.
#' @inheritParams easy_analysis
#' @return TO BE EDITED.
#' @family recipes
#' @examples 
#' library(easyml) # https://github.com/CCS-Lab/easyml
#' 
#' # Gaussian
#' data("prostate", package = "easyml")
#' results <- easy_random_forest(prostate, "lpsa", 
#'                               n_samples = 10L, 
#'                               n_divisions = 10L, 
#'                               n_iterations = 2L, 
#'                               random_state = 12345L, n_core = 1L)
#' 
#' # Binomial
#' data("cocaine_dependence", package = "easyml")
#' results <- easy_random_forest(cocaine_dependence, "diagnosis", 
#'                               family = "binomial", 
#'                               exclude_variables = c("subject"),
#'                               categorical_variables = c("male"),
#'                               n_samples = 10L, 
#'                               n_divisions = 10L, 
#'                               n_iterations = 2L, 
#'                               random_state = 12345L, n_core = 1L)
#' @export
easy_random_forest <- function(.data, dependent_variable, family = "gaussian", 
                               resample = NULL, preprocess = NULL, measure = NULL, 
                               exclude_variables = NULL, categorical_variables = NULL, 
                               train_size = 0.667, 
                               n_samples = 1000L, n_divisions = 1000L, 
                               n_iterations = 10L, random_state = NULL, 
                               progress_bar = TRUE, n_core = 1L, ...) {
  easy_analysis(.data, dependent_variable, algorithm = "random_forest", 
                family = family, resample = resample, 
                preprocess = preprocess, measure = measure, 
                exclude_variables = exclude_variables, 
                categorical_variables = categorical_variables,  
                train_size = train_size, 
                n_samples = n_samples, n_divisions = n_divisions, 
                n_iterations = n_iterations, random_state = random_state, 
                progress_bar = progress_bar, n_core = n_core, ...)
}
