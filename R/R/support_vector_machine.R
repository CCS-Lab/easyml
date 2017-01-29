#' Fit support vector machine gaussian regression model.
#'
#' @param X a data matrix, a vector, or a sparse matrix (object of class Matrix provided by the Matrix package, or of class matrix.csr provided by the SparseM package, or of class simple_triplet_matrix provided by the slam package).
#' @param y a response vector with one label for each row/component of x. Can be either a factor (for classification tasks) or a numeric vector (for regression).
#' @param ... Arguments to be passed to \code{\link[e1071]{svm}}. See that function's documentation for more details.
#' @return TO BE EDITED.
#' @export
support_vector_machine_fit_model_gaussian <- function(X, y, ...) {
  X <- as.matrix(X)
  e1071::svm(X, y, scale = FALSE, type = "nu-regression", ...)
}

#' Fit support vector machine binomial regression model.
#' 
#' @param X a data matrix, a vector, or a sparse matrix (object of class Matrix provided by the Matrix package, or of class matrix.csr provided by the SparseM package, or of class simple_triplet_matrix provided by the slam package).
#' @param y a response vector with one label for each row/component of x. Can be either a factor (for classification tasks) or a numeric vector (for regression).
#' @param ... Arguments to be passed to \code{\link[e1071]{svm}}. See that function's documentation for more details.
#' @return TO BE EDITED.
#' @export
support_vector_machine_fit_model_binomial <- function(X, y, ...) {
  X <- as.matrix(X)
  y <- factor(y)
  e1071::svm(X, y, scale = FALSE, type = "C-classification", ...)
}

#' Predict values for a support vector machine regression model.
#' 
#' @param results TO BE EDITED.
#' @param newx TO BE EDITED.
#' @return TO BE EDITED.
#' @export
support_vector_machine_predict_model <- function(results, newx) {
  as.numeric(stats::predict(results, newdata = newx))
}

#' Easily build and evaluate a support vector machine regression model.
#'
#'@param ... Arguments to be passed to \code{\link[e1071]{svm}}. See that function's documentation for more details.
#' @inheritParams easy_analysis
#' @return TO BE EDITED.
#' @family recipes
#' @examples 
#' library(easyml) # https://github.com/CCS-Lab/easyml
#' 
#' # Gaussian
#' data("prostate", package = "easyml")
#' results <- easy_support_vector_machine(prostate, "lpsa", 
#'                                        n_samples = 10L, 
#'                                        n_divisions = 10L, 
#'                                        n_iterations = 2L, 
#'                                        random_state = 1L, n_core = 1L)
#' 
#' # Binomial
#' data("cocaine_dependence", package = "easyml")
#' results <- easy_support_vector_machine(cocaine_dependence, "diagnosis", 
#'                                        family = "binomial", 
#'                                        preprocesss = preprocess_scaler, 
#'                                        exclude_variables = c("subject"), 
#'                                        categorical_variables = c("male"), 
#'                                        n_samples = 10L, 
#'                                        n_divisions = 10L, 
#'                                        n_iterations = 2L, 
#'                                        random_state = 1L, n_core = 1L)
#' @export
easy_support_vector_machine <- function(.data, dependent_variable, family = "gaussian", 
                     resample = NULL, preprocess = NULL, measure = NULL, 
                     exclude_variables = NULL, categorical_variables = NULL, 
                     train_size = 0.667, 
                     n_samples = 1000L, n_divisions = 1000L, 
                     n_iterations = 10L, random_state = NULL, 
                     progress_bar = TRUE, n_core = 1L, ...) {
  easy_analysis(.data, dependent_variable, algorithm = "support_vector_machine", 
                family = family, resample = resample, 
                preprocess = preprocess, measure = measure, 
                exclude_variables = exclude_variables, 
                categorical_variables = categorical_variables,  
                train_size = train_size, 
                n_samples = n_samples, n_divisions = n_divisions, 
                n_iterations = n_iterations, random_state = random_state, 
                progress_bar = progress_bar, n_core = n_core, ...)
}
