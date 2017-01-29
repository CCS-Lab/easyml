#' Fit penalized gaussian regression model.
#' 
#' @param X input matrix, of dimension nobs x nvars; each row is an observation vector. Can be in sparse matrix format (inherit from class "sparseMatrix" as in package Matrix; not yet available for family="cox")
#' @param y response variable. Quantitative for family="gaussian", or family="poisson" (non-negative counts). For family="binomial" should be either a factor with two levels, or a two-column matrix of counts or proportions (the second column is treated as the target class; for a factor, the last level in alphabetical order is the target class). For family="multinomial", can be a nc>=2 level factor, or a matrix with nc columns of counts or proportions. For either "binomial" or "multinomial", if y is presented as a vector, it will be coerced into a factor. For family="cox", y should be a two-column matrix with columns named 'time' and 'status'. The latter is a binary variable, with '1' indicating death, and '0' indicating right censored. The function Surv() in package survival produces such a matrix. For family="mgaussian", y is a matrix of quantitative responses.
#' @param ... Arguments to be passed to \code{\link[glmnet]{glmnet}}. See that function's documentation for more details.
#' @return TO BE EDITED.
#' @export
glmnet_fit_model_gaussian <- function(X, y, ...) {
  X <- as.matrix(X)
  model <- glmnet::glmnet(X, y, family = "gaussian", ...)
  cv_model <- glmnet::cv.glmnet(X, y, family = "gaussian", ...)
  list(model = model, cv_model = cv_model)
}

#' Fit penalized binomial regression model.
#'
#' @param X input matrix, of dimension nobs x nvars; each row is an observation vector. Can be in sparse matrix format (inherit from class "sparseMatrix" as in package Matrix; not yet available for family="cox")
#' @param y response variable. Quantitative for family="gaussian", or family="poisson" (non-negative counts). For family="binomial" should be either a factor with two levels, or a two-column matrix of counts or proportions (the second column is treated as the target class; for a factor, the last level in alphabetical order is the target class). For family="multinomial", can be a nc>=2 level factor, or a matrix with nc columns of counts or proportions. For either "binomial" or "multinomial", if y is presented as a vector, it will be coerced into a factor. For family="cox", y should be a two-column matrix with columns named 'time' and 'status'. The latter is a binary variable, with '1' indicating death, and '0' indicating right censored. The function Surv() in package survival produces such a matrix. For family="mgaussian", y is a matrix of quantitative responses.
#' @param ... Arguments to be passed to \code{\link[glmnet]{glmnet}}. See that function's documentation for more details.
#' @return TO BE EDITED.
#' @export
glmnet_fit_model_binomial <- function(X, y, ...) {
  X <- as.matrix(X)
  model <- glmnet::glmnet(X, y, family = "binomial", ...)
  cv_model <- glmnet::cv.glmnet(X, y, family = "binomial", ...)
  list(model = model, cv_model = cv_model)
}

#' Extract coefficients from a penalized regression model.
#' 
#' @param results TO BE EDITED.
#' @return TO BE EDITED.
#' @export
glmnet_extract_coefficients <- function(results) {
  model <- results[["model"]]
  cv_model <- results[["cv_model"]]
  coefs <- stats::coef(model, s = cv_model$lambda.min)
  .data <- data.frame(t(as.matrix(as.numeric(coefs), nrow = 1)))
  colnames(.data) <- rownames(coefs)
  .data
}

#' Predict values for a penalized regression model.
#' 
#' @param results TO BE EDITED.
#' @param newx TO BE EDITED.
#' @return TO BE EDITED.
#' @export
glmnet_predict_model <- function(results, newx) {
  newx <- as.matrix(newx)
  model <- results[["model"]]
  cv_model <- results[["cv_model"]]
  stats::predict(model, newx = newx, s = cv_model$lambda.min, type = "response")
}

#' Easily build and evaluate a penalized regression model.
#'
#' @param ... Arguments to be passed to \code{\link[glmnet]{glmnet}}. See that function's documentation for more details.
#' @inheritParams easy_analysis
#' @return TO BE EDITED.
#' @examples 
#' library(easyml) # https://github.com/CCS-Lab/easyml
#' 
#' # Gaussian
#' data("prostate", package = "easyml")
#' results <- easy_glmnet(prostate, "lpsa", 
#'                        n_samples = 10L, n_divisions = 10L, 
#'                        n_iterations = 2L, random_state = 12345L, 
#'                        n_core = 1L, alpha = 1.0)
#' 
#' # Binomial
#' data("cocaine_dependence", package = "easyml")
#' results <- easy_glmnet(cocaine_dependence, "diagnosis", 
#'                        family = "binomial", 
#'                        exclude_variables = c("subject"), 
#'                        categorical_variables = c("male"), 
#'                        preprocess = preprocess_scaler, 
#'                        n_samples = 10L, n_divisions = 10L, 
#'                        n_iterations = 2L, random_state = 12345L, 
#'                        n_core = 1L, alpha = 1.0)
#' @export
easy_glmnet <- function(.data, dependent_variable, family = "gaussian", 
                        resample = NULL, preprocess = NULL, measure = NULL, 
                        exclude_variables = NULL, categorical_variables = NULL, 
                        train_size = 0.667, survival_rate_cutoff = 0.05, 
                        n_samples = 1000L, n_divisions = 1000L, 
                        n_iterations = 10L, random_state = NULL, 
                        progress_bar = TRUE, n_core = 1L, ...) {
  easy_analysis(.data, dependent_variable, algorithm = "glmnet", 
                family = family, resample = resample, 
                preprocess = preprocess, measure = measure, 
                exclude_variables = exclude_variables, 
                categorical_variables = categorical_variables,  
                train_size = train_size, 
                survival_rate_cutoff = survival_rate_cutoff, 
                n_samples = n_samples, n_divisions = n_divisions, 
                n_iterations = n_iterations, random_state = random_state, 
                progress_bar = progress_bar, n_core = n_core, ...)
}
