#' Fit penalized gaussian regression model.
#' 
#' @param X input matrix, of dimension nobs x nvars; each row is an observation vector. Can be in sparse matrix format (inherit from class "sparseMatrix" as in package Matrix; not yet available for family="cox")
#' @param y response variable. Quantitative for family="gaussian", or family="poisson" (non-negative counts). For family="binomial" should be either a factor with two levels, or a two-column matrix of counts or proportions (the second column is treated as the target class; for a factor, the last level in alphabetical order is the target class). For family="multinomial", can be a nc>=2 level factor, or a matrix with nc columns of counts or proportions. For either "binomial" or "multinomial", if y is presented as a vector, it will be coerced into a factor. For family="cox", y should be a two-column matrix with columns named 'time' and 'status'. The latter is a binary variable, with '1' indicating death, and '0' indicating right censored. The function Surv() in package survival produces such a matrix. For family="mgaussian", y is a matrix of quantitative responses.
#' @param ... Arguments to be passed to \code{\link[glmnet]{glmnet}}. See that function's documentation for more details.
#' @return A list, the model and the cross validated model.
#' @export
glmnet_fit_model_gaussian <- function(X, y, ...) {
  # capture additional arguments
  kwargs <- list(...)
  
  # process kwargs
  kwargs[["family"]] <- "gaussian"
  kwargs[["standardize"]] <- FALSE
  kwargs[["x"]] <- as.matrix(X)
  kwargs[["y"]] <- y
  
  # build cv_model
  cv_model <- do.call(glmnet::cv.glmnet, kwargs)
  
  # build model
  kwargs[["nfolds"]] <- NULL
  model <- do.call(glmnet::glmnet, kwargs)
  
  # write output
  list(model = model, cv_model = cv_model)
}

#' Fit penalized binomial regression model.
#'
#' @param X input matrix, of dimension nobs x nvars; each row is an observation vector. Can be in sparse matrix format (inherit from class "sparseMatrix" as in package Matrix; not yet available for family="cox")
#' @param y response variable. Quantitative for family="gaussian", or family="poisson" (non-negative counts). For family="binomial" should be either a factor with two levels, or a two-column matrix of counts or proportions (the second column is treated as the target class; for a factor, the last level in alphabetical order is the target class). For family="multinomial", can be a nc>=2 level factor, or a matrix with nc columns of counts or proportions. For either "binomial" or "multinomial", if y is presented as a vector, it will be coerced into a factor. For family="cox", y should be a two-column matrix with columns named 'time' and 'status'. The latter is a binary variable, with '1' indicating death, and '0' indicating right censored. The function Surv() in package survival produces such a matrix. For family="mgaussian", y is a matrix of quantitative responses.
#' @param ... Arguments to be passed to \code{\link[glmnet]{glmnet}}. See that function's documentation for more details.
#' @return A list, the model and the cross validated model.
#' @export
glmnet_fit_model_binomial <- function(X, y, ...) {
  # capture additional arguments
  kwargs <- list(...)
  
  # process kwargs
  kwargs[["family"]] <- "binomial"
  kwargs[["standardize"]] <- FALSE
  kwargs[["x"]] <- as.matrix(X)
  kwargs[["y"]] <- y
  
  # build cv_model
  cv_model <- do.call(glmnet::cv.glmnet, kwargs)
  
  # build model
  kwargs[["nfolds"]] <- NULL
  model <- do.call( glmnet::glmnet, kwargs)
  
  # write output
  list(model = model, cv_model = cv_model)
}

#' Extract coefficients from a penalized regression model.
#' 
#' @param results The results of \code{\link{glmnet_fit_model_gaussian}} or \code{\link{glmnet_fit_model_binomial}}.
#' @return A data.frame of replicated penalized regression coefficients.
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
#' @param results The results of \code{\link{glmnet_fit_model_gaussian}} or \code{\link{glmnet_fit_model_binomial}}.
#' @param newx A data.frame, the new data to use for predictions.
#' @return A vector, the predicted values for a penalized regression model using the new data.
#' @export
glmnet_predict_model <- function(results, newx = NULL) {
  newx <- as.matrix(newx)
  model <- results[["model"]]
  cv_model <- results[["cv_model"]]
  stats::predict(model, newx = newx, s = cv_model$lambda.min, type = "response")
}

#' Easily build and evaluate a penalized regression model.
#'
#' @param ... Arguments to be passed to \code{\link[glmnet]{glmnet}} or \code{\link[glmnet]{cv.glmnet}}. See those functions' documentation for more details on possible arguments and what they mean. Examples of applicable arguments are \code{alpha}, \code{nlambda}, \code{nlambda.min.ratio}, \code{lambda}, \code{standardize}, \code{intercept}, \code{thresh}, \code{dfmax}, \code{pmax}, \code{exclude}, \code{penalty.factor}, \code{lower.limits}, \code{upper.limits}, \code{maxit}, and \code{standardize.response} for \code{\link[glmnet]{glmnet}} and \code{weights}, \code{offset}, \code{lambda}, \code{type.measure}, \code{nfolds}, \code{foldid}, \code{grouped}, \code{keep}, \code{parallel} for \code{\link[glmnet]{cv.glmnet}}.
#' @inheritParams easy_analysis
#' @return A list with the following values:
#' \describe{
#' \item{resample}{A function; the function for resampling the data.}
#' \item{preprocess}{A function; the function for preprocessing the data.}
#' \item{measure}{A function; the function for measuring the results.}
#' \item{fit_model}{A function; the function for fitting the model to the data.}
#' \item{extract_coefficients}{A function; the function for extracting coefficients from the model.}
#' \item{predict_model}{A function; the function for generating predictions on new data from the model.}
#' \item{plot_predictions}{A function; the function for plotting predictions generated by the model.}
#' \item{plot_metrics}{A function; the function for plotting metrics generated by scoring the model.}
#' \item{data}{A data.frame; the original data.}
#' \item{X}{A data.frame; the full dataset to be used for modeling.}
#' \item{y}{A vector; the full response variable to be used for modeling.}
#' \item{X_train}{A data.frame; the train dataset to be used for modeling.}
#' \item{X_test}{A data.frame; the test dataset to be used for modeling.}
#' \item{y_train}{A vector; the train response variable to be used for modeling.}
#' \item{y_test}{A vector; the test response variable to be used for modeling.}
#' \item{coefficients}{A (n_variables, n_samples) matrix; the replicated coefficients.}
#' \item{coefficients_processed}{A data.frame; the coefficients after being processed.}
#' \item{plot_coefficients_processed}{A ggplot object; the plot of the processed coefficients.}
#' \item{predictions_train}{A (nrow(X_train), n_samples) matrix; the train predictions.}
#' \item{predictions_test}{A (nrow(X_test), n_samples) matrix; the test predictions.}
#' \item{predictions_train_mean}{A vector; the mean train predictions.}
#' \item{predictions_test_mean}{A vector; the mean test predictions.}
#' \item{plot_predictions_train_mean}{A ggplot object; the plot of the mean train predictions.}
#' \item{plot_predictions_test_mean}{A ggplot object; the plot of the mean test predictions.}
#' \item{metrics_train_mean}{A vector of length n_divisions; the mean train metrics.}
#' \item{metrics_test_mean}{A vector of length n_divisions; the mean test metrics.}
#' \item{plot_metrics_train_mean}{A ggplot object; the plot of the mean train metrics.}
#' \item{plot_metrics_test_mean}{A ggplot object; the plot of the mean test metrics.}
#' }
#' @family recipes
#' @examples 
#' library(easyml) # https://github.com/CCS-Lab/easyml
#' 
#' # Gaussian
#' data("prostate", package = "easyml")
#' results <- easy_glmnet(prostate, "lpsa", 
#'                        n_samples = 10, n_divisions = 10, 
#'                        n_iterations = 2, random_state = 12345, 
#'                        n_core = 1, alpha = 1.0)
#' 
#' # Binomial
#' data("cocaine_dependence", package = "easyml")
#' results <- easy_glmnet(cocaine_dependence, "diagnosis", 
#'                        family = "binomial", 
#'                        exclude_variables = c("subject"), 
#'                        categorical_variables = c("male"), 
#'                        preprocess = preprocess_scale, 
#'                        n_samples = 10, n_divisions = 10, 
#'                        n_iterations = 2, random_state = 12345, 
#'                        n_core = 1, alpha = 1.0)
#' @export
easy_glmnet <- function(.data, dependent_variable, family = "gaussian", 
                        resample = NULL, preprocess = NULL, measure = NULL, 
                        exclude_variables = NULL, categorical_variables = NULL, 
                        train_size = 0.667, foldid = NULL, 
                        survival_rate_cutoff = 0.05, 
                        n_samples = 1000, n_divisions = 1000, 
                        n_iterations = 10, random_state = NULL, 
                        progress_bar = TRUE, n_core = 1, ...) {
  easy_analysis(.data, dependent_variable, algorithm = "glmnet", 
                family = family, resample = resample, 
                preprocess = preprocess, measure = measure, 
                exclude_variables = exclude_variables, 
                categorical_variables = categorical_variables,  
                train_size = train_size, foldid = foldid,  
                survival_rate_cutoff = survival_rate_cutoff, 
                n_samples = n_samples, n_divisions = n_divisions, 
                n_iterations = n_iterations, random_state = random_state, 
                progress_bar = progress_bar, n_core = n_core, ...)
}
