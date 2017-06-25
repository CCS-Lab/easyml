#' Fit a penalized regression model.
#' 
#' This function wraps the procedure for fitting a 
#' glmnet model and makes it accessible 
#' to the easyml core framework.
#' 
#' @param object A list of class \code{easy_glmnet}.
#' @return A list of class \code{easy_glmnet}.
#' @export
fit_model.easy_glmnet <- function(object) {
  # set model arguments
  model_args <- object[["model_args"]]

  # process model_args
  model_args[["family"]] <- object[["family"]]
  if (!is.null(model_args[["standardize"]])) {
    model_args[["standardize"]] <- FALSE
  }
  model_args[["x"]] <- as.matrix(object[["X"]])
  model_args[["y"]] <- object[["y"]]

  # build model_cv
  model_cv <- do.call(glmnet::cv.glmnet, model_args)
  object[["model_cv_args"]] <- model_args
  object[["model_cv"]] <- model_cv

  # build model
  model_args[["nfolds"]] <- NULL
  model <- do.call(glmnet::glmnet, model_args)
  object[["model_args"]] <- model_args
  object[["model"]] <- model

  # write output
  object
}

#' Predict values for a penalized regression model.
#' 
#' This function wraps the procedure for predicting values from 
#' a glmnet model and makes it accessible 
#' to the easyml core framework.
#' 
#' @param object A list of class \code{easy_glmnet}.
#' @param newx A data.frame, the new data to use for predictions.
#' @return A vector, the predicted values using the new data.
#' @export
predict_model.easy_glmnet <- function(object, newx = NULL) {
  newx <- as.matrix(newx)
  model <- object[["model"]]
  model_cv <- object[["model_cv"]]
  s <- model_cv$lambda.min
  preds <- stats::predict(model, newx = newx, s = s, type = "response")
  preds
}

#' Extract coefficients from a penalized regression model.
#' 
#' This function wraps the procedure for extracting coefficients from a 
#' glmnet model and makes it accessible 
#' to the easyml core framework.
#' 
#' @param object A list of class \code{easy_glmnet}.
#' @return A data.frame, the replicated penalized regression coefficients.
#' @export
extract_coefficients.easy_glmnet <- function(object) {
  model <- object[["model"]]
  model_cv <- object[["model_cv"]]
  coefs <- stats::coef(model, s = model_cv$lambda.min)
  coefs_df <- data.frame(t(as.matrix(as.numeric(coefs), nrow = 1)))
  colnames(coefs_df) <- rownames(coefs)
  coefs_df
}

#' Easily build and evaluate a penalized regression model.
#' 
#' This function wraps the easyml core framework, allowing a user 
#' to easily run the easyml methodology for a glmnet
#' model.
#'
#' @inheritParams easy_analysis
#' @return A list of class \code{easy_glmnet}.
#' @family recipes
#' @examples 
#' \dontrun{
#' library(easyml) # https://github.com/CCS-Lab/easyml
#' 
#' # Gaussian
#' data("prostate", package = "easyml")
#' results <- easy_glmnet(prostate, "lpsa", 
#'                        n_samples = 10, n_divisions = 10, 
#'                        n_iterations = 2, random_state = 12345, 
#'                        n_core = 1, model_args = list(alpha = 1.0))
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
#'                        n_core = 1, model_args = list(alpha = 1.0))
#' }
#' @export
easy_glmnet <- function(.data, dependent_variable, family = "gaussian", 
                        resample = NULL, preprocess = preprocess_scale, 
                        measure = NULL, exclude_variables = NULL, 
                        categorical_variables = NULL, 
                        train_size = 0.667, foldid = NULL, 
                        survival_rate_cutoff = 0.05, 
                        n_samples = 1000, n_divisions = 1000, 
                        n_iterations = 10, random_state = NULL, 
                        progress_bar = TRUE, n_core = 1, 
                        coefficients = TRUE, variable_importances = FALSE, 
                        predictions = TRUE, model_performance = TRUE, 
                        model_args = list()) {
  easy_analysis(.data, dependent_variable, algorithm = "glmnet", 
                family = family, resample = resample, 
                preprocess = preprocess, measure = measure, 
                exclude_variables = exclude_variables, 
                categorical_variables = categorical_variables,  
                train_size = train_size, foldid = foldid,  
                survival_rate_cutoff = survival_rate_cutoff, 
                n_samples = n_samples, n_divisions = n_divisions, 
                n_iterations = n_iterations, random_state = random_state, 
                progress_bar = progress_bar, n_core = n_core, 
                coefficients = coefficients, 
                variable_importances = variable_importances, 
                predictions = predictions, model_performance = model_performance, 
                model_args = model_args)
}
