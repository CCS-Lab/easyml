#' Fit a penalized regression model with interactions.
#' 
#' This function wraps the procedure for fitting a 
#' glinternet model and makes it accessible 
#' to the easyml core framework.
#' 
#' @param object A list of class \code{easy_glinternet}.
#' @return A list of class \code{easy_glinternet}.
#' @export
fit_model.easy_glinternet <- function(object) {
  # set model arguments
  model_args <- object[["model_args"]]
  
  # process model_args
  X <- as.matrix(object[["X"]])
  model_args[["X"]] <- X
  model_args[["Y"]] <- object[["y"]]
  model_args[["family"]] <- object[["family"]]
  cat_vars <- object[["categorical_variables"]]
  if (is.null(cat_vars)) {
    numLevels <- rep(1, ncol(X))
  } else {
    numLevels <- cat_vars + 1
  }
  model_args[["numLevels"]] <- numLevels

  # build model
  done <- FALSE
  while (!done) {
    model <- try(do.call(glinternet::glinternet.cv, model_args), silent = TRUE)
    done <- class(model) != "try-error"
  }
  
  object[["model_args"]] <- model_args
  object[["model"]] <- model
  
  # write output
  object
}

#' Predict values for a penalized regression model with interactions.
#' 
#' This function wraps the procedure for predicting values from 
#' a glinternet model and makes it accessible 
#' to the easyml core framework.
#' 
#' @param object A list of class \code{easy_glinternet}.
#' @param newx A data.frame, the new data to use for predictions.
#' @return A vector, the predicted values using the new data.
#' @export
predict_model.easy_glinternet <- function(object, newx = NULL) {
  newx <- as.matrix(newx)
  model <- object[["model"]]
  stats::predict(model, X = newx)
}

#' Easily build and evaluate a penalized regression model with interactions.
#' 
#' This function wraps the easyml core framework, allowing a user 
#' to easily run the easyml methodology for a glinternet
#' model.
#'
#' @inheritParams easy_analysis
#' @return A list of class \code{easy_glinternet}.
#' @family recipes
#' @examples 
#' \dontrun{
#' library(easyml) # https://github.com/CCS-Lab/easyml
#' 
#' # Gaussian
#' data("prostate", package = "easyml")
#' results <- easy_glinternet(prostate, "lpsa", 
#'                            n_samples = 10, n_divisions = 10, 
#'                            n_iterations = 2, random_state = 12345, 
#'                            n_core = 1)
#' 
#' # Binomial
#' data("cocaine_dependence", package = "easyml")
#' results <- easy_glinternet(cocaine_dependence, "diagnosis", 
#'                            family = "binomial", 
#'                            exclude_variables = c("subject"), 
#'                            categorical_variables = c("male"), 
#'                            preprocess = preprocess_scale, 
#'                            n_samples = 10, n_divisions = 10, 
#'                            n_iterations = 2, random_state = 12345, 
#'                            n_core = 1)
#' }
#' @export
easy_glinternet <- function(.data, dependent_variable, family = "gaussian", 
                            resample = NULL, preprocess = preprocess_scale, 
                            measure = NULL, exclude_variables = NULL, 
                            categorical_variables = NULL, 
                            train_size = 0.667, foldid = NULL, 
                            survival_rate_cutoff = 0.05, 
                            n_samples = 1000, n_divisions = 1000, 
                            n_iterations = 10, random_state = NULL, 
                            progress_bar = TRUE, n_core = 1, 
                            coefficients = FALSE, 
                            variable_importances = FALSE, 
                            predictions = TRUE, model_performance = TRUE, 
                            model_args = list()) {
  n_core <- 1 # TODO need to explore why glinternet pauses on parallelization
  easy_analysis(.data, dependent_variable, algorithm = "glinternet", 
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
