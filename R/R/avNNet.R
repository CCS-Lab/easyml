#' Fit an average neural network model.
#' 
#' This function wraps the procedure for fitting an 
#' average neural network model and makes it accessible 
#' to the easyml core framework.
#' 
#' @param object A list of class \code{easy_avNNet}.
#' @return A list of class \code{easy_avNNet}.
#' @export
fit_model.easy_avNNet <- function(object) {
  # set model arguments
  model_args <- object[["model_args"]]
  
  # process model_args
  model_args[["x"]] <- as.matrix(object[["X"]])
  model_args[["y"]] <- object[["y"]]
  
  # build model
  model <- do.call(caret::avNNet, model_args)
  object[["model_args"]] <- model_args
  object[["model"]] <- model
  
  # write output
  object
}

#' Predict values for an average neural network model.
#' 
#' This function wraps the procedure for predicting values from 
#' a average neural network model and makes it accessible 
#' to the easyml core framework.
#' 
#' @param object A list of class \code{easy_avNNet}.
#' @param newx A data.frame, the new data to use for predictions.
#' @return A vector, the predicted values using the new data.
#' @export
predict_model.easy_avNNet <- function(object, newx = NULL) {
  newx <- as.matrix(newx)
  model <- object[["model"]]
  family <- object[["family"]]
  if (family == "gaussian") {
    type <- "raw"
  } else if (family == "binomial") {
    type <- "class"
  }
  preds <- stats::predict(model, newdata = newx, type = type)
  preds
}

#' Easily build and evaluate an average neural network model.
#' 
#' This function wraps the easyml core framework, allowing a user 
#' to easily run the easyml methodology for an average neural network
#' model.
#' 
#' @inheritParams easy_analysis
#' @return A list of class \code{easy_avNNet}.
#' @family recipes
#' @examples 
#' \dontrun{
#' library(easyml) # https://github.com/CCS-Lab/easyml
#' 
#' # Gaussian
#' data("prostate", package = "easyml")
#' results <- easy_avNNet(prostate, "lpsa", 
#'                        n_samples = 10, n_divisions = 10, 
#'                        n_iterations = 2, random_state = 12345, 
#'                        n_core = 1)
#' 
#' # Binomial
#' data("cocaine_dependence", package = "easyml")
#' model_args <- list(size = 5, linout = TRUE, trace = FALSE)
#' results <- easy_avNNet(cocaine_dependence, "diagnosis", 
#'                        family = "binomial", 
#'                        exclude_variables = c("subject"), 
#'                        categorical_variables = c("male"), 
#'                        preprocess = preprocess_scale, 
#'                        n_samples = 10, n_divisions = 10, 
#'                        n_iterations = 2, random_state = 12345, 
#'                        n_core = 1, model_args = model_args)
#' }
#' @export
easy_avNNet <- function(.data, dependent_variable, 
                        family = "gaussian", resample = NULL, 
                        preprocess = preprocess_scale, 
                        measure = NULL, 
                        exclude_variables = NULL, 
                        categorical_variables = NULL, 
                        train_size = 0.667, foldid = NULL, 
                        survival_rate_cutoff = 0.05, 
                        n_samples = 1000, n_divisions = 1000, 
                        n_iterations = 10, 
                        random_state = NULL, 
                        progress_bar = TRUE, n_core = 1, 
                        coefficients = FALSE, 
                        variable_importances = FALSE, 
                        predictions = TRUE, model_performance = TRUE, 
                        model_args = list()) {
  easy_analysis(.data, dependent_variable, 
                algorithm = "avNNet", 
                family = family, resample = resample, 
                preprocess = preprocess, measure = measure, 
                exclude_variables = exclude_variables, 
                categorical_variables = categorical_variables,  
                train_size = train_size, foldid = foldid,  
                survival_rate_cutoff = survival_rate_cutoff, 
                n_samples = n_samples, 
                n_divisions = n_divisions, 
                n_iterations = n_iterations, 
                random_state = random_state, 
                progress_bar = progress_bar, n_core = n_core, 
                coefficients = coefficients, 
                variable_importances = variable_importances, 
                predictions = predictions, model_performance = model_performance, 
                model_args = model_args)
}
