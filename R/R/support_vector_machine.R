#' Fit a support vector machine regression model.
#' 
#' This function wraps the procedure for fitting a 
#' support vector machine model and makes it accessible 
#' to the easyml core framework.
#' 
#' @param object A list of class \code{easy_support_vector_machine}.
#' @return A list of class \code{easy_support_vector_machine}.
#' @export
fit_model.easy_support_vector_machine <- function(object) {
  # set model arguments
  model_args <- object[["model_args"]]
  
  # process model_args
  model_args[["x"]] <- as.matrix(object[["X"]])
  model_args[["y"]] <- object[["y"]]
  model_args[["scale"]] <- FALSE
  family <- object[["family"]]
  if (family == "gaussian") {
    type <- "nu-regression"
  } else if (family == "binomial") {
    type <- "C-classification"
    model_args[["probability"]] <- TRUE
  }
  model_args[["type"]] <- type

  # build model
  model <- do.call(e1071::svm, model_args)
  object[["model_args"]] <- model_args
  object[["model"]] <- model
  
  # write output
  object
}

#' Predict values for a support vector machine regression model.
#' 
#' This function wraps the procedure for predicting values from 
#' a support vector machine model and makes it accessible 
#' to the easyml core framework.
#' 
#' @param object A list of class \code{easy_support_vector_machine}.
#' @param newx A data.frame, the new data to use for predictions.
#' @return A vector, the predicted values using the new data.
#' @export
predict_model.easy_support_vector_machine <- function(object, newx = NULL) {
  model <- object[["model"]]
  
  if (object[["family"]] == "binomial") {
    # If newx == NULL (i.e. for training data prediction), do not pass new data
    if (is.null(newx)) {
      preds <- stats::predict(model, model$SV, probability = TRUE)
    } else {
      preds <- stats::predict(model, newdata = newx, probability = TRUE)
    }
    preds <- as.numeric(attr(preds, "probabilities")[, 2])
  } else {
    # If newx == NULL (i.e. for training data prediction), do not pass new data
    if (is.null(newx)) {
      preds <- as.numeric(stats::predict(model))
    } else {
      preds <- as.numeric(stats::predict(model, newdata = newx))
    }
  }
  
  preds
}

#' Easily build and evaluate a support vector machine regression model.
#' 
#' This function wraps the easyml core framework, allowing a user 
#' to easily run the easyml methodology for a support vector machine
#' model.
#'
#' @inheritParams easy_analysis
#' @return A list of class \code{easy_support_vector_machine}.
#' @family recipes
#' @examples 
#' \dontrun{
#' library(easyml) # https://github.com/CCS-Lab/easyml
#' 
#' # Gaussian
#' data("prostate", package = "easyml")
#' results <- easy_support_vector_machine(prostate, "lpsa", 
#'                                        n_samples = 10, 
#'                                        n_divisions = 10, 
#'                                        n_iterations = 2, 
#'                                        random_state = 1, n_core = 1)
#' 
#' # Binomial
#' data("cocaine_dependence", package = "easyml")
#' results <- easy_support_vector_machine(cocaine_dependence, "diagnosis", 
#'                                        family = "binomial", 
#'                                        preprocesss = preprocess_scale, 
#'                                        exclude_variables = c("subject"), 
#'                                        categorical_variables = c("male"), 
#'                                        n_samples = 10, 
#'                                        n_divisions = 10, 
#'                                        n_iterations = 2, 
#'                                        random_state = 1, n_core = 1)
#' }
#' @export
easy_support_vector_machine <- function(.data, dependent_variable, 
                                        family = "gaussian", 
                                        resample = NULL, 
                                        preprocess = preprocess_scale, 
                                        measure = NULL, 
                                        exclude_variables = NULL, 
                                        categorical_variables = NULL, 
                                        train_size = 0.667, foldid = NULL,
                                        n_samples = 1000, n_divisions = 1000, 
                                        n_iterations = 10, 
                                        random_state = NULL, 
                                        progress_bar = TRUE, n_core = 1, 
                                        coefficients = FALSE, 
                                        variable_importances = FALSE, 
                                        predictions = TRUE, model_performance = TRUE, 
                                        model_args = list()) {
  easy_analysis(.data, dependent_variable, 
                algorithm = "support_vector_machine", 
                family = family, resample = resample, 
                preprocess = preprocess, measure = measure, 
                exclude_variables = exclude_variables, 
                categorical_variables = categorical_variables,  
                train_size = train_size, foldid = foldid,  
                n_samples = n_samples, n_divisions = n_divisions, 
                n_iterations = n_iterations, random_state = random_state, 
                progress_bar = progress_bar, n_core = n_core, 
                coefficients = coefficients, 
                variable_importances = variable_importances, 
                predictions = predictions, model_performance = model_performance, 
                model_args = model_args)
}
