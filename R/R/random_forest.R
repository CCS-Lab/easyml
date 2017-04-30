#' Fit a random forest model.
#' 
#' @param object A list of class \code{easy_random_forest}.
#' @return A list of class \code{easy_random_forest}.
#' @export
fit_model.easy_random_forest <- function(object) {
  # set model arguments
  model_args <- object[["model_args"]]
  
  # process model_args
  model_args[["x"]] <- as.matrix(object[["X"]])
  
  if (object[["family"]] == "binomial") {
    model_args[["y"]] <- factor(object[["y"]])
  } else {
    model_args[["y"]] <- object[["y"]]
  }

  # build model
  model <- do.call(randomForest::randomForest, model_args)
  object[["model_args"]] <- model_args
  object[["model"]] <- model
  
  # write output
  object
}

#' Extract variable importance scores from a random forest model.
#' 
#' @param object A list of class \code{easy_random_forest}.
#' @return A data.frame, the replicated random forest variable importance scores.
#' @export
extract_variable_importances.easy_random_forest <- function(object) {
  model <- object[["model"]]
  importance <- randomForest::importance(model)
  importance_df <- data.frame(t(importance))
  rownames(importance_df) <- NULL
  importance_df
}

#' Predict values for a random forest regression model.
#' 
#' @param object A list of class \code{easy_random_forest}.
#' @param newx A data.frame, the new data to use for predictions.
#' @return A vector, the predicted values using the new data.
#' @export
predict_model.easy_random_forest <- function(object, newx = NULL) {
  model <- object[["model"]]
  # If newx == NULL (i.e. for training data prediction), do not pass new data
  if (is.null(newx)) {
    preds <- as.numeric(stats::predict(model, type = "prob"))
  } else {
    preds <- as.numeric(stats::predict(model, newdata = newx, type = "prob"))
    preds <- matrix(preds, ncol = 2)[, 2]
  }
  preds
}

#' Easily build and evaluate a random forest regression model.
#' 
#' @inheritParams easy_analysis
#' @return A list of class \code{easy_random_forest}.
#' @family recipes
#' @examples 
#' \dontrun{
#' library(easyml) # https://github.com/CCS-Lab/easyml
#' 
#' # Gaussian
#' data("prostate", package = "easyml")
#' results <- easy_random_forest(prostate, "lpsa", 
#'                               n_samples = 10L, 
#'                               n_divisions = 10, 
#'                               n_iterations = 2, 
#'                               random_state = 12345, n_core = 1)
#' 
#' # Binomial
#' data("cocaine_dependence", package = "easyml")
#' results <- easy_random_forest(cocaine_dependence, "diagnosis", 
#'                               family = "binomial", 
#'                               exclude_variables = c("subject"),
#'                               categorical_variables = c("male"),
#'                               n_samples = 10, 
#'                               n_divisions = 10, 
#'                               n_iterations = 2, 
#'                               random_state = 12345, n_core = 1)
#' }
#' @export
easy_random_forest <- function(.data, dependent_variable, 
                               family = "gaussian", resample = NULL, 
                               preprocess = preprocess_identity, 
                               measure = NULL, 
                               exclude_variables = NULL, 
                               categorical_variables = NULL, 
                               train_size = 0.667, foldid = NULL, 
                               n_samples = 1000, n_divisions = 1000, 
                               n_iterations = 10, random_state = NULL, 
                               progress_bar = TRUE, n_core = 1, 
                               coefficients = FALSE, 
                               variable_importances = TRUE, 
                               predictions = TRUE, metrics = TRUE, 
                               model_args = list()) {
  easy_analysis(.data, dependent_variable, algorithm = "random_forest", 
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
                predictions = predictions, metrics = metrics, 
                model_args = model_args)
}
