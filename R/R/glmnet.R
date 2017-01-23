#' TO BE EDITED.
#' 
#' TO BE EDITED.
#'
#' @return TO BE EDITED.
#' @export
glmnet_fit_model_gaussian <- function(X, y, ...) {
  X <- as.matrix(X)
  model <- glmnet::glmnet(X, y, family = "gaussian", ...)
  cv_model <- glmnet::cv.glmnet(X, y, family = "gaussian", ...)
  list(model = model, cv_model = cv_model)
}

#' TO BE EDITED.
#' 
#' TO BE EDITED.
#'
#' @return TO BE EDITED.
#' @export
glmnet_fit_model_binomial <- function(X, y, ...) {
  X <- as.matrix(X)
  model <- glmnet::glmnet(X, y, family = "binomial", ...)
  cv_model <- glmnet::cv.glmnet(X, y, family = "binomial", ...)
  list(model = model, cv_model = cv_model)
}

#' TO BE EDITED.
#' 
#' TO BE EDITED.
#'
#' @return TO BE EDITED.
#' @export
glmnet_extract_coefficients <- function(results) {
  model <- results[["model"]]
  cv_model <- results[["cv_model"]]
  coefs <- coef(model, s = cv_model$lambda.min)
  .data <- data.frame(t(as.matrix(as.numeric(coefs), nrow = 1)))
  colnames(.data) <- rownames(coefs)
  .data
}

#' TO BE EDITED.
#' 
#' TO BE EDITED.
#'
#' @return TO BE EDITED.
#' @export
glmnet_predict_model <- function(results, newx) {
  newx <- as.matrix(newx)
  model <- results[["model"]]
  cv_model <- results[["cv_model"]]
  predict(model, newx = newx, s = cv_model$lambda.min, type = "response")
}

#' TO BE EDITED.
#' 
#' TO BE EDITED.
#'
#' @return TO BE EDITED.
#' @export
easy_glmnet <- function(.data, dependent_variable, family = "gaussian", 
                        resample = NULL, preprocess = NULL, measure = NULL, 
                        exclude_variables = NULL, categorical_variables = NULL, 
                        train_size = 0.667, survival_rate_cutoff = 0.05, 
                        n_samples = 1000, n_divisions = 1000, 
                        n_iterations = 10, random_state = NULL, 
                        progress_bar = TRUE, n_core = 1, ...) {
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
