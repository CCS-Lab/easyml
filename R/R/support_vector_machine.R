#' TO BE EDITED.
#' 
#' TO BE EDITED.
#'
#' @return TO BE EDITED.
#' @export
support_vector_machine_fit_model_gaussian <- function(X, y, ...) {
  X <- as.matrix(X)
  e1071::svm(X, y, scale = FALSE, type = "nu-regression", ...)
}

#' TO BE EDITED.
#' 
#' TO BE EDITED.
#'
#' @return TO BE EDITED.
#' @export
support_vector_machine_fit_model_binomial <- function(X, y, ...) {
  X <- as.matrix(X)
  y <- factor(y)
  e1071::svm(X, y, scale = FALSE, type = "C-classification", ...)
}

#' TO BE EDITED.
#' 
#' TO BE EDITED.
#'
#' @return TO BE EDITED.
#' @export
support_vector_machine_predict_model <- function(results, newx) {
  as.numeric(predict(results, newdata = newx))
}

#' TO BE EDITED.
#' 
#' TO BE EDITED.
#'
#' @return TO BE EDITED.
#' @export
easy_support_vector_machine <- function(.data, dependent_variable, family = "gaussian", 
                     resample = NULL, preprocess = NULL, measure = NULL, 
                     exclude_variables = NULL, categorical_variables = NULL, 
                     train_size = 0.667, survival_rate_cutoff = 0.05, 
                     n_samples = 1000, n_divisions = 1000, 
                     n_iterations = 10, random_state = NULL, 
                     progress_bar = TRUE, n_core = 1, ...) {
  easy_analysis(.data, dependent_variable, algorithm = "support_vector_machine", 
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
