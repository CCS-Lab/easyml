#' Fit model.
#' 
#' The generic function  for fitting a model within 
#' the easyml core framework. Users can create their own 
#' fit_model function for their own class of model.
#' 
#' @param object A list of class \code{easy_*}, where * is the name of the algorithm.
#' @return A list of class \code{easy_*}, where * is the name of the algorithm.
#' @export
fit_model <- function(object) {
  UseMethod("fit_model")
}

#' Extract coefficients.
#' 
#' The generic function  for extracting coefficients from
#' a model within the easyml core framework, if such an
#' operation is applicable for that model. Users can create their own 
#' extract_coefficients function for their own class of model.
#' 
#' @param object A list of class \code{easy_*}, where * is the name of the algorithm.
#' @return A data.frame, the generated coefficients.
#' @export
extract_coefficients <- function(object) {
  UseMethod("extract_coefficients")
}

#' Extract variable importances.
#' 
#' The generic function  for extracting variable importances from
#' a model within the easyml core framework, if such an
#' operation is applicable for that model. Users can create their own 
#' extract_variable_importances function for their own class of model.
#' 
#' @param object A list of class \code{easy_*}, where * is the name of the algorithm.
#' @return A data.frame, the generated random forest variable importance scores.
#' @export
extract_variable_importances <- function(object) {
  UseMethod("extract_variable_importances")
}

#' Predict model.
#' 
#' The generic function  for generating predictions from a model within 
#' the easyml core framework. Users can create their own 
#' predict_model function for their own class of model.
#' 
#' @param object A list of class \code{easy_*}, where * is the name of the algorithm.
#' @param newx A data.frame, the new data to use for predictions.
#' @return A vector, the predicted values using the new data.
#' @export
predict_model <- function(object, newx = NULL) {
  UseMethod("predict_model")
}

#' The core recipe of easyml.
#' 
#' This recipe is the workhorse behind all of the easy_* functions. 
#'
#' @param .data A data.frame; the data to be analyzed.
#' @param dependent_variable A character vector of length one; the dependent variable for this analysis.
#' @param algorithm A character vector of length one; the algorithm to run on the data. Choices are currently one of c("deep_neural_network", "glinternet", "glmnet", "neural_network", "random_forest", "support_vector_machine").
#' @param family A character vector of length one; the type of regression to run on the data. Choices are one of c("gaussian", "binomial"). Defaults to "gaussian".
#' @param resample A function; the function for resampling the data. Defaults to NULL.
#' @param preprocess A function; the function for preprocessing the data. Defaults to NULL.
#' @param measure A function; the function for measuring the results. Defaults to NULL.
#' @param exclude_variables A character vector; the variables from the data set to exclude. Defaults to NULL.
#' @param categorical_variables A character vector; the variables that are categorical. Defaults to NULL.
#' @param train_size A numeric vector of length one; specifies what proportion of the data should be used for the training data set. Defaults to 0.667.
#' @param foldid A vector with length equal to \code{length(y)} which identifies cases belonging to the same fold. 
#' @param survival_rate_cutoff A numeric vector of length one; for \code{\link{easy_glmnet}}, specifies the minimal threshold (as a percentage) a coefficient must appear out of n_samples. Defaults to 0.05.
#' @param n_samples An integer vector of length one; specifies the number of times the coefficients and predictions should be generated. Defaults to 1000. 
#' @param n_divisions An integer vector of length one; specifies the number of times the data should be divided when replicating the measures of model performance. Defaults to 1000.
#' @param n_iterations An integer vector of length one; during each division, specifies the number of times the predictions should be generated. Defaults to 10.
#' @param random_state An integer vector of length one; specifies the seed to be used for the analysis. Defaults to NULL.
#' @param progress_bar A logical vector of length one; specifies whether to display a progress bar during calculations. Defaults to TRUE.
#' @param n_core An integer vector of length one; specifies the number of cores to use for this analysis. Currently only works on Mac OSx and Unix/Linux systems. Defaults to 1.
#' @param coefficients A logical vector of length one; whether or not to generate coefficients for this analysis.
#' @param variable_importances A logical vector of length one; whether or not to generate variable importances for this analysis.
#' @param predictions A logical vector of length one; whether or not to generate predictions for this analysis.
#' @param model_performance A logical vector of length one; whether or not to generate measures of model performance for this analysis.
#' @param model_args A list; the arguments to be passed to the algorithm specified.
#' @return A list of class \code{easy_*}, where * is the name of the algorithm.
#' \describe{
#' \item{call}{An object of class \code{call}; the original function call.}
#' \item{data}{A data.frame; the original data.}
#' \item{dependent_variable}{A character vector of length one; the dependent variable for this analysis.}
#' \item{algorithm}{A character vector of length one; the algorithm to run on the data.}
#' \item{class}{A character vector of length one; the class of the object.}
#' \item{family}{A character vector of length one; the type of regression to run on the data. Choices are one of c("gaussian", "binomial"). Defaults to "gaussian".}
#' \item{resample}{A function; the function for resampling the data.}
#' \item{preprocess}{A function; the function for preprocessing the data.}
#' \item{measure}{A function; the function for measuring the results.}
#' \item{exclude_variables}{A character vector; the variables from the data set to exclude.}
#' \item{train_size}{A numeric vector of length one; specifies what proportion of the data should be used for the training data set.}
#' \item{survival_rate_cutoff}{A numeric vector of length one; for \code{\link{easy_glmnet}}, specifies the minimal threshold (as a percentage) a coefficient must appear out of n_samples.}
#' \item{n_samples}{An integer vector of length one; specifies the number of times the coefficients and predictions should be generated.}
#' \item{n_divisions}{An integer vector of length one; specifies the number of times the data should be divided when generating measures of model performance.}
#' \item{n_iterations}{An integer vector of length one; during each division, specifies the number of times the predictions should be generated.}
#' \item{random_state}{An integer vector of length one; specifies the seed to be used for the analysis.}
#' \item{progress_bar}{A logical vector of length one; specifies whether to display a progress bar during calculations.}
#' \item{n_core}{An integer vector of length one; specifies the number of cores to use for this analysis.}
#' \item{generate_coefficients}{A logical vector of length one; whether or not to generate coefficients for this analysis.}
#' \item{generate_variable_importances}{A logical vector of length one; whether or not to generate variable importances for this analysis.}
#' \item{generate_predictions}{A logical vector of length one; whether or not to generate predictions for this analysis.}
#' \item{generate_model_performance}{A logical vector of length one; whether or not to generate measures of model performance for this analysis.}
#' \item{model_args}{A list; the arguments to be passed to the algorithm specified.}
#' \item{column_names}{A character vector; the column names.}
#' \item{categorical_variables}{A logical vector; the variables that are categorical.}
#' \item{X}{A data.frame; the full dataset to be used for modeling.}
#' \item{y}{A vector; the full response variable to be used for modeling.}
#' \item{coefficients}{A (n_variables, n_samples) matrix; the generated coefficients.}
#' \item{coefficients_processed}{A data.frame; the coefficients after being processed.}
#' \item{plot_coefficients_processed}{A ggplot object; the plot of the processed coefficients.}
#' \item{X_train}{A data.frame; the train dataset to be used for modeling.}
#' \item{X_test}{A data.frame; the test dataset to be used for modeling.}
#' \item{y_train}{A vector; the train response variable to be used for modeling.}
#' \item{y_test}{A vector; the test response variable to be used for modeling.}
#' \item{predictions_train}{A (nrow(X_train), n_samples) matrix; the train predictions.}
#' \item{predictions_test}{A (nrow(X_test), n_samples) matrix; the test predictions.}
#' \item{predictions_train_mean}{A vector; the mean train predictions.}
#' \item{predictions_test_mean}{A vector; the mean test predictions.}
#' \item{plot_predictions}{A function; the function for plotting predictions generated by the model.}
#' \item{plot_predictions_train_mean}{A ggplot object; the plot of the mean train predictions.}
#' \item{plot_predictions_test_mean}{A ggplot object; the plot of the mean test predictions.}
#' \item{model_performance_train}{A vector of length n_divisions; the measures of model performance on the train datasets.}
#' \item{model_performance_test}{A vector of length n_divisions; the measures of model performance on the test datasets.}
#' \item{plot_model_performance}{A function; the function for plotting the measures of model performance.}
#' \item{plot_model_performance_train}{A ggplot object; the plot of the measures of model performance on the train datasets.}
#' \item{plot_model_performance_test}{A ggplot object; the plot of the measures of model performance on the test datasets.}
#' }
#' @family recipes
#' @export
easy_analysis <- function(.data, dependent_variable, algorithm, 
                          family = "gaussian", resample = NULL, 
                          preprocess = NULL, measure = NULL, 
                          exclude_variables = NULL, 
                          categorical_variables = NULL, train_size = 0.667, 
                          foldid = NULL, survival_rate_cutoff = 0.05, 
                          n_samples = 1000, n_divisions = 1000, 
                          n_iterations = 10, random_state = NULL, 
                          progress_bar = TRUE, n_core = 1, 
                          coefficients = NULL, variable_importances = NULL, 
                          predictions = NULL, model_performance = NULL, 
                          model_args = list()) {
  # Instantiate object
  object <- list()
  
  # Capture call
  object[["call"]] <- match.call(call = sys.call(sys.parent(1L)))
  
  # Capture data
  object[["data"]] <- .data
  
  # Capture dependent variable
  object[["dependent_variable"]] <- dependent_variable
  
  # Capture algorithm
  object[["algorithm"]] <- algorithm
  
  # Capture class
  .class <- paste0("easy_", algorithm)
  object[["class"]] <- .class
  
  # Capture family
  object[["family"]] <- family
  
  # Set and capture resample
  resample <- set_resample(resample, family)
  object[["resample"]] <- resample
  
  # Set and capture preprocess
  preprocess <- set_preprocess(preprocess, algorithm)
  object[["preprocess"]] <- preprocess
  
  # Set and capture measure
  measure <- set_measure(measure, algorithm, family)
  object[["measure"]] <- measure
  
  # Capture exclude variables
  object[["exclude_variables"]] <- exclude_variables
  
  # Capture train size
  object[["train_size"]] <- train_size
  
  # Capture foldid
  object[["foldid"]] <- foldid

  # Capture survival rate cutoff
  object[["survival_rate_cutoff"]] <- survival_rate_cutoff

  # Capture n samples
  object[["n_samples"]] <- n_samples

  # Capture n divisions
  object[["n_divisions"]] <- n_divisions

  # Capture n iterations
  object[["n_iterations"]] <- n_iterations

  # Capture random state
  set_random_state(random_state)
  object[["random_state"]] <- random_state
  
  # Capture progress bar
  object[["progress_bar"]] <- progress_bar
  
  # Capture number of cores
  object[["n_core"]] <- n_core
  
  # Capture coefficients
  object[["generate_coefficients"]] <- coefficients
  
  # Capture variable importances
  object[["generate_variable_importances"]] <- variable_importances
  
  # Capture predictions
  object[["generate_predictions"]] <- predictions
  
  # Capture model performance
  object[["generate_model_performance"]] <- model_performance

  # Capture model arguments
  object[["model_args"]] <- model_args
  
  # Set and capture column names
  column_names <- colnames(.data)
  column_names <- set_column_names(column_names, dependent_variable, 
                                   preprocess, exclude_variables, 
                                   categorical_variables)
  object[["column_names"]] <- column_names
  
  # Set and capture categorical variables
  cat_vars <- set_categorical_variables(column_names, categorical_variables)
  object[["categorical_variables"]] <- cat_vars
  
  # Remove variables and capture data
  .data <- remove_variables(.data, exclude_variables)
  object[["data"]] <- .data
  
  # Set and capture dependent variable
  y <- set_dependent_variable(.data, dependent_variable)
  object[["y"]] <- y
  
  # Set and capture independent variables
  X <- set_independent_variables(.data, dependent_variable)
  X <- X[, column_names]
  object[["X"]] <- X
  
  # Set class of the object
  object <- structure(object, class = .class)
  
  # Assess if coefficients should be generated for this algorithm
  if (coefficients) {
    # Set random state
    if (!is.null(random_state)) {
      set_random_state(random_state)
    }
    
    # Generate coefficients
    coefs <- generate_coefficients(object)
    object[["coefficients"]] <- coefs
    
    # Process and capture coefficients
    coefs_processed <- process_coefficients(coefs, survival_rate_cutoff)
    object[["coefficients_processed"]] <- coefs_processed
    
    # Create and capture plots of coefficients
    g <- plot_coefficients_processed(coefs_processed)
    object[["plot_coefficients"]] <- g
  }
  
  # Assess if variable importances should be generated for this algorithm
  if (variable_importances) {
    # Set random state
    if (!is.null(random_state)) {
      set_random_state(random_state)
    }
    
    # Generate variable importances
    variable_imps <- generate_variable_importances(object)
    object[["variable_importances"]] <- variable_imps
    
    # Process and capture variable_importances
    variable_imps_processed <- process_variable_importances(variable_imps)
    object[["variable_importances_processed"]] <- variable_imps_processed
    
    # Create and capture plots of variable importances
    g <- plot_variable_importances_processed(variable_imps_processed)
    object[["plot_variable_importances"]] <- g
  }
  
  # Assess if predictions for a single train-test split 
  # should be generated for this algorithm.
  if (predictions) {
    # Set random state
    if (!is.null(random_state)) {
      set_random_state(random_state)
    }
    
    # Resample and capture data
    split_data <- resample(X, y, train_size = train_size, foldid = foldid)
    object[["X_train"]] <- split_data[["X_train"]]
    object[["X_test"]] <- split_data[["X_test"]]
    y_train <- object[["y_train"]] <- split_data[["y_train"]]
    y_test <- object[["y_test"]] <- split_data[["y_test"]]
    
    # Generate predictions
    preds <- generate_predictions(object)
    predictions_train <- preds[["predictions_train"]]
    predictions_test <- preds[["predictions_test"]]

    # Process and capture predictions
    predictions_train <- apply(predictions_train, 1, mean)
    predictions_test <- apply(predictions_test, 1, mean)
    object[["predictions_train"]] <- predictions_train
    object[["predictions_test"]] <- predictions_test
    
    # Set and capture plot_predictions function
    plot_predictions <- set_plot_predictions(algorithm, family)
    object[["plot_predictions"]] <- plot_predictions
    
    # Create and capture plots of predictions
    p_train <- plot_predictions(y_train, predictions_train) + 
      ggplot2::labs(subtitle = "Train Dataset")
    object[["plot_predictions_single_train_test_split_train"]] <- p_train
    p_test <- plot_predictions(y_test, predictions_test) + 
      ggplot2::labs(subtitle = "Test Dataset")
    object[["plot_predictions_single_train_test_split_test"]] <- p_test
    
    if (family == "binomial") {
      # Create and capture predictions plots
      p_train <- plot_roc_curve(y_train, predictions_train) + 
        ggplot2::labs(subtitle = "Train Dataset")
      object[["plot_roc_single_train_test_split_train"]] <- p_train
      p_test <- plot_roc_curve(y_test, predictions_test) + 
        ggplot2::labs(subtitle = "Test Dataset")
      object[["plot_roc_single_train_test_split_test"]] <- p_test
    }
  }
  
  # Assess if measures of model performance for multiple train-test splits
  # should be generated for this algorithm
  if (model_performance) {
    # Set random state
    if (!is.null(random_state)) {
      set_random_state(random_state)
    }
    
    # Generate measures of model performance
    mets <- generate_model_performance(object)
    model_performance_train <- mets[["model_performance_train"]]
    object[["model_performance_train"]] <- model_performance_train
    model_performance_test <- mets[["model_performance_test"]]
    object[["model_performance_test"]] <- model_performance_test
    
    # Set and capture plot_model_performance function
    plot_model_performance <- set_plot_model_performance(measure)
    object[["plot_model_performance"]] <- plot_model_performance
    
    # Create and capture plots of model performance
    plot_model_performance_train <- plot_model_performance(model_performance_train) + 
      ggplot2::labs(subtitle = "Train Dataset")
    object[["plot_model_performance_train"]] <- plot_model_performance_train
    plot_model_performance_test <- plot_model_performance(model_performance_test) + 
      ggplot2::labs(subtitle = "Test Dataset")
    object[["plot_model_performance_test"]] <- plot_model_performance_test
  }
  
  # Return object
  object
}
