library(caret)

# load data
data("cocaine_dependence", package = "easyml")

# fit model
modelFit <- avNNet(cocaine_dependence[, c(-1, -2)], 
                   factor(cocaine_dependence$diagnosis), 
                   size = 5, linout = TRUE, trace = FALSE)

# predictions
head(predict(modelFit, type = "class"))

# easyml way
library(easyml)

# wrap the avNNet function
fit_model <- function(X, y, ...) {
  avNNet(X, y, ...)
}

# wrap the predict function
predict_model <- function(results, newx = NULL) {
  if (is.null(newx)) {
    predictions <- predict(results)
  } else {
    predictions <- predict(results, newx)
  }
  predictions
}

# pass those two functions in, specify other functions and arguments, 
# and store the results
metrics <- replicate_metrics(fit_model = fit_model, 
                             predict_model = predict_model, 
                             resample = resample_stratified_train_test_split, 
                             preprocess = preprocess_identity, 
                             measure = measure_area_under_curve, 
                             X = cocaine_dependence[, c(-1, -2)], 
                             y = cocaine_dependence$diagnosis,
                             train_size = 0.667, 
                             n_divisions = 1000, n_iterations = 2, 
                             progress_bar = TRUE, n_core = 1, 
                             size = 5, linout = TRUE, trace = FALSE)

# plot the results
plot_metrics_binomial_area_under_curve(metrics$metrics_train_mean) + 
  ggplot2::labs(subtitle = "Train Metrics")

plot_metrics_binomial_area_under_curve(metrics$metrics_test_mean) + 
  ggplot2::labs(subtitle = "Test Metrics")
