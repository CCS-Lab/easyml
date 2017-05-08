library(easyml)
context("plot")

# Load data
data("cocaine_dependence", package = "easyml")

# Settings
.n_samples <- 5
.n_divisions <- 5
.n_iterations <- 2

# # Analyze data
# glmnet_results <- easy_glmnet(cocaine_dependence, "diagnosis",
#                               family = "binomial",
#                               preprocess = preprocess_scale,
#                               exclude_variables = c("subject"),
#                               categorical_variables = c("male"),
#                               n_samples = .n_samples, n_divisions = .n_divisions,
#                               n_iterations = .n_iterations, random_state = 12345, n_core = 1)
# 
# test_that("Test plot_coefficients_processed.", {
#   g <- plot_coefficients_processed(glmnet_results$coefficients_processed)
#   expect_equal(class(g), c("gg", "ggplot"))
# })

# Analyze data
random_forest_results <- easy_random_forest(cocaine_dependence, "diagnosis", 
                                            family = "binomial", exclude_variables = c("subject"), 
                                            categorical_variables = c("male"), 
                                            n_samples = .n_samples, n_divisions = .n_divisions, 
                                            n_iterations = .n_iterations, random_state = 1, n_core = 1)

test_that("Test plot_variable_importances_processed.", {
  g <- plot_variable_importances_processed(random_forest_results$variable_importances_processed)
  expect_equal(class(g), c("gg", "ggplot"))
})

test_that("Test plot_predictions_gaussian.", {
  g <- plot_predictions_gaussian(rnorm(100, 0, 1), rnorm(100, 0, 1))
  expect_equal(class(g), c("gg", "ggplot"))
})

test_that("Test plot_predictions_binomial.", {
  g <- plot_predictions_binomial(rbinom(100, 1, 0.25), rbinom(100, 1, 0.25))
  expect_equal(class(g), c("gg", "ggplot"))
})

test_that("Test plot_model_performance_gaussian_mean_squared_error.", {
  g <- plot_model_performance_gaussian_mse_score(rnorm(100))
  expect_equal(class(g), c("gg", "ggplot"))
})

test_that("Test plot_model_performance_gaussian_r2_score.", {
  g <- plot_model_performance_gaussian_r2_score(runif(100))
  expect_equal(class(g), c("gg", "ggplot"))
})

test_that("Test plot_model_performance_gaussian_cor_score.", {
  g <- plot_model_performance_gaussian_correlation_score(runif(100))
  expect_equal(class(g), c("gg", "ggplot"))
})

test_that("Test plot_model_performance_binomial_area_under_curve.", {
  g <- plot_model_performance_binomial_auc_score(runif(100))
  expect_equal(class(g), c("gg", "ggplot"))
})
