library(easyml)
context("plot")

test_that("Test plot_coefficients_processed.", {
})

test_that("Test plot_variable_importances.", {
})

test_that("Test plot_predictions_gaussian.", {
  g <- plot_predictions_gaussian(rnorm(100, 0, 1), rnorm(100, 0, 1))
  expect_equal(class(g), c("gg", "ggplot"))
})

test_that("Test plot_predictions_binomial.", {
})

test_that("Test plot_metrics_gaussian_mean_squared_error.", {
})

test_that("Test plot_metrics_gaussian_r2_score.", {
})

test_that("Test plot_metrics_gaussian_cor_score.", {
})

test_that("Test plot_metrics_binomial_area_under_curve.", {
})
