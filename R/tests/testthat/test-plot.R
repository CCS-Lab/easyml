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
  g <- plot_predictions_binomial(rbinom(100, 1, 0.25), rbinom(100, 1, 0.25))
  expect_equal(class(g), c("gg", "ggplot"))
})

test_that("Test plot_metrics_gaussian_mean_squared_error.", {
  g <- plot_metrics_gaussian_mean_squared_error(rnorm(100))
  expect_equal(class(g), c("gg", "ggplot"))
})

test_that("Test plot_metrics_gaussian_r2_score.", {
  g <- plot_metrics_gaussian_r2_score(runif(100))
  expect_equal(class(g), c("gg", "ggplot"))
})

test_that("Test plot_metrics_gaussian_cor_score.", {
  g <- plot_metrics_gaussian_cor_score(runif(100))
  expect_equal(class(g), c("gg", "ggplot"))
})

test_that("Test plot_metrics_binomial_area_under_curve.", {
  g <- plot_metrics_binomial_area_under_curve(runif(100))
  expect_equal(class(g), c("gg", "ggplot"))
})
