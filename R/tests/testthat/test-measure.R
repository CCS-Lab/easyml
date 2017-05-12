library(easyml)
context("measure")

test_that("Test measure_mean_squared_error.", {
  expect_equal(measure_mse_score(1, 1), 0)
})

test_that("Test measure_r2_score.", {
  expect_equal(measure_r2_score(c(1, 2, 3), c(1, 2, 3)), 1)
})

test_that("Test measure_cor_score.", {
  expect_equal(measure_correlation_score(c(1, 2, 3), c(1, 2, 3)), 1)
})

test_that("Test measure_area_under_curve.", {
  expect_equal(measure_auc_score(c(0, 1), c(0, 1)), 1)
})
