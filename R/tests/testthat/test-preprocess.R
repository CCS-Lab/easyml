library(easyml)
context("preprocess")

test_that("Test preprocess_identity.", {
  expect_equal(preprocess_identity(identity), identity)
})

test_that("Test preprocess_scaler_.", {
  # set.seed(12345)
  # preprocess_scaler(list(X = mtcars))
  # expect_equal()
})
