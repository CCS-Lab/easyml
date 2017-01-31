library(easyml)
context("preprocess")

test_that("Test preprocess_identity.", {
  expect_equal(preprocess_identity(identity), identity)
})

test_that("Test preprocess_scale.", {
  # set.seed(12345)
  # preprocess_scale(list(X = mtcars))
  # expect_equal()
})
