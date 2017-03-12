library(easyml)
context("utils")

test_that("Test reduce_cores.", {
  expect_equal(reduce_cores(2, 4), 2)
  expect_equal(reduce_cores(4, 4), 4)
  expect_equal(reduce_cores(8, 4), 4)
})

test_that("Test remove_variables.", {
  expect_equal(remove_variables(mtcars, "mpg"), mtcars[, -1])
})

test_that("Test correlation_test.", {
  m <- correlation_test(mtcars)
  expect_equal(length(m), 3)
  expect_true(nrow(m$p_value) == ncol(m$p_value))
})
