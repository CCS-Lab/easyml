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

# Load data
data("cocaine_dependence", package = "easyml")
cocaine_dependence <- cocaine_dependence

# Settings
.n_samples <- 10
.n_divisions <- 10
.n_iterations <- 2

# Analyze data
glmnet_results <- easy_glmnet(cocaine_dependence, "diagnosis", 
                              family = "binomial", 
                              resample = resample_stratified_class_train_test_split, 
                              preprocess = preprocess_scale, 
                              exclude_variables = c("subject"), 
                              categorical_variables = c("male"), 
                              n_samples = .n_samples, n_divisions = .n_divisions, 
                              n_iterations = .n_iterations, random_state = 12345, n_core = 1, 
                              alpha = 1, nlambda = 200)

test_that("Test process_coefficients.", {
  coefficients_processed <- process_coefficients(glmnet_results$coefficients)
  expect_equal(class(coefficients_processed), c("data.frame"))
})
