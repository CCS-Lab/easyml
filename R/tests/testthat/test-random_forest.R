library(easyml)
context("random_forest")

test_that("Test easy_random_forest.", {
  # Load data
  data("cocaine_dependence", package = "easyml")
  data("prostate", package = "easyml")
  
  # Settings
  .n_samples <- 5
  .n_divisions <- 5
  .n_iterations <- 2
  .n_core <- 1
  
  # Test binomial
  results <- easy_random_forest(cocaine_dependence, "diagnosis",
                                family = "binomial", 
                                exclude_variables = c("subject"),
                                categorical_variables = c("male"),
                                n_samples = .n_samples, n_divisions = .n_divisions,
                                n_iterations = .n_iterations, random_state = 12345, n_core = .n_core)
  expect_equal(class(results), "easy_random_forest")
  
  # Test gaussian
  results <- easy_random_forest(prostate, "lpsa", 
                                n_samples = .n_samples, n_divisions = .n_divisions, 
                                n_iterations = .n_iterations, random_state = 12345, n_core = .n_core)
  expect_equal(class(results), "easy_random_forest")
})
