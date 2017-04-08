library(easyml)
context("support_vector_machine")

test_that("Test easy_support_vector_machine.", {
  # Load data
  data("cocaine_dependence", package = "easyml")
  data("prostate", package = "easyml")
  
  # Settings
  .n_samples <- 5
  .n_divisions <- 5
  .n_iterations <- 2
  .n_core <- 1
  
  # Test binomial
  results <- easy_support_vector_machine(cocaine_dependence, "diagnosis",
                                         family = "binomial", preprocess = preprocess_scale,
                                         exclude_variables = c("subject"),
                                         categorical_variables = c("male"),
                                         n_samples = .n_samples, n_divisions = .n_divisions,
                                         n_iterations = .n_iterations, random_state = 12345, n_core = .n_core)
  expect_equal(class(results), "easy_support_vector_machine")
  
  # Test gaussian
  suppressWarnings(results <- easy_support_vector_machine(prostate, "lpsa", 
                                                          preprocess = preprocess_scale, 
                                                          n_samples = .n_samples, n_divisions = .n_divisions, 
                                                          n_iterations = .n_iterations, random_state = 12345, n_core = .n_core))
  expect_equal(class(results), "easy_support_vector_machine")
})
