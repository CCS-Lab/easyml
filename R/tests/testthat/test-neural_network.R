library(easyml)
context("neural_network")

test_that("Test easy_neural_network.", {
  # Load data
  data("cocaine_dependence", package = "easyml")
  data("prostate", package = "easyml")
  
  # Settings
  .n_samples <- 5
  .n_divisions <- 5
  .n_iterations <- 2
  .n_core <- 1
  
  # Test binomial
  results <- easy_neural_network(cocaine_dependence, "diagnosis",
                                 family = "binomial", preprocess = preprocess_scale,
                                 exclude_variables = c("subject"),
                                 categorical_variables = c("male"),
                                 n_samples = .n_samples, n_divisions = .n_divisions,
                                 n_iterations = .n_iterations, random_state = 12345, n_core = .n_core, 
                                 model_args = list(size = c(40)))
  expect_equal(class(results), "easy_neural_network")
  
  # Test gaussian
  results <- easy_neural_network(prostate, "lpsa", 
                                 preprocess = preprocess_scale, 
                                 n_samples = .n_samples, n_divisions = .n_divisions, 
                                 n_iterations = .n_iterations, random_state = 12345, n_core = .n_core, 
                                 model_args = list(size = c(40)))
  expect_equal(class(results), "easy_neural_network")
})
