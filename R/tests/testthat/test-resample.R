library(easyml)
context("resample")

# Load data
data("cocaine_dependence", package = "easyml")
data("prostate", package = "easyml")

# Settings
.n_samples <- 5
.n_divisions <- 5
.n_iterations <- 2
.n_core <- 1

test_that("Test resample_simple_train_test_split.", {
  results <- easy_random_forest(prostate, "lpsa", 
                                resample = resample_simple_train_test_split, 
                                n_samples = .n_samples, n_divisions = .n_divisions, 
                                n_iterations = .n_iterations, random_state = 12345, n_core = .n_core)
  expect_equal(class(results), "easy_random_forest")
})

test_that("Test resample_stratified_simple_train_test_split.", {
  results <- easy_random_forest(cocaine_dependence, "diagnosis",
                                family = "binomial", 
                                resample = resample_stratified_simple_train_test_split, 
                                exclude_variables = c("subject"),
                                categorical_variables = c("male"),
                                n_samples = .n_samples, n_divisions = .n_divisions,
                                n_iterations = .n_iterations, random_state = 12345, n_core = .n_core, 
                                foldid = rep(x = 1:3, 18))
  expect_equal(class(results), "easy_random_forest")
})

test_that("Test resample_stratified_class_train_test_split.", {
  results <- easy_random_forest(cocaine_dependence, "diagnosis",
                                family = "binomial", 
                                resample = resample_stratified_class_train_test_split, 
                                exclude_variables = c("subject"),
                                categorical_variables = c("male"),
                                n_samples = .n_samples, n_divisions = .n_divisions,
                                n_iterations = .n_iterations, random_state = 12345, n_core = .n_core)
  expect_equal(class(results), "easy_random_forest")
})

test_that("Test resample_fold_train_test_split.", {
  results <- easy_random_forest(cocaine_dependence, "diagnosis",
                                family = "binomial",
                                resample = resample_fold_train_test_split,
                                exclude_variables = c("subject"),
                                categorical_variables = c("male"),
                                n_samples = .n_samples, n_divisions = .n_divisions,
                                n_iterations = .n_iterations, random_state = 12345, n_core = .n_core,
                                foldid = rep(x = 1:3, 18))
  expect_equal(class(results), "easy_random_forest")
})
