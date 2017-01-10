library(easyml)
context("utils")

test_that("Test identify_parallel.", {
  expect_equal(identify_parallel(1), FALSE)
  expect_equal(identify_parallel(2), TRUE)
})

test_that("Test reduce_cores.", {
  expect_equal(reduce_cores(2, 4), 2)
  expect_equal(reduce_cores(4, 4), 4)
  expect_equal(reduce_cores(8, 4), 4)
})

test_that("Test identify_looper_.", {
  expect_equal(identify_looper_(TRUE, TRUE), pbmcapply::pbmclapply)
  expect_equal(identify_looper_(TRUE, FALSE), pbapply::pblapply)
  expect_equal(identify_looper_(FALSE, TRUE), parallel::mclapply)
  expect_equal(identify_looper_(FALSE, FALSE), base::lapply)
})

foo <- data.frame(y = c(0, 1, 0, 1), 
                  x1 = c(1, 2, 3, 4), 
                  x2 = c("m", "f", "m", "f"), 
                  x3 = c(5, 6, 7, 8))

test_that("Test set_column_names.", {
  expect_equal(set_column_names(colnames(foo), "y", 
                                exclude_variables = "x3", 
                                preprocessor = preprocess_scaler, 
                                categorical_variables = "x2"), 
               c("x2", "x1"))
  expect_equal(set_column_names(colnames(foo), "y", 
                                exclude_variables = "x3", 
                                categorical_variables = "x2"), 
               c("x1", "x2"))
  expect_equal(set_column_names(colnames(foo), "y", 
                                categorical_variables = "x2"), 
               c("x1", "x2", "x3"))
})

test_that("Test set_categorical_variables.", {
})


test_that("Test set_preprocessor.", {
  expect_equal(set_preprocessor(NULL), preprocess_identity)
  expect_equal(set_preprocessor(identity), identity)
})

test_that("Test set_random_state.", {
})

test_that("Test set_sampler.", {
  # expect_equal(set_sampler(NULL, NULL), preprocess_identity)
  expect_equal(set_sampler(NULL, "gaussian"), resample_simple_train_test_split)
  expect_equal(set_sampler(NULL, "binomial"), resample_stratified_train_test_split)
  expect_equal(set_sampler(identity), identity)
})

test_that("Test isolate_dependent_variable.", {
})

test_that("Test isolate_independent_variables.", {
})

test_that("Test remove_variables.", {
})
