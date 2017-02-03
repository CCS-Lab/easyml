library(easyml)
context("setters")

test_that("Test set_parallel.", {
  expect_equal(set_parallel(1), FALSE)
  expect_equal(set_parallel(2), TRUE)
})

test_that("Test set_looper_.", {
  expect_equal(set_looper_(TRUE, TRUE), pbmcapply::pbmclapply)
  expect_equal(set_looper_(TRUE, FALSE), pbapply::pblapply)
  expect_equal(set_looper_(FALSE, TRUE), parallel::mclapply)
  expect_equal(set_looper_(FALSE, FALSE), base::lapply)
})

foo <- data.frame(y = c(0, 1, 0, 1), 
                  x1 = c(1, 2, 3, 4), 
                  x2 = c("m", "f", "m", "f"), 
                  x3 = c(5, 6, 7, 8))

test_that("Test set_column_names.", {
  expect_equal(set_column_names(colnames(foo), "y", 
                                exclude_variables = "x3", 
                                preprocess = preprocess_scale, 
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


test_that("Test set_preprocess.", {
  expect_equal(set_preprocess(NULL), preprocess_identity)
  expect_equal(set_preprocess(identity), identity)
})

test_that("Test set_random_state.", {
})

test_that("Test set_resample.", {
  # expect_equal(set_resample(NULL, NULL), preprocess_identity)
  expect_equal(set_resample(NULL, "gaussian"), resample_simple_train_test_split)
  expect_equal(set_resample(NULL, "binomial"), resample_stratified_train_test_split)
  expect_equal(set_resample(identity), identity)
})

test_that("Test set_dependent_variable.", {
})

test_that("Test set_independent_variables.", {
})
