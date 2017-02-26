library(easyml)
library(withr)
context("setters")

test_that("Test set_random_state.", {
  expect_equal(set_random_state(), NULL)
})

test_that("Test set_coefficients_boolean.", {
  expect_equal(set_coefficients_boolean("glmnet"), TRUE)
  expect_equal(set_coefficients_boolean("random_forest"), FALSE)
  expect_equal(set_coefficients_boolean("support_vector_machine"), FALSE)
})

test_that("Test set_predictions_boolean.", {
  expect_equal(set_predictions_boolean("glmnet"), TRUE)
  expect_equal(set_predictions_boolean("random_forest"), TRUE)
  expect_equal(set_predictions_boolean("support_vector_machine"), TRUE)
})

test_that("Test set_variable_importances_boolean.", {
  expect_equal(set_variable_importances_boolean("glmnet"), FALSE)
  expect_equal(set_variable_importances_boolean("random_forest"), TRUE)
  expect_equal(set_variable_importances_boolean("support_vector_machine"), FALSE)
})

test_that("Test set_metrics_boolean.", {
  expect_equal(set_metrics_boolean("glmnet"), TRUE)
  expect_equal(set_metrics_boolean("random_forest"), TRUE)
  expect_equal(set_metrics_boolean("support_vector_machine"), TRUE)
})

test_that("Test set_parallel.", {
  expect_equal(set_parallel(1), FALSE)
  expect_equal(set_parallel(2), TRUE)
})

test_that("Test set_cores.", {
  options(mc.cores = NULL)
  with_options(
    c(mc.cores = NULL), 
    {
      set_cores(2)
      expect_identical(getOption("mc.cores"), 2)
    }
  )
})

test_that("Test set_looper.", {
  options(mc.cores = NULL)
  with_options(
    c(mc.cores = NULL), 
    {
      expected <- set_looper(TRUE, 2)
      expect_identical(getOption("mc.cores"), 2)
      expect_equal(expected, pbmcapply::pbmclapply)
    }
  )
  
  options(mc.cores = NULL)
  with_options(
    c(mc.cores = NULL), 
    {
      expected <- set_looper(TRUE, 1)
      expect_identical(getOption("mc.cores"), NULL)
      expect_equal(expected, pbapply::pblapply)
    }
  )
  
  options(mc.cores = NULL)
  with_options(
    c(mc.cores = NULL), 
    {
      expected <- set_looper(FALSE, 2)
      expect_identical(getOption("mc.cores"), 2)
      expect_equal(expected, parallel::mclapply)
    }
  )
  
  options(mc.cores = NULL)
  with_options(
    c(mc.cores = NULL), 
    {
      expected <- set_looper(FALSE, 1)
      expect_identical(getOption("mc.cores"), NULL)
      expect_equal(expected, base::lapply)
    }
  )
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

test_that("Test set_dependent_variable.", {
})

test_that("Test set_independent_variables.", {
})

test_that("Test set_resample.", {
  expect_error(set_resample(NULL, NULL))
  expect_equal(set_resample(NULL, "gaussian"), resample_simple_train_test_split)
  expect_equal(set_resample(NULL, "binomial"), resample_stratified_train_test_split)
  expect_equal(set_resample(identity, NULL), identity)
  expect_equal(set_resample(identity, "gaussian"), identity)
  expect_equal(set_resample(identity, "binomial"), identity)
})

test_that("Test set_preprocess.", {
  expect_error(set_preprocess(NULL, NULL))
  expect_equal(set_preprocess(NULL, "glmnet"), preprocess_scale)
  expect_equal(set_preprocess(NULL, "random_forest"), preprocess_identity)
  expect_equal(set_preprocess(NULL, "support_vector_machine"), preprocess_scale)
  expect_equal(set_resample(identity, NULL), identity)
  expect_equal(set_resample(identity, "glmnet"), identity)
  expect_equal(set_resample(identity, "random_forest"), identity)
  expect_equal(set_resample(identity, "support_vector_machine"), identity)
})

# test_that("Test set_measure.", {
#   expect_error(set_measure(NULL, NULL, NULL))
#   expect_equal(set_measure(NULL, "glmnet", "gaussian"), F)
#   expect_equal(set_measure(NULL, "glmnet", "binomial"), F)
#   expect_equal(set_measure(NULL, "random_forest", "gaussian"), F)
#   expect_equal(set_measure(NULL, "random_forest", "binomial"), F)
#   expect_equal(set_measure(NULL, "support_vector_machine", "gaussian"), F)
#   expect_equal(set_measure(NULL, "support_vector_machine", "binomial"), F)
#   expect_equal(set_measure(identity), identity)
#   expect_equal(set_measure(identity, "", NULL), identity)
#   expect_equal(set_measure(identity, NULL, ""), identity)
#   expect_equal(set_measure(identity, "", ""), identity)
# })

test_that("Test set_fit_model.", {
})

test_that("Test set_extract_coefficients.", {
})

test_that("Test set_predict_model.", {
})

test_that("Test set_plot_predictions.", {
})

test_that("Test set_plot_metrics.", {
})
