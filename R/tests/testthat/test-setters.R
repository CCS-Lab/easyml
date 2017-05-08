library(easyml)
library(withr)
context("setters")

test_that("Test set_random_state.", {
  expect_equal(set_random_state(NULL), NULL)
  expect_equal(set_random_state(12345), NULL)
  
  set_random_state(12345)
  a <- .Random.seed
  set_random_state(123456)
  b <- .Random.seed
  set_random_state(12345)
  c <- .Random.seed
  set_random_state(123456)
  d <- .Random.seed
  
  expect_equal(a, c)
  expect_equal(b, d)
})

test_that("Test set_parallel.", {
  expect_equal(set_parallel(1), FALSE)
  expect_equal(set_parallel(2), TRUE)
  expect_error(set_parallel(0))
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
  letters <- c("a", "b", "c")
  expect_equal(set_categorical_variables(letters, categorical_variables = NULL), NULL)
  expect_equal(set_categorical_variables(letters, categorical_variables = "c"), 
               c(FALSE, FALSE, TRUE))
})

test_that("Test set_dependent_variable.", {
  expect_equal(set_dependent_variable(mtcars, "mpg"), mtcars[, "mpg"])
})

test_that("Test set_independent_variables.", {
  expect_equal(set_independent_variables(mtcars, "mpg"), mtcars[, -1])
})

test_that("Test set_resample.", {
  expect_error(set_resample(NULL, NULL))
  expect_equal(set_resample(NULL, "gaussian"), resample_simple_train_test_split)
  expect_equal(set_resample(NULL, "binomial"), resample_stratified_class_train_test_split)
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

test_that("Test set_measure.", {
  expect_error(set_measure(NULL, NULL, NULL))
  expect_equal(set_measure(NULL, "glmnet", "gaussian"), measure_correlation_score)
  expect_equal(set_measure(NULL, "glmnet", "binomial"), measure_auc_score)
  expect_equal(set_measure(NULL, "random_forest", "gaussian"), measure_correlation_score)
  expect_equal(set_measure(NULL, "random_forest", "binomial"), measure_auc_score)
  expect_equal(set_measure(NULL, "support_vector_machine", "gaussian"), measure_correlation_score)
  expect_equal(set_measure(NULL, "support_vector_machine", "binomial"), measure_auc_score)
  expect_equal(set_measure(identity), identity)
  expect_equal(set_measure(identity, "", NULL), identity)
  expect_equal(set_measure(identity, NULL, ""), identity)
  expect_equal(set_measure(identity, "", ""), identity)
})

test_that("Test set_plot_predictions.", {
  expect_error(set_plot_predictions("", ""))
  expect_equal(set_plot_predictions("glmnet", "gaussian"), plot_predictions_gaussian)
  expect_equal(set_plot_predictions("glmnet", "binomial"), plot_predictions_binomial)
  expect_equal(set_plot_predictions("random_forest", "gaussian"), plot_predictions_gaussian)
  expect_equal(set_plot_predictions("random_forest", "binomial"), plot_predictions_binomial)
  expect_equal(set_plot_predictions("support_vector_machine", "gaussian"), plot_predictions_gaussian)
  expect_equal(set_plot_predictions("support_vector_machine", "binomial"), plot_predictions_binomial)
})

test_that("Test set_plot_model_performance.", {
  expect_error(set_plot_model_performance(""))
  expect_equal(set_plot_model_performance(measure_mse_score), plot_model_performance_gaussian_mse_score)
  expect_equal(set_plot_model_performance(measure_correlation_score), plot_model_performance_gaussian_correlation_score)
  expect_equal(set_plot_model_performance(measure_r2_score), plot_model_performance_gaussian_r2_score)
  expect_equal(set_plot_model_performance(measure_auc_score), plot_model_performance_binomial_auc_score)
})
