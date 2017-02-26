library(easyml)
context("preprocess")

test_that("Test preprocess_identity.", {
  expect_equal(preprocess_identity(identity), identity)
  
  actual <- preprocess_identity(identity, categorical_variables = c("a", "b"))
  expect_equal(actual, identity)
})

test_that("Test preprocess_scale.", {
  # X
  set.seed(12345)
  actual <- preprocess_scale(list(X = mtcars))
  expected <- list(X = data.frame(base::scale(mtcars)))
  expect_equal(actual, expected)
  
  # X, with categorical_variables
  set.seed(12345)
  cat_vars <- c(rep(FALSE, 10), TRUE)
  actual <- preprocess_scale(list(X = mtcars), 
                             categorical_variables = cat_vars)
  X <- base::scale(mtcars[, -11])
  expected <- list(X = cbind(mtcars[, 11, drop = FALSE], data.frame(X)))
  expect_equal(actual, expected)

  # X_train and X_test
  actual <- preprocess_scale(list(X_train = mtcars[1:25, ], 
                                  X_test = mtcars[26:32, ]))
  
  X_train <- base::scale(mtcars[1:25, ])
  X_test <- base::scale(mtcars[26:32, ], 
                        attr(X_train, "scaled:center"), 
                        attr(X_train, "scaled:scale"))
  expected <- list(X_train = data.frame(X_train), 
                   X_test = data.frame(X_test))
  expect_equal(actual, expected)
  
  # X_train and X_test, with categorical_variables
  cat_vars <- c(rep(FALSE, 10), TRUE)
  actual <- preprocess_scale(list(X_train = mtcars[1:25, ], 
                                  X_test = mtcars[26:32, ]), 
                             categorical_variables = cat_vars)
  
  X_train <- base::scale(mtcars[1:25, -11])
  X_test <- base::scale(mtcars[26:32, -11], 
                        attr(X_train, "scaled:center"), 
                        attr(X_train, "scaled:scale"))
  expected <- list(X_train = cbind(mtcars[1:25, 11, drop = FALSE], data.frame(X_train)), 
                   X_test = cbind(mtcars[26:32, 11, drop = FALSE], data.frame(X_test)))
  expect_equal(actual, expected)
  
  expect_error(preprocess_scale(list(X = 1, Y = 2, Z = 3)))
})
