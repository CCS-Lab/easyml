library(easyml) # devtools::install_github("CCS-Lab/easyml", subdir = "R")
library(glmnet)

data("prostate", package = "easyml")

# Set X, y, and scale X
X <- as.matrix(prostate[, -9])
y <- prostate[, 9]
X_scaled <- scale(X)

# no seed
m <- 10
n <- ncol(X)
Z <- matrix(NA, nrow = m, ncol = n)
for (i in (1:m)) {
  model_cv <- cv.glmnet(X_scaled, y, standardize = FALSE)
  model <- glmnet(X_scaled, y)
  coefs <- coef(model, s = model_cv$lambda.min)
  Z[i, ] <- as.numeric(coefs)[-1]
}
print(Z)

# Seed set at outer level
set.seed(43210)
m <- 10
n <- ncol(X)
Z <- matrix(NA, nrow = m, ncol = n)
for (i in (1:m)) {
  model_cv <- cv.glmnet(X_scaled, y, standardize = FALSE)
  model <- glmnet(X_scaled, y)
  coefs <- coef(model, s = model_cv$lambda.min)
  Z[i, ] <- as.numeric(coefs)[-1]
}
print(Z)

# Seed set at inner level
Z <- matrix(NA, nrow = m, ncol = n)
for (i in (1:m)) {
  set.seed(43210)
  model_cv <- cv.glmnet(X_scaled, y, standardize = FALSE)
  model <- glmnet(X_scaled, y)
  coefs <- coef(model, s = model_cv$lambda.min)
  Z[i, ] <- as.numeric(coefs)[-1]
}
print(Z)

# Different seed set each loop at inner level
Z <- matrix(NA, nrow = m, ncol = n)
for (i in (1:m)) {
  set.seed(i)
  model_cv <- cv.glmnet(X_scaled, y, standardize = FALSE)
  model <- glmnet(X_scaled, y)
  coefs <- coef(model, s = model_cv$lambda.min)
  Z[i, ] <- as.numeric(coefs)[-1]
}
print(Z)
