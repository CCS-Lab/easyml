library(easyml) # https://github.com/CCS-Lab/easyml

# Load data
data("cocaine_dependence", package = "easyml")

# Settings
.n_samples <- 1
.n_divisions <- 5
.n_iterations <- 1
.n_core <- 1

# Analyze data
results <- easy_glmnet(cocaine_dependence, "diagnosis",
                       family = "binomial",
                       resample = resample_stratified_class_train_test_split,
                       preprocess = preprocess_scale,
                       exclude_variables = c("subject"),
                       categorical_variables = c("male"),
                       n_samples = .n_samples, n_divisions = .n_divisions,
                       n_iterations = .n_iterations, random_state = 12345, n_core = .n_core,
                       model_args = list(alpha = 1, nlambda = 200))

library(ggplot2)
n <- 1000
df <- data.frame(y_true = results$y_train, 
                 y_pred = results$predictions_train)

ggplot(df, aes(x = y_pred, y = y_true)) + 
  geom_point() + 
  stat_smooth(method="glm", method.args = list(family = "binomial"), se=FALSE) + 
  scale_x_continuous("Predicted y values", limits = c(0, 1), 
                     breaks = seq(0, 1, 0.05), minor_breaks = seq(0, 1, 0.01)) + 
  scale_y_continuous("True y values", limits = c(0, 1), 
                     breaks = seq(0, 1, 0.05), minor_breaks = seq(0, 1, 0.01)) + 
  ggtitle("Actual vs. Predicted y values (ROC AUC = 0.90)") + 
  labs(subtitle = "Train Predictions") + 
  theme_bw()
