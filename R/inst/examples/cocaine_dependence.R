library(easyml) # https://github.com/CCS-Lab/easyml

# Load data
data("cocaine_dependence", package = "easyml")

# Settings
.n_samples <- 100
.n_divisions <- 5
.n_iterations <- 2
.n_core <- 1

# Analyze data
glmnet_results <- easy_glmnet(cocaine_dependence, "diagnosis",
                              family = "binomial",
                              resample = resample_stratified_class_train_test_split,
                              preprocess = preprocess_scale,
                              exclude_variables = c("subject"),
                              categorical_variables = c("male"),
                              n_samples = .n_samples, n_divisions = .n_divisions,
                              n_iterations = .n_iterations, random_state = 12345, n_core = .n_core,
                              model_args = list(alpha = 1, nlambda = 200))

glinternet_results <- easy_glinternet(cocaine_dependence, "diagnosis",
                                      family = "binomial",
                                      resample = resample_stratified_class_train_test_split,
                                      preprocess = preprocess_scale,
                                      exclude_variables = c("subject"),
                                      categorical_variables = c("male"),
                                      n_samples = .n_samples, n_divisions = .n_divisions,
                                      n_iterations = .n_iterations, random_state = 12345, n_core = .n_core)

random_forest_results <- easy_random_forest(cocaine_dependence, "diagnosis",
                                            family = "binomial", exclude_variables = c("subject"),
                                            categorical_variables = c("male"),
                                            n_samples = .n_samples, n_divisions = .n_divisions,
                                            n_iterations = .n_iterations, random_state = 12345, n_core = .n_core)

support_vector_machine_results <- easy_support_vector_machine(cocaine_dependence, "diagnosis",
                                                              family = "binomial", preprocess = preprocess_scale,
                                                              exclude_variables = c("subject"),
                                                              categorical_variables = c("male"),
                                                              n_samples = .n_samples, n_divisions = .n_divisions,
                                                              n_iterations = .n_iterations, random_state = 12345, n_core = .n_core)

neural_network_results <- easy_neural_network(cocaine_dependence, "diagnosis",
                                              family = "binomial", preprocess = preprocess_scale,
                                              exclude_variables = c("subject"),
                                              categorical_variables = c("male"),
                                              n_samples = .n_samples, n_divisions = .n_divisions,
                                              n_iterations = .n_iterations, random_state = 12345, n_core = .n_core, 
                                              model_args = list(size = c(40)))

library(darch)
deep_neural_network_results <- easy_deep_neural_network(cocaine_dependence, "diagnosis",
                                                        family = "binomial", preprocess = preprocess_scale,
                                                        exclude_variables = c("subject"),
                                                        categorical_variables = c("male"),
                                                        n_samples = .n_samples, n_divisions = .n_divisions,
                                                        n_iterations = .n_iterations, random_state = 12345, n_core = .n_core)

model_args <- list(size = 5, linout = TRUE, trace = FALSE)
b <- easy_avNNet(cocaine_dependence, "diagnosis", 
                 family = "binomial", 
                 preprocess = preprocess_scale, 
                 exclude_variables = c("subject"),
                 categorical_variables = c("male"),
                 n_samples = 10,  n_divisions = 10, 
                 n_iterations = 10, random_state = 12345, 
                 n_core = 1, model_args = model_args)
