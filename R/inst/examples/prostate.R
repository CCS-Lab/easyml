library(easyml) # https://github.com/CCS-Lab/easyml

# Load data
data("prostate", package = "easyml")

# Settings
.n_samples <- 50
.n_divisions <- 50
.n_iterations <- 2
.n_core <- 1

results <- easy_random_forest(prostate, "lpsa", 
                              n_samples = .n_samples, n_divisions = .n_divisions, 
                              n_iterations = .n_iterations, random_state = 1, n_core = .n_core)

results$plot_variable_importances
results$plot_predictions_single_train_test_split_train
results$plot_predictions_single_train_test_split_test
results$plot_model_performance_train
results$plot_model_performance_test

# Analyze data
results <- easy_glmnet(prostate, "lpsa", 
                       n_samples = .n_samples, n_divisions = .n_divisions, 
                       n_iterations = .n_iterations, random_state = 1, n_core = .n_core, 
                       model_args = list(alpha = 1, nlambda = 200))
results$plot_coefficients
results$plot_predictions_single_train_test_split_train
results$plot_predictions_single_train_test_split_test
results$plot_model_performance_train
results$plot_model_performance_test

# glinternet_results <- easy_glinternet(prostate, "lpsa", 
#                                       n_samples = .n_samples, n_divisions = .n_divisions, 
#                                       n_iterations = .n_iterations, random_state = 1, n_core = .n_core)

neural_network_results <- easy_neural_network(prostate, "lpsa", 
                                              preprocess = preprocess_scale, 
                                              measure = measure_r2_score, 
                                              n_samples = .n_samples, n_divisions = .n_divisions, 
                                              n_iterations = .n_iterations, random_state = 1, n_core = .n_core, 
                                              model_args = list(size = 50, decay = 1))

library(darch)
deep_neural_network_results <- easy_deep_neural_network(prostate, "lpsa", 
                                                        preprocess = preprocess_scale, 
                                                        measure = measure_r2_score, 
                                                        n_samples = .n_samples, n_divisions = .n_divisions, 
                                                        n_iterations = .n_iterations, random_state = 1, n_core = .n_core)

model_args <- list(size = 5, linout = TRUE, trace = FALSE)
g <- easy_avNNet(prostate, "lpsa", 
                 preprocess = preprocess_scale,
                 n_samples = 10,  n_divisions = 10, 
                 n_iterations = 10, 
                 random_state = 12345, n_core = 1, 
                 model_args = model_args)