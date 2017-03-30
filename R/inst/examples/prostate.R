library(easyml) # https://github.com/CCS-Lab/easyml

# Load data
data("prostate", package = "easyml")

# Settings
.n_samples <- 10
.n_divisions <- 5
.n_iterations <- 2
.n_core <- 1

# Analyze data
glmnet_results <- easy_glmnet(prostate, "lpsa", 
                              n_samples = .n_samples, n_divisions = .n_divisions, 
                              n_iterations = .n_iterations, random_state = 1, n_core = .n_core)

glinternet_results <- easy_glinternet(prostate, "lpsa", 
                                      n_samples = .n_samples, n_divisions = .n_divisions, 
                                      n_iterations = .n_iterations, random_state = 1, n_core = .n_core)

random_forest_results <- easy_random_forest(prostate, "lpsa", 
                                            n_samples = .n_samples, n_divisions = .n_divisions, 
                                            n_iterations = .n_iterations, random_state = 1, n_core = .n_core)

support_vector_machine_results <- easy_support_vector_machine(prostate, "lpsa", 
                                                              preprocess = preprocess_scale, 
                                                              n_samples = .n_samples, n_divisions = .n_divisions, 
                                                              n_iterations = .n_iterations, random_state = 1, n_core = .n_core)

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
