library(easyml) # https://github.com/CCS-Lab/easyml

# Load data
data("prostate", package = "easyml")

# Settings
.n_samples <- 10
.n_divisions <- 10
.n_iterations <- 2

# Analyze data
glmnet_results <- easy_glmnet(prostate, "lpsa", 
                              n_samples = .n_samples, n_divisions = .n_divisions, 
                              n_iterations = .n_iterations, random_state = 1, n_core = 8)

glinternet_results <- easy_glinternet(prostate, "lpsa", 
                                      n_samples = .n_samples, n_divisions = .n_divisions, 
                                      n_iterations = .n_iterations, random_state = 1, n_core = 8)

random_forest_results <- easy_random_forest(prostate, "lpsa", 
                                            n_samples = .n_samples, n_divisions = .n_divisions, 
                                            n_iterations = .n_iterations, random_state = 1, n_core = 8)

support_vector_machine_results <- easy_support_vector_machine(prostate, "lpsa", 
                                                              preprocess = preprocess_scaler, 
                                                              n_samples = .n_samples, n_divisions = .n_divisions, 
                                                              n_iterations = .n_iterations, random_state = 1, n_core = 8)
