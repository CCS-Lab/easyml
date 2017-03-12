library(easyml) # https://github.com/CCS-Lab/easyml

# Load data
data("prostate", package = "easyml")

# Settings
.n_samples <- 10L
.n_divisions <- 10L
.n_iterations <- 2L

# Analyze data
glmnet_results <- easy_glmnet(prostate, "lpsa", 
                              n_samples = .n_samples, n_divisions = .n_divisions, 
                              n_iterations = .n_iterations, random_state = 1L, n_core = 8L)

glinternet_results <- easy_glinternet(prostate, "lpsa", 
                                      n_samples = .n_samples, n_divisions = .n_divisions, 
                                      n_iterations = .n_iterations, random_state = 1L, n_core = 8L)

random_forest_results <- easy_random_forest(prostate, "lpsa", 
                                            n_samples = .n_samples, n_divisions = .n_divisions, 
                                            n_iterations = .n_iterations, random_state = 1L, n_core = 8L)

support_vector_machine_results <- easy_support_vector_machine(prostate, "lpsa", 
                                                              preprocess = preprocess_scaler, 
                                                              n_samples = .n_samples, n_divisions = .n_divisions, 
                                                              n_iterations = .n_iterations, random_state = 1L, n_core = 8L)
