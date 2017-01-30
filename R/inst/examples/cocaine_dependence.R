library(easyml) # https://github.com/CCS-Lab/easyml

# Load data
data("cocaine_dependence", package = "easyml")

# Settings
.n_samples <- 10L
.n_divisions <- 10L
.n_iterations <- 2L

# Analyze data
glmnet_results <- easy_glmnet(cocaine_dependence, "diagnosis", 
                              family = "binomial", exclude_variables = c("subject"), 
                              categorical_variables = c("male"), preprocess = preprocess_scaler, 
                              n_samples = .n_samples, n_divisions = .n_divisions, 
                              n_iterations = .n_iterations, random_state = 1, n_core = 8, 
                              alpha = 1)

random_forest_results <- easy_random_forest(cocaine_dependence, "diagnosis", 
                                            family = "binomial", exclude_variables = c("subject"), 
                                            categorical_variables = c("male"), 
                                            n_samples = .n_samples, n_divisions = .n_divisions, 
                                            n_iterations = .n_iterations, random_state = 1, n_core = 8)

support_vector_machine_results <- easy_support_vector_machine(cocaine_dependence, "diagnosis", 
                                                              family = "binomial", preprocess = preprocess_scaler, 
                                                              exclude_variables = c("subject"), 
                                                              categorical_variables = c("male"), 
                                                              n_samples = .n_samples, n_divisions = .n_divisions, 
                                                              n_iterations = .n_iterations, random_state = 1, n_core = 8)
