library(easyml) # https://github.com/CCS-Lab/easyml

# Load data
data("cocaine_dependence", package = "easyml")

# Settings
.n_samples <- 1000L
.n_divisions <- 10L
.n_iterations <- 2L

# Analyze data
glmnet_results <- easy_glmnet(bar, "diagnosis", 
                              family = "binomial", exclude_variables = c("subject"), 
                              categorical_variables = c("male"), 
                              n_samples = .n_samples, n_divisions = .n_divisions, 
                              n_iterations = .n_iterations, random_state = 12345, n_core = 8, 
                              alpha = 1, nlambda = 200)

random_forest_results <- easy_random_forest(cocaine_dependence, "diagnosis", 
                                            family = "binomial", exclude_variables = c("subject"), 
                                            categorical_variables = c("male"), 
                                            n_samples = .n_samples, n_divisions = .n_divisions, 
                                            n_iterations = .n_iterations, random_state = 1, n_core = 8)

support_vector_machine_results <- easy_support_vector_machine(cocaine_dependence, "diagnosis", 
                                                              family = "binomial", preprocess = preprocess_scale, 
                                                              exclude_variables = c("subject"), 
                                                              categorical_variables = c("male"), 
                                                              n_samples = .n_samples, n_divisions = .n_divisions, 
                                                              n_iterations = .n_iterations, random_state = 1, n_core = 8)
