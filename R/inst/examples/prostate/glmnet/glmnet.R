library(easyml) # https://github.com/CCS-Lab/easyml

# Load data
data("prostate", package = "easyml")

# Analyze data
easy_glmnet(prostate, "lpsa", 
            n_samples = 10, n_divisions = 10, n_iterations = 10, 
            random_state = 1, n_core = 8)
