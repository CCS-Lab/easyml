library(easyml) # https://github.com/CCS-Lab/easyml

# Load data
data("prostate", package = "easyml")

# Analyze data
easy_glmnet(prostate, "lpsa", 
            n_samples = 100, n_divisions = 50, n_iterations = 5, 
            random_state = 1, n_core = 8)

# Analyze data
system.time(easy_glmnet(prostate, "lpsa", 
                        n_samples = 500, n_divisions = 500, n_iterations = 1, 
                        random_state = 1, n_core = 1)
)

# Analyze data
system.time(easy_glmnet(prostate, "lpsa", 
                        n_samples = 500, n_divisions = 500, n_iterations = 1, 
                        random_state = 1, n_core = 8)
)
