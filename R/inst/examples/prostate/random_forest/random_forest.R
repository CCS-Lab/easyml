library(easyml) # https://github.com/CCS-Lab/easyml

# Load data
data("prostate", package = "easyml")

# Analyze data
system.time(easy_random_forest(prostate, "lpsa", 
                               n_samples = 100, n_divisions = 100, n_iterations = 10, 
                               random_state = 1, n_core = 8)
)
