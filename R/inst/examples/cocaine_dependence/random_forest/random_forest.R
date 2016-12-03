library(easyml) # https://github.com/CCS-Lab/easyml

# Load data
data("cocaine_dependence", package = "easyml")

# Analyze data
system.time(easy_random_forest(cocaine_dependence, "DIAGNOSIS", 
                               family = "binomial", exclude_variables = c("subject"), 
                               categorical_variables = c("Male"), 
                               n_samples = 100, n_divisions = 100, n_iterations = 10, 
                               random_state = 1, n_core = 8)
)
