library(easyml) # https://github.com/CCS-Lab/easyml

# Load data
data("cocaine_dependence", package = "easyml")

# Analyze data
easy_glmnet(cocaine_dependence, "DIAGNOSIS", 
            family = "binomial", exclude_variables = c("subject"), 
            categorical_variables = c("Male"), 
            n_samples = 50, n_divisions = 50, n_iterations = 5, 
            random_state = 1, progress_bar = TRUE, parallel = TRUE)

# Analyze data
system.time(easy_glmnet(cocaine_dependence, "DIAGNOSIS", 
                        family = "binomial", exclude_variables = c("subject"), 
                        categorical_variables = c("Male"), 
                        n_samples = 100, n_divisions = 100, n_iterations = 1, 
                        random_state = 1, progress_bar = TRUE, parallel = FALSE)
)

# Analyze data
options(mc.cores = parallel::detectCores())
system.time(easy_glmnet(cocaine_dependence, "DIAGNOSIS", 
                        family = "binomial", exclude_variables = c("subject"), 
                        categorical_variables = c("Male"), 
                        n_samples = 100, n_divisions = 100, n_iterations = 1, 
                        random_state = 1, progress_bar = TRUE, parallel = TRUE)
)
