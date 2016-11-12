library(easyml) # https://github.com/CCS-Lab/easyml

# Load data
data("cocaine", package = "easyml")

# Analyze data
easy_glmnet(data = cocaine,
            dependent_variable = "DIAGNOSIS",
            family = "binomial",
            exclude_variables = c("subject", "AGE"))



# DEBUG
data = cocaine
dependent_variable = "DIAGNOSIS"
family = "binomial"
exclude_variables = c("subject", "AGE")
train_size = 0.667
n_divisions = 1000
n_iterations = 10
n_samples = 1000
out_directory = '.'
random_state = NULL
