library(easyml) # https://github.com/CCS-Lab/easyml

# Load data
data("cocaine", package = "easyml")

# Analyze data
easy_glmnet(cocaine, dependent_variable = "DIAGNOSIS",
            family = "binomial", exclude_variables = c("subject", "AGE"))
