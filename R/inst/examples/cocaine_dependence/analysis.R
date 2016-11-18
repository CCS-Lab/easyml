library(easyml) # https://github.com/CCS-Lab/easyml

# Load data
data("cocaine_dependence", package = "easyml")

# Analyze data
easy_glmnet(cocaine_dependence, dependent_variable = "DIAGNOSIS",
            family = "binomial", exclude_variables = c("subject", "AGE"), 
            categorical_variables = c("Male"))
