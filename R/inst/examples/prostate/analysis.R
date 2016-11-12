library(easyml) # https://github.com/CCS-Lab/easyml

# Load data
data("prostate", package = "easyml")

# Analyze data
easy_glmnet(prostate, dependent_variable = "lpsa")
