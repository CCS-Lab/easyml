library(easyml) # https://github.com/CCS-Lab/easyml

# Load data
data("cocaine_dependence", package = "easyml"); set.seed(43210)

# Analyze data
easy_glmnet(prostate, "lpsa")
