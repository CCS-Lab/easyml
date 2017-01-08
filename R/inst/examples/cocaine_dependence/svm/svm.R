library(easyml) # https://github.com/CCS-Lab/easyml

# Load data
data("cocaine_dependence", package = "easyml")

# Analyze data
results <- easy_svm(cocaine_dependence, "diagnosis", 
                    family = "binomial", preprocessor = preprocess_scaler, 
                    exclude_variables = c("subject"), 
                    categorical_variables = c("male"), 
                    n_samples = 1000, n_divisions = 1000, n_iterations = 100, 
                    random_state = 1, n_core = 8)
