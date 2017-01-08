library(easyml) # https://github.com/CCS-Lab/easyml

# Load data
data("prostate", package = "easyml")

# Analyze data
results <- easy_svm(prostate, "lpsa", 
                    preprocessor = preprocess_scaler, 
                    n_samples = 1000, n_divisions = 1000, n_iterations = 100, 
                    random_state = 1, n_core = 8)
