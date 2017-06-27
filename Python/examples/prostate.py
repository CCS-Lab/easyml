from easymlpy import glmnet, random_forest, support_vector_machine
from easymlpy.datasets import load_prostate


# Settings
n_samples = 50
n_divisions = 50
n_iterations = 2

# Load data
prostate = load_prostate()

# Analyze data
output = support_vector_machine.easy_support_vector_machine(prostate, 'lpsa',
                                                            random_state=1, progress_bar=True, n_core=1,
                                                            n_samples=50, n_divisions=50, n_iterations=2)

output.plot_predictions_single_train_test_split_train()
output.plot_predictions_single_train_test_split_test()
output.plot_model_performance_train()
output.plot_model_performance_test()

# Analyze data
output = random_forest.easy_random_forest(prostate, 'lpsa',
                                          random_state=1, progress_bar=True, n_core=1,
                                          n_samples=50, n_divisions=50, n_iterations=2,
                                          model_args={'n_estimators': 10})
output.plot_variable_importances()
output.plot_predictions_single_train_test_split_train()
output.plot_predictions_single_train_test_split_test()
output.plot_model_performance_train()
output.plot_model_performance_test()

# Analyze data
output = glmnet.easy_glmnet(prostate, 'lpsa',
                            random_state=1, progress_bar=True, n_core=1,
                            n_samples=50, n_divisions=50, n_iterations=2,
                            model_args={'alpha': 1, 'n_lambda': 200})
output.plot_coefficients()
output.plot_predictions_single_train_test_split_train()
output.plot_predictions_single_train_test_split_test()
output.plot_model_performance_train()
output.plot_model_performance_test()
