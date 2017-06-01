from easyml import glmnet, random_forest, support_vector_machine
from easyml.datasets import load_cocaine_dependence


# Settings
n_samples = 50
n_divisions = 50
n_iterations = 2

# Load data
cocaine_dependence = load_cocaine_dependence()

# Analyze data
results = random_forest.easy_random_forest(cocaine_dependence, 'diagnosis',
                                          family='binomial',
                                          exclude_variables=['subject'],
                                          categorical_variables=['male'],
                                          random_state=1, progress_bar=True, n_core=1,
                                          n_samples=n_samples, n_divisions=n_divisions,
                                          n_iterations=n_iterations,
                                          model_args={'n_estimators': 10})
print(results.plot_variable_importances())
print(results.plot_predictions_single_train_test_split_train())
print(results.plot_predictions_single_train_test_split_test())
print(results.plot_roc_single_train_test_split_train())
print(results.plot_roc_single_train_test_split_test())
print(results.plot_model_performance_train())
print(results.plot_model_performance_test())


# Analyze data
results = glmnet.easy_glmnet(cocaine_dependence, 'diagnosis',
                             family='binomial',
                             exclude_variables=['subject'],
                             categorical_variables=['male'],
                             random_state=12345, progress_bar=True, n_core=1,
                             n_samples=n_samples, n_divisions=n_divisions, n_iterations=n_iterations,
                             model_args={'alpha': 1, 'n_lambda': 200})

print(results.plot_coefficients())
print(results.plot_predictions_single_train_test_split_train())
print(results.plot_predictions_single_train_test_split_test())
print(results.plot_roc_single_train_test_split_train())
print(results.plot_roc_single_train_test_split_test())
print(results.plot_model_performance_train())
print(results.plot_model_performance_test())

# Analyze data
results = support_vector_machine.easy_support_vector_machine(cocaine_dependence, 'diagnosis',
                                                            family='binomial',
                                                            exclude_variables=['subject'],
                                                            categorical_variables=['male'],
                                                            random_state=1, progress_bar=True, n_core=1,
                                                            n_samples=n_samples, n_divisions=n_divisions,
                                                             n_iterations=n_iterations)

print(results.plot_predictions_single_train_test_split_train())
print(results.plot_predictions_single_train_test_split_test())
print(results.plot_roc_single_train_test_split_train())
print(results.plot_roc_single_train_test_split_test())
print(results.plot_model_performance_train())
print(results.plot_model_performance_test())
