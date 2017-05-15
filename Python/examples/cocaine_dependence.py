from easyml.glmnet import easy_glmnet
from easyml import random_forest, support_vector_machine


# Load data
import pandas as pd
cocaine_dependence = pd.read_table('./Python/examples/cocaine_dependence.txt')

# Analyze data
results = easy_glmnet(cocaine_dependence, 'DIAGNOSIS',
                      family='binomial',
                      exclude_variables=['subject'],
                      categorical_variables=['Male'],
                      random_state=12345, progress_bar=True, n_core=1,
                      n_samples=5, n_divisions=5, n_iterations=2,
                      model_args={'alpha': 1, 'n_lambda': 200})
print(results.plot_coefficients())

# Analyze data
results = random_forest.easy_random_forest(cocaine_dependence, 'DIAGNOSIS',
                                          family='binomial',
                                          exclude_variables=['subject'],
                                          categorical_variables=['Male'],
                                          random_state=1, progress_bar=True, n_core=1,
                                          n_samples=5, n_divisions=20, n_iterations=2,
                                          model_args={'n_estimators': 10})
print(results.plot_variable_importances())

results = support_vector_machine.easy_support_vector_machine(cocaine_dependence, 'DIAGNOSIS',
                                                            family='binomial',
                                                            exclude_variables=['subject'],
                                                            categorical_variables=['Male'],
                                                            random_state=1, progress_bar=True, n_core=1,
                                                            n_samples=5, n_divisions=5, n_iterations=2)

print(results.plot_predictions_single_train_test_split_train())
print(results.plot_predictions_single_train_test_split_test())
print(results.plot_roc_single_train_test_split_train())
print(results.plot_roc_single_train_test_split_test())
print(results.plot_model_performance_train())
print(results.plot_model_performance_test())
