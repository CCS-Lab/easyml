from easyml import glmnet, random_forest, support_vector_machine
import pandas as pd


# Load data
cocaine_dependence = pd.read_table('./Python/examples/cocaine_dependence.txt')

# Analyze data
output = glmnet.easy_glmnet(cocaine_dependence, 'DIAGNOSIS',
                            family='binomial',
                            exclude_variables=['subject'],
                            categorical_variables=['Male'],
                            random_state=12345, progress_bar=True, n_core=1,
                            n_samples=5, n_divisions=5, n_iterations=2,
                            model_args={'alpha': 1, 'n_lambda': 200})

# Analyze data
output = random_forest.easy_random_forest(cocaine_dependence, 'DIAGNOSIS',
                                          family='binomial',
                                          exclude_variables=['subject'],
                                          categorical_variables=['Male'],
                                          random_state=1, progress_bar=True, n_core=1,
                                          n_samples=5, n_divisions=5, n_iterations=2,
                                          model_args={'n_estimators': 10})

output = support_vector_machine.easy_support_vector_machine(cocaine_dependence, 'DIAGNOSIS',
                                                            family='binomial',
                                                            exclude_variables=['subject'],
                                                            categorical_variables=['Male'],
                                                            random_state=1, progress_bar=True, n_core=1,
                                                            n_samples=5, n_divisions=5, n_iterations=2)
