from easyml import glmnet, random_forest, support_vector_machine
import pandas as pd


# Load data
prostate = pd.read_table('./Python/examples/prostate.txt')

# Analyze data
output = glmnet.easy_glmnet(prostate, 'lpsa',
                            random_state=1, progress_bar=True, n_core=1,
                            n_samples=10, n_divisions=10, n_iterations=2,
                            model_args={'alpha': 1})

# Analyze data
output = random_forest.easy_random_forest(prostate, 'lpsa',
                                          random_state=1, progress_bar=True, n_core=1,
                                          n_samples=5, n_divisions=5, n_iterations=2,
                                          model_args={'n_estimators': 10})

# Analyze data
output = support_vector_machine.easy_support_vector_machine(prostate, 'lpsa',
                                                            random_state=1, progress_bar=True, n_core=1,
                                                            n_samples=5, n_divisions=5, n_iterations=2)
