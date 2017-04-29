import matplotlib.pyplot as plt
import os
import pandas as pd

from easyml.glmnet import easy_glmnet
from easyml.random_forest import easy_random_forest
from easyml.support_vector_machine import easy_support_vector_machine


# Set matplotlib settings
plt.style.use('ggplot')

if __name__ == '__main__':
    # Load data
    directory = './Python/examples/'
    prostate = pd.read_table(os.path.join(directory, 'prostate.txt'))

    # Analyze data
    output = easy_glmnet(prostate, 'lpsa',
                         random_state=1, progress_bar=True, n_core=1,
                         n_samples=10, n_divisions=10, n_iterations=2,
                         model_args={'alpha': 1})

    # Analyze data
    output = easy_random_forest(prostate, 'lpsa',
                                random_state=1, progress_bar=True, n_core=1,
                                n_samples=5, n_divisions=5, n_iterations=2,
                                model_args={'n_estimators': 10})

    # Analyze data
    output = easy_support_vector_machine(prostate, 'lpsa',
                                         random_state=1, progress_bar=True, n_core=1,
                                         n_samples=5, n_divisions=5, n_iterations=2)
