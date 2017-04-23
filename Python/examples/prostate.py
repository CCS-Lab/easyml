import matplotlib.pyplot as plt
import os
import pandas as pd

from easyml.glmnet import EasyGlmnet
from easyml.random_forest import EasyRandomForest
from easyml.support_vector_machine import EasySupportVectorMachine


# Set matplotlib settings
plt.style.use('ggplot')

if __name__ == '__main__':
    # Load data
    directory = './Python/examples/'
    prostate = pd.read_table(os.path.join(directory, 'prostate.txt'))

    # Analyze data
    output = EasyGlmnet(prostate, 'lpsa',
                        random_state=1, progress_bar=True, n_core=1,
                        n_samples=5, n_divisions=5, n_iterations=2,
                        model_args={'alpha': 1})

    # Analyze data
    output = EasyRandomForest(prostate, 'lpsa',
                              random_state=1, progress_bar=True, n_core=1,
                              n_samples=5, n_divisions=5, n_iterations=2,
                              model_args={'n_estimators': 10})

    # Analyze data
    output = EasySupportVectorMachine(prostate, 'lpsa',
                                      random_state=1, progress_bar=True, n_core=1,
                                      n_samples=5, n_divisions=5, n_iterations=2)
