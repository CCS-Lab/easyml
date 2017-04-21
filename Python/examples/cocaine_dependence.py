import matplotlib.pyplot as plt
import os
import pandas as pd
from sklearn.metrics import roc_auc_score

from easyml.glmnet import EasyGlmnet
from easyml.random_forest import EasyRandomForest
from easyml.support_vector_machine import EasySupportVectorMachine


# Set matplotlib settings
plt.style.use('ggplot')

if __name__ == '__main__':
    # Load data
    directory = './Python/examples/'
    cocaine_dependence = pd.read_table(os.path.join(directory, 'cocaine_dependence.txt'))

    # Analyze data
    output = EasyGlmnet(cocaine_dependence, 'DIAGNOSIS',
                        family='binomial', measure=roc_auc_score,
                        exclude_variables=['subject'],
                        categorical_variables=['Male'],
                        random_state=1, progress_bar=True, n_core=1,
                        n_samples=10, n_divisions=1000, n_iterations=10)

    # Analyze data
    output = EasyRandomForest(cocaine_dependence, 'DIAGNOSIS',
                              family='binomial', measure=roc_auc_score,
                              exclude_variables=['subject'],
                              categorical_variables=['Male'],
                              random_state=1, progress_bar=True, n_core=1,
                              n_samples=5, n_divisions=5, n_iterations=2)

    output = EasySupportVectorMachine(cocaine_dependence, 'DIAGNOSIS',
                                      family='binomial', measure=roc_auc_score,
                                      exclude_variables=['subject'],
                                      categorical_variables=['Male'],
                                      random_state=1, progress_bar=True, n_core=1,
                                      n_samples=5, n_divisions=5, n_iterations=2)
