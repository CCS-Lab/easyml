import matplotlib as mpl; mpl.use('TkAgg')
import matplotlib.pyplot as plt
import os
import pandas as pd

from easyml.glmnet import easy_glmnet
from easyml.random_forest import easy_random_forest


# Set matplotlib settings
plt.style.use('ggplot')
os.chdir('./Python/examples/cocaine_dependence')

if __name__ == "__main__":
    # Load data
    cocaine_depedence = pd.read_table('./cocaine_depedence.txt')

    # Analyze data
    easy_glmnet(cocaine_depedence, 'DIAGNOSIS',
                family='binomial', exclude_variables=['subject'], categorical_variables=['Male'],
                random_state=1, progress_bar=True, n_core=os.cpu_count(),
                n_samples=100, n_divisions=10, n_iterations=5,
                alpha=1, n_lambda=200, standardize=False, cut_point=0, max_iter=1e6)

    # Analyze data
    easy_random_forest(cocaine_depedence, 'DIAGNOSIS',
                       family='binomial', exclude_variables=['subject'], categorical_variables=['Male'],
                       random_state=1, progress_bar=True, n_core=os.cpu_count(),
                       n_samples=100, n_divisions=10, n_iterations=5)
