import matplotlib as mpl; mpl.use('TkAgg')
import matplotlib.pyplot as plt
import pandas as pd

from easyml.factory import easy_glmnet


# Set matplotlib settings
plt.style.use('ggplot')

import os
os.chdir('./Python/examples/cocaine_dependence')


if __name__ == "__main__":
    # Load data
    cocaine_depedence = pd.read_table('./cocaine_depedence.txt')

    # Analyze data
    easy_glmnet(cocaine_depedence, 'DIAGNOSIS',
                family='binomial', exclude_variables=['subject'], categorical_variables=['Male'],
                random_state=1, progress_bar=True, n_core=2,
                n_samples=10, n_divisions=10, n_iterations=2,
                alpha=1, n_lambda=200, standardize=False, cut_point=0, max_iter=1e6)
