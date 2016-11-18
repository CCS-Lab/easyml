import matplotlib as mpl
import pandas as pd

# Set matplotlib settings
mpl.get_backend()
mpl.use('TkAgg')
import matplotlib.pyplot as plt
plt.style.use('ggplot')

from easyml.factory import easy_glmnet


if __name__ == "__main__":
    # Load data
    cocaine_depedence = pd.read_table('./cocaine_depedence.txt')

    # Analyze data
    easy_glmnet(cocaine_depedence, 'DIAGNOSIS',
                family='binomial', exclude_variables=['subject'], categorical_variables=['Male'],
                alpha=1, n_lambda=200, standardize=False, cut_point=0, max_iter=1e6)
