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
    prostate = pd.read_table('./prostate.txt')

    # Analyze data
    easy_glmnet(prostate, 'lpsa',
                alpha=1, n_lambda=200, standardize=False, cut_point=0, max_iter=1e6)
