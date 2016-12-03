import matplotlib as mpl; mpl.use('TkAgg')
import matplotlib.pyplot as plt
import os
import pandas as pd

from easyml.random_forest import easy_random_forest


# Set matplotlib settings
plt.style.use('ggplot')
os.chdir('./Python/examples/prostate/random_forest')


if __name__ == "__main__":
    # Load data
    prostate = pd.read_table('./prostate.txt')

    # Analyze data
    easy_random_forest(prostate, 'lpsa',
                       random_state=1, progress_bar=True, n_core=1,
                       n_samples=100, n_divisions=10, n_iterations=5)

    # Analyze data
    easy_random_forest(prostate, 'lpsa',
                       random_state=1, progress_bar=True, n_core=os.cpu_count(),
                       n_samples=100, n_divisions=10, n_iterations=5)
