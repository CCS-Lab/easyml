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
    data = pd.read_table('./cocaine.txt')

    # Analyze data
    easy_glmnet(data, dependent_variable='DIAGNOSIS', family='binomial', exclude_variables=['subject', 'AGE'],
                n_divisions=10, n_iterations=2, n_samples=10)
