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
    data = pd.read_table('./prostate.txt')

    # Analyze data
    easy_glmnet(data, dependent_variable='lpsa')
