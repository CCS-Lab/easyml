import matplotlib.pyplot as plt
import os
import pandas as pd
from sklearn.metrics import mean_squared_error

from easyml.glmnet import EasyGlmnet


# Set matplotlib settings
plt.style.use('ggplot')

if __name__ == '__main__':
    # Load data
    directory = './Python/examples/'
    prostate = pd.read_table(os.path.join(directory, 'prostate.txt'))

    # Analyze data
    output = EasyGlmnet(prostate, 'lpsa',
                        measure=mean_squared_error,
                        random_state=1, progress_bar=True, n_core=1,
                        n_samples=5, n_divisions=5, n_iterations=2)
