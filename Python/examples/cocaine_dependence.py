import matplotlib.pyplot as plt
import os
import pandas as pd

from easyml.glmnet import EasyGlmnet


# Set matplotlib settings
plt.style.use('ggplot')

if __name__ == '__main__':
    # Load data
    directory = './Python/examples/'
    cocaine_dependence = pd.read_table(os.path.join(directory, 'cocaine_dependence.txt'))

    # Analyze data
    output = EasyGlmnet(cocaine_dependence, 'DIAGNOSIS',
                        family='binomial', exclude_variables=['subject'],
                        categorical_variables=['Male'],
                        random_state=1, progress_bar=True, n_core=1,
                        n_samples=5, n_divisions=5, n_iterations=2)
