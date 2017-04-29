import matplotlib.pyplot as plt
import os
import pandas as pd

from easyml import plot
from easyml.glmnet import easy_glmnet
from easyml.random_forest import easy_random_forest
from easyml.support_vector_machine import easy_support_vector_machine


# Set matplotlib settings
plt.style.use('ggplot')

if __name__ == '__main__':
    # Load data
    directory = './Python/examples/'
    cocaine_dependence = pd.read_table(os.path.join(directory, 'cocaine_dependence.txt'))

    # Analyze data
    output = easy_glmnet(cocaine_dependence, 'DIAGNOSIS',
                         family='binomial',
                         exclude_variables=['subject'],
                         categorical_variables=['Male'],
                         random_state=12345, progress_bar=True, n_core=1,
                         n_samples=5, n_divisions=5, n_iterations=2,
                         model_args={'alpha': 1, 'n_lambda': 200})
    print(output.estimator)
    plot.plot_coefficients_processed(output.coefficients)
    print(output.column_names)



    # Analyze data
    output = easy_random_forest(cocaine_dependence, 'DIAGNOSIS',
                                family='binomial',
                                exclude_variables=['subject'],
                                categorical_variables=['Male'],
                                random_state=1, progress_bar=True, n_core=1,
                                n_samples=5, n_divisions=5, n_iterations=2,
                                model_args={'n_estimators': 10})

    output = easy_support_vector_machine(cocaine_dependence, 'DIAGNOSIS',
                                         family='binomial',
                                         exclude_variables=['subject'],
                                         categorical_variables=['Male'],
                                         random_state=1, progress_bar=True, n_core=1,
                                         n_samples=5, n_divisions=5, n_iterations=2)
    plot.plot_metrics_binomial_area_under_curve(output.metrics_train)
    plot.plot_metrics_binomial_area_under_curve(output.metrics_test)
    plot.plot_predictions_binomial(output.predictions_train_mean)
