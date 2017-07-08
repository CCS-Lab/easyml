"""
Functions for random forest analysis.
"""
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

from .core import easy_analysis


__all__ = ['easy_random_forest']

# Settings
sns.set_style('whitegrid')


class easy_random_forest(easy_analysis):
    """
    Easily build and evaluate a random forest model.

    This function wraps the easyml core framework, allowing a user
    to easily run the easyml methodology for a
    random forest model.

    Please see the core class `easy_analysis` for more details on arguments.
    """
    def __init__(self, data, dependent_variable,
                 algorithm='random_forest', family='gaussian',
                 resample=None, preprocess=None, measure=None,
                 exclude_variables=None, categorical_variables=None,
                 train_size=0.667, survival_rate_cutoff=0.05,
                 n_samples=1000, n_divisions=1000, n_iterations=10,
                 random_state=None, progress_bar=True, n_core=1,
                 generate_coefficients=False,
                 generate_variable_importances=True,
                 generate_predictions=True, generate_model_performance=True,
                 model_args=None):
        super().__init__(data, dependent_variable,
                         algorithm=algorithm, family=family,
                         resample=resample, preprocess=preprocess, measure=measure,
                         exclude_variables=exclude_variables, categorical_variables=categorical_variables,
                         train_size=train_size, survival_rate_cutoff=survival_rate_cutoff,
                         n_samples=n_samples, n_divisions=n_divisions, n_iterations=n_iterations,
                         random_state=random_state, progress_bar=progress_bar, n_core=n_core,
                         generate_coefficients=generate_coefficients,
                         generate_variable_importances=generate_variable_importances,
                         generate_predictions=generate_predictions,
                         generate_model_performance=generate_model_performance,
                         model_args=model_args)

    def create_estimator(self):
        """
        Create an estimator.

        Creates an estimator depending on the family of regression.

        :return: A scikit-learn estimator.
        """
        if self.family == 'gaussian':
            estimator = RandomForestRegressor()
        elif self.family == 'binomial':
            estimator = RandomForestClassifier()
        return estimator

    def extract_variable_importances(self, estimator):
        """
        Extract variable importances from a random forest model.

        :param estimator: An estimator that has been fit to data.
        :return: An ndarray.
        """
        return estimator.feature_importances_

    def process_variable_importances(self, variable_importances):
        """
        Process variable importances for plotting.

        :return: An ndarray.
        """
        return variable_importances

    def predict_model(self, model, X):
        """
        Predict values from model.

        Generates predictions from a model depending on the family of regression.

        :param model: The model to use for generating predictions.
        :param X: The data to use for generating predictions.
        :return: An ndarray.
        """
        if self.family == 'gaussian':
            predictions = model.predict(X)
        elif self.family == 'binomial':
            predictions = model.predict_proba(X)
            predictions = predictions[:, 1]
        return predictions

    def plot_variable_importances(self):
        """
        Plots the variable importances.

        :return: Figure and axe.
        """
        n = self.variable_importances.shape[1]
        importances_mean = np.mean(self.variable_importances, axis=0)
        column_names = [v[1] for v in sorted(zip(importances_mean, self.column_names))]
        importances_std = np.std(self.variable_importances, axis=0)
        importances_std = [v[1] for v in sorted(zip(importances_mean, importances_std))]
        importances_mean = sorted(importances_mean)

        fig, ax = plt.subplots()
        ax.barh(range(n), importances_mean, color='grey', ecolor='black',
                xerr=importances_std, align='center')
        ax.set_xlabel('Variable Importance (Mean Decrease in Gini Index)')
        ax.set_yticks(range(n))
        ax.set_yticklabels(column_names)
        ax.set_ylabel('Predictors')
        ax.set_title('Variable Importances', loc='left')
        return fig, ax
