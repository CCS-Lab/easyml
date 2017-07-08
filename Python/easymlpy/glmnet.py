"""
Functions for glmnet analysis.
"""
from glmnet import ElasticNet, LogitNet
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .core import easy_analysis
from .preprocess import preprocess_scale


__all__ = ['easy_glmnet']

# Settings
sns.set_style('whitegrid')


class easy_glmnet(easy_analysis):
    """
    Easily build and evaluate a penalized regression model.

    This function wraps the easyml core framework, allowing a user
    to easily run the easyml methodology for a glmnet model.

    Please see the core class `easy_analysis` for more details on arguments.
    """
    def __init__(self, data, dependent_variable,
                 algorithm='glmnet', family='gaussian',
                 resample=None, preprocess=preprocess_scale, measure=None,
                 exclude_variables=None, categorical_variables=None,
                 train_size=0.667, survival_rate_cutoff=0.05,
                 n_samples=1000, n_divisions=1000, n_iterations=10,
                 random_state=None, progress_bar=True, n_core=1,
                 generate_coefficients=True,
                 generate_variable_importances=False,
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
            estimator = ElasticNet(standardize=False, cut_point=0)
        elif self.family == 'binomial':
            estimator = LogitNet(standardize=False, cut_point=0)
        return estimator

    def extract_coefficients(self, estimator):
        """
        Extract coefficients from a penalized regression model.

        :param estimator: An estimator that has been fit to data.
        :return: An ndarray.
        """
        if self.family == 'gaussian':
            coefficient = estimator.coef_
        elif self.family == 'binomial':
            coefficient = estimator.coef_[0]
        return coefficient

    def process_coefficients(self, coefficients, column_names, survival_rate_cutoff=0.05):
        """
        Process coefficients for plotting.

        :param coefficients: An ndarray.
        :param column_names: A list of strings, the columns of the data.
        :param survival_rate_cutoff: The cutoff for survival.
        :return: An object of class pandas.DataFrame.
        """
        n = coefficients.shape[0]
        survived = 1 * (abs(coefficients) > 0)
        survival_rate = np.sum(survived, axis=0) / float(n)
        mask = 1 * (survival_rate > survival_rate_cutoff)
        coefficients_updated = coefficients * mask
        betas = pd.DataFrame({'predictor': column_names})
        betas['mean'] = np.mean(coefficients_updated, axis=0)
        betas['lb'] = np.percentile(coefficients_updated, q=2.5, axis=0)
        betas['ub'] = np.percentile(coefficients_updated, q=97.5, axis=0)
        betas['survival'] = mask
        betas['sig'] = betas['survival']
        betas['dotColor1'] = 1 * (betas['mean'] != 0)
        betas['dotColor2'] = (1 * np.logical_and(betas['dotColor1'] > 0, betas['sig'] > 0)) + 1
        betas['dotColor'] = betas['dotColor1'] * betas['dotColor2']
        return betas

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

    def plot_coefficients(self):
        """
        Plots the coefficients.

        :return: Figure and axe.
        """
        n = self.coefficients.shape[1]
        coefficients_mean = np.mean(self.coefficients, axis=0)
        column_names = [v[1] for v in sorted(zip(coefficients_mean, self.column_names))]
        coefficients_std = np.std(self.coefficients, axis=0)
        coefficients_std = [v[1] for v in sorted(zip(coefficients_mean, coefficients_std))]
        coefficients_mean = sorted(coefficients_mean)

        fig, ax = plt.subplots()
        colors = ['grey' if i == 0 else 'black' for i in coefficients_mean]
        for i, coef_mean, coef_std, col in zip(range(n), coefficients_mean, coefficients_std, colors):
            ax.errorbar(coef_mean, i, xerr=coef_std, fmt='o', color=col)
        ax.set_xlabel('Coefficient estimates')
        ax.set_yticks(range(0, n))
        ax.set_ylim(-0.5, n - 0.5)
        ax.set_yticklabels(column_names)
        ax.set_ylabel('Predictors')
        ax.set_title('Estimates of coefficients', loc='left')
        return fig, ax
