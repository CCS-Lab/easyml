"""
Functions for glmnet analysis.
"""
from glmnet import ElasticNet, LogitNet
import numpy as np
import pandas as pd

from .core import EasyAnalysis
from .preprocess import preprocess_scale


__all__ = ['EasyGlmnet']


class EasyGlmnet(EasyAnalysis):
    def __init__(self, data, dependent_variable,
                 algorithm='glmnet', family='gaussian',
                 resample=None, preprocess=preprocess_scale, measure=None,
                 exclude_variables=None, categorical_variables=None,
                 train_size=0.667, survival_rate_cutoff=0.05,
                 n_samples=1000, n_divisions=1000, n_iterations=10,
                 random_state=None, progress_bar=True, n_core=1,
                 generate_coefficients=True,
                 generate_variable_importances=False,
                 generate_predictions=True, generate_metrics=True,
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
                         generate_predictions=generate_predictions, generate_metrics=generate_metrics,
                         model_args=model_args)

    def create_estimator(self):
        if self.family == 'gaussian':
            estimator = ElasticNet(alpha=1)
        elif self.family == 'binomial':
            estimator = LogitNet(alpha=1)
        return estimator

    def extract_coefficients(self, estimator):
        if self.family == 'gaussian':
            coefficient = estimator.coef_
        elif self.family == 'binomial':
            coefficient = estimator.coef_[0]
        return coefficient

    def process_coefficients(self, coefs, column_names, survival_rate_cutoff=0.05):
        n = coefs.shape[0]
        survived = 1 * (abs(coefs) > 0)
        survival_rate = np.sum(survived, axis=0) / float(n)
        mask = 1 * (survival_rate > survival_rate_cutoff)
        coefs_updated = coefs * mask
        betas = pd.DataFrame({'predictor': column_names})
        betas['mean'] = np.mean(coefs_updated, axis=0)
        betas['lb'] = np.percentile(coefs_updated, q=2.5, axis=0)
        betas['ub'] = np.percentile(coefs_updated, q=97.5, axis=0)
        betas['survival'] = mask
        betas['sig'] = betas['survival']
        betas['dotColor1'] = 1 * (betas['mean'] != 0)
        betas['dotColor2'] = (1 * np.logical_and(betas['dotColor1'] > 0, betas['sig'] > 0)) + 1
        betas['dotColor'] = betas['dotColor1'] * betas['dotColor2']
        return betas

    def predict_model(self, model, X):
        if self.family == 'gaussian':
            predictions = model.predict(X)
        elif self.family == 'binomial':
            predictions = model.predict_proba(X)
            predictions = predictions[:, 1]
        return predictions
