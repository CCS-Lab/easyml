"""Functions for glmnet analysis.
"""
from glmnet import ElasticNet, LogitNet
import numpy as np

from .core import EasyAnalysis


__all__ = ['EasyGlmnet']


class EasyGlmnet(EasyAnalysis):
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

    def predict_model(self):
        predictions = self.estimator.predict_proba(self.X)
        if self.family == 'binomial':
            predictions = predictions[:, 1]
        return predictions
