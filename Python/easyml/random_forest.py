"""Functions for glmnet analysis.
"""
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import numpy as np
import pandas as pd

from .core import EasyAnalysis


__all__ = ['EasyRandomForest']


class EasyRandomForest(EasyAnalysis):
    def create_estimator(self):
        if self.family == 'gaussian':
            estimator = RandomForestRegressor()
        elif self.family == 'binomial':
            estimator = RandomForestClassifier()
        return estimator

    def predict_model(self, model, X):
        if self.family == 'gaussian':
            predictions = model.predict(X)
        elif self.family == 'binomial':
            predictions = model.predict_proba(X)
            predictions = predictions[:, 1]
        return predictions
