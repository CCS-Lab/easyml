"""
Functions for support vector machine analysis.
"""
from sklearn.svm import SVR, SVC

from .core import EasyAnalysis


__all__ = ['EasySupportVectorMachine']


class EasySupportVectorMachine(EasyAnalysis):
    def create_estimator(self):
        if self.family == 'gaussian':
            estimator = SVR()
        elif self.family == 'binomial':
            estimator = SVC(probability=True)
        return estimator

    def predict_model(self, model, X):
        if self.family == 'gaussian':
            predictions = model.predict(X)
        elif self.family == 'binomial':
            predictions = model.predict_proba(X)
            predictions = predictions[:, 1]
        return predictions
