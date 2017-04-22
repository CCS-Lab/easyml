"""
Functions for random forest analysis.
"""
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

from .core import EasyAnalysis


__all__ = ['EasyRandomForest']


class EasyRandomForest(EasyAnalysis):
    def __init__(self, data, dependent_variable,
                 algorithm='random_forest', family='gaussian',
                 resample=None, preprocess=None, measure=None,
                 exclude_variables=None, categorical_variables=None,
                 train_size=0.667, survival_rate_cutoff=0.05,
                 n_samples=1000, n_divisions=1000, n_iterations=10,
                 random_state=None, progress_bar=True, n_core=1,
                 generate_coefficients=False,
                 generate_variable_importances=True,
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
            estimator = RandomForestRegressor()
        elif self.family == 'binomial':
            estimator = RandomForestClassifier()
        return estimator

    def extract_variable_importances(self, estimator):
        return estimator.feature_importances_

    def process_variable_importances(self, variable_importances):
        return variable_importances

    def predict_model(self, model, X):
        if self.family == 'gaussian':
            predictions = model.predict(X)
        elif self.family == 'binomial':
            predictions = model.predict_proba(X)
            predictions = predictions[:, 1]
        return predictions
