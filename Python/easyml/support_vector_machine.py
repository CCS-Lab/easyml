"""
Functions for support vector machine analysis.
"""
from sklearn.svm import SVR, SVC

from .core import easy_analysis
from .preprocess import preprocess_scale


__all__ = ['easy_support_vector_machine']


class easy_support_vector_machine(easy_analysis):
    def __init__(self, data, dependent_variable,
                 algorithm='support_vector_machine', family='gaussian',
                 resample=None, preprocess=preprocess_scale, measure=None,
                 exclude_variables=None, categorical_variables=None,
                 train_size=0.667, survival_rate_cutoff=0.05,
                 n_samples=1000, n_divisions=1000, n_iterations=10,
                 random_state=None, progress_bar=True, n_core=1,
                 generate_coefficients=False,
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
