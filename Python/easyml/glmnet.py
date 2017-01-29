"""Functions for glmnet analysis.
"""
from glmnet import ElasticNet, LogitNet

from . import core


__all__ = ['easy_glmnet']


def glmnet_fit_model(X, y, **kwargs):
    model = ElasticNet(**kwargs)
    return model.fit(X, y)


def glmnet_extract_coefficients_gaussian(e):
    return e.coef_


def glmnet_extract_coefficients_binomial(e):
    return e.coef_[0]


def glmnet_predict_model_gaussian(e, X):
    return e.predict(X)


def glmnet_predict_model_binomial(e, X):
    return e.predict_proba(X)[:, 1]


def easy_glmnet(data, dependent_variable, family='gaussian',
                resample=None, preprocess=None, measure=None,
                exclude_variables=None, categorical_variables=None,
                train_size=0.667, survival_rate_cutoff=0.05,
                n_samples=1000, n_divisions=1000, n_iterations=10,
                random_state=None, progress_bar=True, n_core=1, **kwargs):
    output = core.easy_analysis(data=data, dependent_variable=dependent_variable,
                                algorithm='glmnet', family=family,
                                resample=resample, preprocess=preprocess, measure=measure,
                                exclude_variables=exclude_variables, categorical_variables=categorical_variables,
                                train_size=train_size, survival_rate_cutoff=survival_rate_cutoff,
                                n_samples=n_samples, n_divisions=n_divisions, n_iterations=n_iterations,
                                random_state=random_state, progress_bar=progress_bar, n_core=n_core, **kwargs)
    return output
