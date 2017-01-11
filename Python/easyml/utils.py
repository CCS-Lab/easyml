"""Utility functions.
"""
import numpy as np
import os
import pandas as pd
from sklearn.model_selection import train_test_split

from . import preprocess
from . import resample


__all__ = []


def check_args():
    return 1


def identify_parallel(n_core):
    if n_core == 1:
        parallel = False
    elif n_core > 1:
        parallel = True
    else:
        raise ValueError
    return parallel


def reduce_cores(n_core, cpu_count=None):
    if cpu_count is None:
        cpu_count = os.cpu_count()
    n_core = min(n_core, cpu_count)
    return n_core


def set_column_names(column_names, dependent_variable,
                     exclude_variables=None, preprocessor=None, categorical_variables=None):
    column_names = [c for c in column_names if c != dependent_variable]
    if exclude_variables is not None:
        column_names = [c for c in column_names if c not in exclude_variables]
    if categorical_variables is not None and preprocessor is not None:
        column_names = [c for c in column_names if c not in categorical_variables]
        column_names = categorical_variables + column_names
    return column_names


def set_categorical_variables(column_names, categorical_variables=None):
    if categorical_variables is not None:
        categorical_variables = [True if c in categorical_variables else False for c in column_names]
    return categorical_variables


def set_preprocessor(preprocessor=None):
    if preprocessor is None:
        preprocessor = preprocess.preprocess_identity
    return preprocessor


def set_sampler(sampler=None, family=None):
    if sampler is None:
        if family == "gaussian":
            sampler = train_test_split
        elif family == "binomial":
            sampler = resample.sample_equal_proportion
        else:
            raise ValueError
    return sampler


def isolate_dependent_variable(data, dependent_variable):
    y = data[dependent_variable].values
    return y


def isolate_independent_variables(data, dependent_variable):
    X = data.drop(dependent_variable, axis=1, inplace=False).values
    return X


def remove_variables(data, exclude_variables=None):
    if exclude_variables is not None:
        data = data.drop(exclude_variables, axis=1, inplace=False)
    return data


def process_coefficients(coefs, column_names, survival_rate_cutoff=0.05):
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


def set_random_state(random_state=None):
    if random_state is not None:
        np.random.seed(random_state)
    return None
