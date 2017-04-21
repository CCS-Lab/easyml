"""
Functions for setting certain functions and parameters.
"""
import numpy as np
from sklearn.model_selection import train_test_split

from . import measure as meas
from . import preprocess as prep
from . import resample as res


__all__ = []


def set_parallel(n_core):
    n_core = int(n_core)
    if n_core == 1:
        parallel = False
    elif n_core > 1:
        parallel = True
    else:
        raise ValueError
    return parallel


def set_column_names(column_names, dependent_variable,
                     exclude_variables=None, preprocess=None, categorical_variables=None):
    column_names = [c for c in column_names if c != dependent_variable]
    if exclude_variables:
        column_names = [c for c in column_names if c not in exclude_variables]
    if categorical_variables and preprocess is prep.preprocess_scale:
        column_names = [c for c in column_names if c not in categorical_variables]
        column_names = categorical_variables + column_names
    return column_names


def set_categorical_variables(column_names, categorical_variables=None):
    if categorical_variables:
        categorical_variables = np.in1d(column_names, categorical_variables)
    return categorical_variables


def set_random_state(random_state=None):
    if random_state:
        np.random.seed(random_state)
    return None


def set_resample(resample=None, family=None):
    if not resample:
        if family == 'gaussian':
            resample = train_test_split
        elif family == 'binomial':
            resample = res.resample_equal_proportion
        else:
            raise ValueError
    return resample


def set_preprocess(preprocess=None):
    if not preprocess:
        preprocess = prep.preprocess_identity
    return preprocess


def set_measure(measure=None, family=None):
    if not measure:
        if family == 'gaussian':
            measure = 1
        elif family == 'binomial':
            measure = 1
        else:
            raise ValueError
    return measure


def set_dependent_variable(data, dependent_variable):
    y = data[dependent_variable].values
    return y


def set_independent_variables(data, dependent_variable):
    X = data.drop(dependent_variable, axis=1).values
    return X


def set_plot_metrics(measure):
    if measure is 1:
        plot_metrics = 1
    elif measure is 2:
        plot_metrics = 2
    else:
        raise ValueError
    return plot_metrics
