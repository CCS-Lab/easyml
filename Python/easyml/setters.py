"""TO BE EDITED.
"""
import numpy as np
from sklearn.model_selection import train_test_split

from . import preprocess as prep
from . import resample as res


__all__ = []


def set_coefficients_boolean(algorithm):
    algorithms = ['glmnet']
    boolean = algorithm in algorithms
    return boolean


def set_predictions_boolean(algorithm):
    algorithms = ['glmnet', 'random_forest', 'support_vector_machine']
    boolean = algorithm in algorithms
    return boolean


def set_metrics_boolean(algorithm):
    algorithms = ['glmnet', 'random_forest', 'support_vector_machine']
    boolean = algorithm in algorithms
    return boolean


def set_parallel(n_core):
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
    if exclude_variables is not None:
        column_names = [c for c in column_names if c not in exclude_variables]
    if categorical_variables is not None and preprocess is prep.preprocess_scaler:
        column_names = [c for c in column_names if c not in categorical_variables]
        column_names = categorical_variables + column_names
    return column_names


def set_categorical_variables(column_names, categorical_variables=None):
    if categorical_variables is not None:
        categorical_variables = [True if c in categorical_variables else False for c in column_names]
    return categorical_variables


def set_random_state(random_state=None):
    if random_state is not None:
        np.random.seed(random_state)
    return None


def set_resample(resample=None, family=None):
    if resample is None:
        if family == "gaussian":
            resample = train_test_split
        elif family == "binomial":
            resample = res.resample_equal_proportion
        else:
            raise ValueError
    return resample


def set_preprocess(preprocess=None):
    if preprocess is None:
        preprocess = prep.preprocess_identity
    return preprocess


def set_measure(measure=None, algorithm=None, family=None):
    return measure


def set_dependent_variable(data, dependent_variable):
    y = data[dependent_variable].values
    return y


def set_independent_variables(data, dependent_variable):
    X = data.drop(dependent_variable, axis=1, inplace=False).values
    return X


def set_fit_model(algorithm=None, family=None):
    return None


def set_extract_coefficients(algorithm=None, family=None):
    return None


def set_predict_model(algorithm=None, family=None):
    return None


def set_plot_predictions(algorithm=None, family=None):
    return None


def set_plot_metrics(algorithm=None, family=None):
    return None
