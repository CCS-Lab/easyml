"""
Functions for preprocessing.
"""
import numpy as np
from sklearn.preprocessing import StandardScaler


__all__ = ['preprocess_identity', 'preprocess_scale']


def preprocess_identity(*data, categorical_variables=None):
    """    
    An identify function for preprocessing.

    Returns inputs without modifying them.

    :param data: array(s).
    :param categorical_variables: A list of strings representing the variables that are categorical.
    :return: array(s).
    """
    if len(data) == 1:
        output = data[0]
    elif len(data) == 2:
        output = data[0], data[1]
    else:
        raise ValueError
    return output


def preprocess_scale(*data, categorical_variables=None):
    """    
    A function for scaling data.

    Takes one or two arrays and scales them using a standard scaler.

    :param data: array(s).
    :param categorical_variables: A list of strings representing the variables that are categorical.
    :return: array(s).
    """
    sclr = StandardScaler()
    if len(data) == 1:
        X = data[0]
        if categorical_variables is not None:
            # import pdb; pdb.set_trace()
            X_categorical = X[:, categorical_variables]
            X_numerical = X[:, np.logical_not(categorical_variables)]
            X_standardized = sclr.fit_transform(X_numerical)
            output = np.concatenate([X_categorical, X_standardized], axis=1)
        else:
            output = sclr.fit_transform(X)
    elif len(data) == 2:
        X_train, X_test = data[0], data[1]
        if categorical_variables is not None:
            X_train_categorical = X_train[:, categorical_variables]
            X_train_numerical = X_train[:, np.logical_not(categorical_variables)]
            X_test_categorical = X_test[:, categorical_variables]
            X_test_numerical = X_test[:, np.logical_not(categorical_variables)]
            X_train_standardized = sclr.fit_transform(X_train_numerical)
            X_test_standardized = sclr.transform(X_test_numerical)
            X_train_output = np.concatenate([X_train_categorical, X_train_standardized], axis=1)
            X_test_output = np.concatenate([X_test_categorical, X_test_standardized], axis=1)
            output = X_train_output, X_test_output
        else:
            X_train_standardized = sclr.fit_transform(X_train)
            X_test_standardized = sclr.transform(X_test)
            output = X_train_standardized, X_test_standardized
    else:
        raise ValueError
    return output
