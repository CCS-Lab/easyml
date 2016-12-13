"""Utility functions for bootstrapping estimators.
"""
import numpy as np
from sklearn.preprocessing import StandardScaler


__all__ = []


def preprocess_identity(*data, categorical_variables=None):
    return data


def preprocess_scaler(*data, categorical_variables=None):
    sclr = StandardScaler()
    if len(data) == 1:
        X = data
        if categorical_variables is None:
            sclr.fit_transform(X)
            output = X
        else:
            X_categorical = X[:, categorical_variables]
            X_numerical = X[:, np.logical_not(categorical_variables)]
            sclr.fit_transform(X_numerical)
            output = np.concatenate([X_categorical, X_numerical], axis=1)
    elif len(data) == 2:
        X_train, X_test = data
        if categorical_variables is None:
            sclr.fit_transform(X_train)
            sclr.transform(X_test)
            output = X_train, X_test
        else:
            X_train_categorical = X_train[:, categorical_variables]
            X_train_numerical = X_train[:, np.logical_not(categorical_variables)]
            X_test_categorical = X_test[:, categorical_variables]
            X_test_numerical = X_test[:, np.logical_not(categorical_variables)]
            sclr.fit_transform(X_train_numerical)
            sclr.transform(X_test_numerical)
            X_train_output = np.concatenate([X_train_categorical, X_train_numerical], axis=1)
            X_test_output = np.concatenate([X_test_categorical, X_test_numerical], axis=1)
            output = X_train_output, X_test_output
    else:
        raise ValueError
    return output
