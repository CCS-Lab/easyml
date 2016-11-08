"""Utility functions for prcoessing data.
"""
import numpy as np
import pandas as pd


__all__ = ['process_coefficients', 'process_data']


def process_coefficients(coefs, survival_rate_cutoff=0.05):
    survived = 1 * (abs(coefs) > 0)
    survival_rate = np.sum(survived, axis=0) / float(1000)
    mask = 1 * (survival_rate > survival_rate_cutoff)
    coefs_updated = coefs * mask
    coefs_q025 = np.percentile(coefs_updated, q=2.5, axis=0)
    coefs_mean = np.mean(coefs_updated, axis=0)
    coefs_q975 = np.percentile(coefs_updated, q=97.5, axis=0)
    betas = pd.DataFrame({'mean': coefs_mean})
    betas['lb'] = coefs_q025
    betas['ub'] = coefs_q975
    betas['survival'] = mask
    betas['sig'] = betas['survival']
    betas['dotColor1'] = 1 * (betas['mean'] != 0)
    betas['dotColor2'] = (1 * np.logical_and(betas['dotColor1'] > 0, betas['sig'] > 0)) + 1
    betas['dotColor'] = betas['dotColor1'] * betas['dotColor2']
    return betas


def process_data(data, dependent_variable=None, exclude_variables=None):
    # Handle dependent variable
    if dependent_variable is not None:
        y = data[dependent_variable].values
        data = data.drop(dependent_variable, axis=1)
    else:
        raise ValueError

    # Possibly exclude columns
    if exclude_variables is not None:
        data = data.drop(exclude_variables, axis=1)

    # Create X array
    X = data.values

    return X, y
