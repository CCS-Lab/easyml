"""Utility functions for prcoessing data.
"""
import numpy as np
import pandas as pd


__all__ = ['correlation_test', 'process_coefficients', 'process_data']


def correlation_test():
    # correlation_test < - function(x, confidence_level=0.95, ...)
    # {
    # # Initialize matrices
    # x < - as.matrix(x)
    # n < - ncol(x)
    # p_value < - lower_bound < - upper_bound < - matrix(NA, n, n)
    # diag(p_value) < - 0
    # diag(lower_bound) < - diag(upper_bound) < - 1
    #
    # # Loop through and test for correlation at some confidence_level
    # for (i in 1: (n - 1)) {
    # for (j in (i + 1): n) {
    #     result < - stats::cor.test(x[, i], x[, j], conf.level = confidence_level, ...)
    # p_value[i, j] < - p_value[j, i] < - result$p.value
    # lower_bound[i, j] < - lower_bound[j, i] < - result$conf.int[1]
    # upper_bound[i, j] < - upper_bound[j, i] < - result$conf.int[2]
    # }
    # }
    #
    # # Return a list containing three matrices; p_value, lower_bound, and upper bound.
    # list(p_value=p_value,
    #      lower_bound=lower_bound,
    #      upper_bound=upper_bound)
    # }
    return 1


def process_coefficients(coefs):
    survived = 1 * (abs(coefs) > 0)
    survival_rate = np.sum(survived, axis=0) / float(1000)
    mask = 1 * (survival_rate > 0.05)
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


def process_data(data, dependent_variables=None, exclude_variables=None):
    # Handle dependent variable
    if dependent_variables is not None:
        raise ValueError
    else:
        y = data[dependent_variables].values
        data = data.drop(dependent_variables, axis=1)

    # Possibly exclude columns
    if exclude_variables is not None:
        data = data.drop(exclude_variables, axis=1)

    # Create X array
    X = data.values

    return X, y
