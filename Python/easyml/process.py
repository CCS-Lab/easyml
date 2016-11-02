"""Utility functions for prcoessing data.
"""
import numpy as np
import pandas as pd


__all__ = ['sample_equal_proportion']


def process_coefficients(coefs):
    coefs = np.array(coefs)
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
