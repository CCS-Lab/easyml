"""
Functions for measuring model performance.
"""
import numpy as np
from sklearn.metrics import mean_squared_error, roc_auc_score


__all__ = []


def measure_mean_squared_error(y_true, y_pred):
    return mean_squared_error(y_true, y_pred)


def measure_cor_score(y_true, y_pred):
    return np.corrcoef(y_true, y_pred)[0, 1]


def measure_r2_score(y_true, y_pred):
    return measure_cor_score(y_true, y_pred) ** 2


def measure_area_under_curve(y_true, y_pred):
    return roc_auc_score(y_true, y_pred)
