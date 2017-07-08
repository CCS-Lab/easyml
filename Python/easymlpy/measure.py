"""
Functions for measuring model performance.
"""
import numpy as np
from sklearn.metrics import mean_squared_error, roc_auc_score


__all__ = ['measure_mean_squared_error', 'measure_cor_score',
           'measure_r2_score', 'measure_area_under_curve']


def measure_mean_squared_error(y_true, y_pred):
    """
    Measure mean squared error.

    Given the ground truth (correct) target values and the estimated target
    values, calculates the correlation metric.

    :param y_true: An ndarray; the ground truth (correct) target values.
    :param y_pred: An ndarray; the estimated target values.
    :return: A float.
    """
    return mean_squared_error(y_true, y_pred)


def measure_cor_score(y_true, y_pred):
    """
    Measure Pearsons Correlation Coefficient.

    Given the ground truth (correct) target values and the estimated target
    values, calculates the mean squared error metric.

    :param y_true: An ndarray; the ground truth (correct) target values.
    :param y_pred: An ndarray; the estimated target values.
    :return: A float.
    """
    return np.corrcoef(y_true, y_pred)[0, 1]


def measure_r2_score(y_true, y_pred):
    """
    Measure Coefficient of Determination (R^2 Score).

    Given the ground truth (correct) target values and the estimated target
    values, calculates the the R^2 metric.

    :param y_true: An ndarray; the ground truth (correct) target values.
    :param y_pred: An ndarray; the estimated target values.
    :return: A float.
    """
    return measure_cor_score(y_true, y_pred) ** 2


def measure_area_under_curve(y_true, y_pred):
    """
    Measure area under the curve.

    Given the ground truth (correct) target values and the estimated target
    values, calculates the the AUC metric.

    :param y_true: An ndarray; the ground truth (correct) target values.
    :param y_pred: An ndarray; the estimated target values.
    :return: A float.
    """
    return roc_auc_score(y_true, y_pred)
