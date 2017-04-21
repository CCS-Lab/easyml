"""
Functions for resampling data.
"""
import numpy as np
from sklearn.model_selection import train_test_split


__all__ = []



def resample_simple_train_test_split(X, y, train_size=0.667):
    return train_test_split(X, y, train_size=train_size)


def resample_stratified_simple_train_test_split(X, y, train_size=0.667):
    return 1


def resample_stratified_class_train_test_split(X, y, train_size=0.667):
    """Sample in equal proportion.

    :param y: array, shape (n_obs) Input data to be split.
    :param train_size: float, default: 0.667 Proportion to split into train and test.

    :return: array, shape (n_obs) A boolean array of length n_obs where True represents .
    that observation should be in the train set.
    """

    # calculate number of observations
    n_obs = len(y)

    # identify index number for class1 and class2
    index_class1 = np.where(y == 0)[0]
    index_class2 = np.where(y == 1)[0]

    # calculate  number of class1 and class2 observations
    n_class1 = len(index_class1)
    n_class2 = len(index_class2)

    # calculate number of class1 and class2 observations in the train set
    n_class1_train = int(np.round(n_class1 * train_size))
    n_class2_train = int(np.round(n_class2 * train_size))

    # generate indices for class1 and class2 observations in the train set
    index_class1_train = np.random.choice(index_class1, size=n_class1_train, replace=False)
    index_class2_train = np.random.choice(index_class2, size=n_class2_train, replace=False)
    index_train = np.append(index_class1_train, index_class2_train)

    # return a boolean vector of len n_obs where TRUE represents
    # that observation should be in the train set
    mask = np.in1d(np.arange(n_obs), index_train)

    # Create train and test splits
    X_train = X[mask, :]
    X_test = X[np.logical_not(mask), :]
    y_train = y[mask]
    y_test = y[np.logical_not(mask)]

    return X_train, X_test, y_train, y_test


def resample_fold_train_test_split(X, y, train_size=0.667):
    return 1
