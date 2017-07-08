"""
Functions for resampling data.
"""
import numpy as np
from sklearn.model_selection import train_test_split

from . import setters


__all__ = ['resample_fold_train_test_split',
           'resample_simple_train_test_split',
           'resample_stratified_class_train_test_split',
           'resample_stratified_simple_train_test_split']


def resample_simple_train_test_split(X, y, train_size=0.667, foldid=None, random_state=None):
    """
    Train test split.

    This will split the data into train and test.

    :param X: An ndarray, the data to be resampled.
    :param y: An ndarray with two classes, 0 and 1.
    :param train_size: A float; specifies what proportion of the data should be used for the training data set. Defaults to 0.667.
    :param foldid: Not currently supported in this function.
    :param random_state: An integer; specifies the seed to be used for the analysis. Defaults to None.
    :return: A tuple of arrays; the arrays X, y split into X_train, X_test, y_train, y_test.
    """
    return train_test_split(X, y, train_size=train_size, random_state=random_state)


def resample_stratified_simple_train_test_split(X, y, train_size=0.667, foldid=None, random_state=None):
    """
    Sample in equal proportion.
    
    This will sample in equal proportion.
    
    :param X: An ndarray, the data to be resampled.
    :param y: An ndarray with two classes, 0 and 1.
    :param train_size: A float; specifies what proportion of the data should be used for the training data set. Defaults to 0.667.
    :param foldid: A vector with length equal to len(y) which identifies cases belonging to the same fold.
    :param random_state: An integer; specifies the seed to be used for the analysis. Defaults to None.
    :return: A tuple of arrays; the arrays X, y split into X_train, X_test, y_train, y_test.
    """
    unique_foldids = np.unique(foldid)
    for i, unique_foldid in enumerate(unique_foldids):
        mask = foldid == unique_foldid
        X_subset = X[mask]
        y_subset = y[mask]
        arrays = train_test_split(X_subset, y_subset, train_size=train_size, random_state=random_state)
        X_subset_train, X_subset_test, y_subset_train, y_subset_test = arrays
        if i == 0:
            X_train = X_subset_train
            X_test = X_subset_test
            y_train = y_subset_train
            y_test = y_subset_test
        else:
            X_train = np.concatenate((X_train, X_subset_train))
            X_test = np.concatenate((X_test, X_subset_test))
            y_train = np.concatenate((y_train, y_subset_train))
            y_test = np.concatenate((y_test, y_subset_test))

    return X_train, X_test, y_train, y_test


def resample_stratified_class_train_test_split(X, y, train_size=0.667, foldid=None, random_state=None):
    """
    Sample in equal proportion.
    
    This will sample in equal proportion.
    
    :param X: An ndarray, the data to be resampled.
    :param y: An ndarray with two classes, 0 and 1.
    :param train_size: A float; specifies what proportion of the data should be used for the training data set. Defaults to 0.667.
    :param foldid: Not currently supported in this function.
    :param random_state: An integer; specifies the seed to be used for the analysis. Defaults to None.
    :return: A tuple of arrays; the arrays X, y split into X_train, X_test, y_train, y_test.
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

    # if random state is passed, set random state
    if random_state:
        setters.set_random_state(random_state)

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


def resample_fold_train_test_split(X, y, foldid=None, train_size=0.667, random_state=None):
    """
    Sample with respect to an identification vector
    
    This will sample the training and test sets so that case identifiers (e.g. subject ID's) are not shared across training and test sets.
    
    :param X: An ndarray, the data to be resampled.
    :param y: An ndarray with two classes, 0 and 1.
    :param train_size: A float; specifies what proportion of the data should be used for the training data set. Defaults to 0.667.
    :param foldid: A vector with length equal to len(y) which identifies cases belonging to the same fold.
    :param random_state: An integer; specifies the seed to be used for the analysis. Defaults to None.
    :return: A tuple of arrays; the arrays X, y split into X_train, X_test, y_train, y_test.
    """
    unique_foldids = np.unique(foldid)
    unique_foldids_train, _ = train_test_split(unique_foldids, train_size=train_size, random_state=random_state)
    mask = np.in1d(foldid, unique_foldids_train)

    # Create train and test splits
    X_train = X[mask, :]
    X_test = X[np.logical_not(mask), :]
    y_train = y[mask]
    y_test = y[np.logical_not(mask)]

    return X_train, X_test, y_train, y_test
