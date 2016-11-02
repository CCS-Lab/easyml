"""Utility functions for sampling data.
"""
import numpy as np


__all__ = ['sample_equal_proportion']


def sample_equal_proportion(y, proportion=0.667, random_state=None):
    """Sample in equal proportion.

    Parameters
    ----------
    :param y: array, shape (n_obs) Input data to be split
    :param proportion: float, default: 0.667
        Proportion to split into train and test
    :param random_state: int seed, default: None
        The seed of the pseudo random number generator to use when shuffling the data.
    TODO figure out best practices for documenting Python functions

    Returns
    -------
    self: array, shape (n_obs)
        A boolean array of length n_obs where True represents that observation should be in the train set.
    """
    # Set random_state
    if random_state is not None:
        np.random.seed(random_state)

    # calculate number of observations
    n_obs = len(y)

    # identify index number for class1 and class2
    index_class1 = np.where(y == 0)[0]
    index_class2 = np.where(y == 1)[0]

    # calculate  number of class1 and class2 observations
    n_class1 = len(index_class1)
    n_class2 = len(index_class2)

    # calculate number of class1 and class2 observations in the train set
    n_class1_train = int(np.round(n_class1 * proportion))
    n_class2_train = int(np.round(n_class2 * proportion))

    # generate indices for class1 and class2 observations in the train set
    index_class1_train = np.random.choice(index_class1, size=n_class1_train, replace=False)
    index_class2_train = np.random.choice(index_class2, size=n_class2_train, replace=False)
    index_train = list(np.append(index_class1_train, index_class2_train))

    # return a boolean vector of len n_obs where TRUE represents
    # that observation should be in the train set
    sequence = list(np.arange(n_obs))
    index = np.array([True if i in index_train else False for i in sequence])
    return index
