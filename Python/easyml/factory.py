"""Factory functions for quick and easy analysis.
"""
from glmnet import ElasticNet, LogitNet
import matplotlib.pyplot as plt
import numpy as np
from os import path

from .bootstrap import bootstrap_aucs, bootstrap_coefficients, bootstrap_predictions
from .plot import plot_auc_histogram, plot_roc_curve
from .utils import process_coefficients, process_data
from .sample import sample_equal_proportion


__all__ = ['easy_glmnet']


def easy_glmnet(data, dependent_variable=None, family='gaussian', exclude_variables=None, train_size=0.667,
                n_divisions=1000, n_iterations=10, n_samples=1000, out_directory='.', random_state=None,
                **kwargs):

    # Create model
    if family == 'gaussian':
        model = ElasticNet(**kwargs)
    elif family == 'binomial':
        model = LogitNet(**kwargs)
    else:
        raise ValueError

    # Process the data
    X, y = process_data(data, dependent_variable=dependent_variable, exclude_variables=exclude_variables)

    # Bootstrap coefficients
    coefs = bootstrap_coefficients(model, X, y)

    # Process coefficients
    betas = process_coefficients(coefs)
    betas.to_csv(path.join(out_directory, 'betas.csv'), index=False)

    # Split data
    mask = sample_equal_proportion(y, proportion=train_size, random_state=random_state)
    y_train = y[mask]
    y_test = y[np.logical_not(mask)]
    X_train = X[mask, :]
    X_test = X[np.logical_not(mask), :]

    # Bootstrap predictions
    all_y_train_scores, all_y_test_scores = bootstrap_predictions(model, X_train, y_train, X_test, n_samples=n_samples)

    # Generate scores for training and test sets
    y_train_scores_mean = np.mean(all_y_train_scores, axis=0)
    y_test_scores_mean = np.mean(all_y_test_scores, axis=0)

    # Compute ROC curve and ROC area for training
    plot_roc_curve(y_train, y_train_scores_mean)
    plt.savefig(path.join(out_directory, 'train_roc_curve.png'))

    # Compute ROC curve and ROC area for test
    plot_roc_curve(y_test, y_test_scores_mean)
    plt.savefig(path.join(out_directory, 'test_roc_curve.png'))

    # Bootstrap training and test AUCS
    all_train_aucs, all_test_aucs = bootstrap_aucs(model, X, y, n_divisions=n_divisions, n_iterations=n_iterations)

    # Plot histogram of training AUCS
    plot_auc_histogram(all_train_aucs)
    plt.savefig(path.join(out_directory, 'train_auc_distribution.png'))

    # Plot histogram of test AUCS
    plot_auc_histogram(all_test_aucs)
    plt.savefig(path.join(out_directory, 'test_auc_distribution.png'))
