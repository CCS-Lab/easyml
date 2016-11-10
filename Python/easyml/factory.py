"""Factory functions for quick and easy analysis.
"""
from glmnet import ElasticNet, LogitNet
import matplotlib.pyplot as plt
import numpy as np
from os import path
from sklearn.model_selection import train_test_split

from .bootstrap import bootstrap_aucs, bootstrap_coefficients, bootstrap_predictions
from .plot import plot_auc_histogram, plot_roc_curve
from .utils import process_coefficients, process_data
from .sample import sample_equal_proportion


__all__ = ['easy_glmnet']


def easy_glmnet(data, dependent_variable=None, family='gaussian', sampler=None,
                exclude_variables=None, train_size=0.667, n_divisions=1000, n_iterations=10,
                n_samples=1000, out_directory='.', random_state=None):

    # Process the data
    X, y = process_data(data, dependent_variable=dependent_variable, exclude_variables=exclude_variables)

    if family == 'gaussian':
        # Set binomial specific functions
        model = ElasticNet()

        if sampler is None:
            sampler = train_test_split

        # Bootstrap coefficients
        coefs = bootstrap_coefficients(model, X, y)

        # Process coefficients
        betas = process_coefficients(coefs)
        betas.to_csv(path.join(out_directory, 'betas.csv'), index=False)

    elif family == 'binomial':
        # Set binomial specific functions
        model = LogitNet()

        if sampler is None:
            sampler = sample_equal_proportion

        # Bootstrap coefficients
        coefs = bootstrap_coefficients(model, X, y)

        # Process coefficients
        betas = process_coefficients(coefs)
        betas.to_csv(path.join(out_directory, 'betas.csv'), index=False)

        # Split data
        X_train, X_test, y_train, y_test = sampler(X, y)

        # Bootstrap predictions
        y_train_pred, y_test_pred = bootstrap_predictions(model, X_train, y_train, X_test, n_samples=n_samples)

        # Generate scores for training and test sets
        y_train_pred_mean = np.mean(y_train_pred, axis=0)
        y_test_pred_mean = np.mean(y_test_pred, axis=0)

        # Compute ROC curve and ROC area for training
        plot_roc_curve(y_train, y_train_pred_mean)
        plt.savefig(path.join(out_directory, 'train_roc_curve.png'))

        # Compute ROC curve and ROC area for test
        plot_roc_curve(y_test, y_test_pred_mean)
        plt.savefig(path.join(out_directory, 'test_roc_curve.png'))

        # Bootstrap training and test AUCS
        train_aucs, test_aucs = bootstrap_aucs(model, sampler, X, y, n_divisions=n_divisions, n_iterations=n_iterations)

        # Plot histogram of training AUCS
        plot_auc_histogram(train_aucs)
        plt.savefig(path.join(out_directory, 'train_auc_distribution.png'))

        # Plot histogram of test AUCS
        plot_auc_histogram(test_aucs)
        plt.savefig(path.join(out_directory, 'test_auc_distribution.png'))

    else:
        raise ValueError
