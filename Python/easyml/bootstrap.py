"""Utility functions for bootstrapping estimators.
"""
import numpy as np
from sklearn import metrics

from .sample import sample_equal_proportion


__all__ = ['bootstrap_aucs', 'bootstrap_coefficients', 'bootstrap_predictions']


def bootstrap_aucs(estimator, X, y, n_divisions=1000, n_iterations=100):
    # Create temporary containers
    all_train_aucs = []
    all_test_aucs = []

    # Loop over number of divisions
    for i in range(n_divisions):
        # Split data
        mask = sample_equal_proportion(y, random_state=i)
        y_train = y[mask]
        y_test = y[np.logical_not(mask)]
        X_train = X[mask, :]
        X_test = X[np.logical_not(mask), :]

        # Create temporary containers
        train_aucs = []
        test_aucs = []

        # Loop over number of iterations
        for j in range(n_iterations):
            # Fit estimator with the training set
            e = estimator()
            e = e.fit(X_train, y_train)

            # Generate scores for training and test sets
            y_train_scores = e.predict_proba(X_train)[:, 1]
            y_test_scores = e.predict_proba(X_test)[:, 1]

            # Calculate AUC on training and test sets
            train_auc = metrics.roc_auc_score(y_train, y_train_scores)
            test_auc = metrics.roc_auc_score(y_test, y_test_scores)

            # Save AUCs
            train_aucs.append(train_auc)
            test_aucs.append(test_auc)

        # Process loop and save in temporary containers
        all_train_aucs.append(np.mean(train_aucs))
        all_test_aucs.append(np.mean(test_aucs))
    return all_train_aucs, all_test_aucs


def bootstrap_coefficients(estimator, X, y, n_samples=1000):
    # Initialize containers
    coefs = []

    # Loop over number of iterations
    for _ in range(n_samples):
        # Fit estimator with the training set
        e = estimator()
        e.fit(X, y)

        # Extract and save coefficients
        coefs.append(list(e.coef_[0]))

    return coefs


def bootstrap_predictions(estimator, X_train, y_train, X_test, y_test, n_samples=1000):
    # Initialize containers
    all_y_train_predictions = []
    all_y_test_predictions = []

    # Loop over number of iterations
    for _ in range(n_samples):
        # Fit estimator with the training set
        e = estimator()
        e.fit(X_train, y_train)

        # Generate predictions for training and test sets
        y_train_predictions = e.predict_proba(X_train)[:, 1]
        y_test_predictions = e.predict_proba(X_test)[:, 1]

        # Save predictions
        all_y_train_predictions.append(y_train_predictions)
        all_y_test_predictions.append(y_test_predictions)

    return all_y_train_predictions, all_y_test_predictions
