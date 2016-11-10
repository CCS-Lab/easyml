"""Utility functions for bootstrapping estimators.
"""
import numpy as np
from sklearn import metrics


__all__ = ['bootstrap_aucs', 'bootstrap_coefficients', 'bootstrap_predictions']


def bootstrap_aucs(estimator, sampler, X, y, n_divisions=1000, n_iterations=100):
    # Create temporary containers
    all_train_aucs = []
    all_test_aucs = []

    # Loop over number of divisions
    for i in range(n_divisions):
        # Split data
        X_train, X_test, y_train, y_test = sampler(X, y)

        # Create temporary containers
        train_aucs = []
        test_aucs = []

        # Loop over number of iterations
        for j in range(n_iterations):
            # Fit estimator with the training set
            estimator.fit(X_train, y_train)

            # Generate scores for training and test sets
            y_train_scores = estimator.predict_proba(X_train)[:, 1]
            y_test_scores = estimator.predict_proba(X_test)[:, 1]

            # Calculate AUC on training and test sets
            train_auc = metrics.roc_auc_score(y_train, y_train_scores)
            test_auc = metrics.roc_auc_score(y_test, y_test_scores)

            # Save AUCs
            train_aucs.append(train_auc)
            test_aucs.append(test_auc)

        # Process loop and save in temporary containers
        all_train_aucs.append(np.mean(train_aucs))
        all_test_aucs.append(np.mean(test_aucs))

    # cast to np.ndarray
    all_train_aucs = np.array(all_train_aucs)
    all_test_aucs = np.array(all_test_aucs)

    return all_train_aucs, all_test_aucs


def bootstrap_coefficients(estimator, X, y, n_samples=1000):
    # Initialize containers
    coefs = []

    # Loop over number of iterations
    for _ in range(n_samples):
        # Fit estimator with the training set
        estimator.fit(X, y)

        # Extract and save coefficients
        coef = estimator.coef_
        if len(coef.shape) == 1:
            pass
        elif len(coef.shape) > 1:
            coef = coef[0]
        else:
            raise ValueError

        coefs.append(list(coef))

    # cast to np.ndarray
    coefs = np.array(coefs)

    return coefs


def bootstrap_predictions(estimator, X_train, y_train, X_test, n_samples=1000):
    # Initialize containers
    all_y_train_predictions = []
    all_y_test_predictions = []

    # Loop over number of iterations
    for _ in range(n_samples):
        # Fit estimator with the training set
        estimator.fit(X_train, y_train)

        # Generate predictions for training and test sets
        y_train_predictions = estimator.predict_proba(X_train)[:, 1]
        y_test_predictions = estimator.predict_proba(X_test)[:, 1]

        # Save predictions
        all_y_train_predictions.append(y_train_predictions)
        all_y_test_predictions.append(y_test_predictions)

    # cast to np.ndarray
    all_y_train_predictions = np.array(all_y_train_predictions)
    all_y_test_predictions = np.array(all_y_test_predictions)

    return all_y_train_predictions, all_y_test_predictions
