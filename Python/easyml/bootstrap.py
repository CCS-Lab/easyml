"""Utility functions for bootstrapping estimators.
"""
import numpy as np
from sklearn import metrics


__all__ = []


def bootstrap_coefficients(estimator, fit, extract, X, y, n_samples=1000):
    # Initialize containers
    coefs = []

    # Loop over number of iterations
    for _ in range(n_samples):
        # Fit estimator with the training set
        estimator = fit(estimator, X, y)

        # Extract coefficients
        coef = extract(estimator)

        # Save coefficients
        coefs.append(coef)

    # cast to np.ndarray
    coefs = np.array(coefs)

    return coefs


def bootstrap_predictions(estimator, fit, predict, X_train, y_train, X_test, n_samples=1000):
    # Initialize containers
    y_train_preds = []
    y_test_preds = []

    # Loop over number of iterations
    for _ in range(n_samples):
        # Fit estimator with the training set
        estimator = fit(estimator, X_train, y_train)

        # Generate predictions for training and test sets
        y_train_pred = predict(estimator, X_train)
        y_test_pred = predict(estimator, X_test)

        # Save predictions
        y_train_preds.append(y_train_pred)
        y_test_preds.append(y_test_pred)

    # cast to np.ndarray
    y_train_preds = np.array(y_train_preds)
    y_test_preds = np.array(y_test_preds)

    return y_train_preds, y_test_preds


def bootstrap_metrics(estimator, sample, fit, predict, measure, X, y,
                      n_divisions=1000, n_iterations=100):
    # Create temporary containers
    all_train_metrics = []
    all_test_metrics = []

    # Loop over number of divisions
    for i in range(n_divisions):
        # Split data
        X_train, X_test, y_train, y_test = sample(X, y)

        # Create temporary containers
        train_metrics = []
        test_metrics = []

        # Loop over number of iterations
        for j in range(n_iterations):
            # Fit estimator with the training set
            estimator = fit(estimator, X_train, y_train)

            # Generate predictions for training and test sets
            y_train_pred = predict(estimator, X_train)
            y_test_pred = predict(estimator, X_test)

            # Calculate metric on training and test sets
            train_metric = measure(y_train, y_train_pred)
            test_metric = measure(y_test, y_test_pred)

            # Save AUCs
            train_metrics.append(train_metric)
            test_metrics.append(test_metric)

        # Process loop and save in temporary containers
        all_train_metrics.append(np.mean(train_metrics))
        all_test_metrics.append(np.mean(test_metrics))

    # cast to np.ndarray
    all_train_metrics = np.array(all_train_metrics)
    all_test_metrics = np.array(all_test_metrics)

    return all_train_metrics, all_test_metrics


def bootstrap_aucs(estimator, sample, fit, predict, X, y,
                   n_divisions=1000, n_iterations=100):
    return bootstrap_metrics(estimator=estimator, sample=sample,
                             fit=fit, predict=predict, measure=metrics.roc_auc_score,
                             X=X, y=y, n_divisions=n_divisions, n_iterations=n_iterations)


def bootstrap_mses(estimator, sample, fit, predict, X, y,
                    n_divisions=1000, n_iterations=100):
    return bootstrap_metrics(estimator=estimator, sample=sample,
                             fit=fit, predict=predict, measure=metrics.mean_squared_error,
                             X=X, y=y, n_divisions=n_divisions, n_iterations=n_iterations)
