"""Utility functions for bootstrapping estimators.
"""
import concurrent.futures
import numpy as np
import os
import progressbar
from sklearn import metrics


__all__ = []


def bootstrap_coefficient(estimator, fit, extract, X, y):
    # Fit estimator with the training set
    estimator = fit(estimator, X, y)

    # Extract coefficients
    coef = extract(estimator)

    # Save coefficients
    return coef


def bootstrap_coefficients(estimator, fit, extract, X, y,
                           n_samples=1000, progress_bar=True, n_core=1):
    # Initialize progress bar (optional)
    if progress_bar:
        bar = progressbar.ProgressBar(max_value=n_samples)
        i = 0

    # Initialize containers
    coefs = []

    # Evaluate parallelism
    if n_core == 1:
        # Run sequentially
        print("Bootstrapping coefficients:")

        # Loop over number of iterations
        for _ in range(n_samples):
            coef = bootstrap_coefficients(estimator, fit, extract, X, y)
            coefs.append(coef)

            #increment progress bar
            if progress_bar:
                bar.update(i)
                i += 1

    elif n_core > 1:
        # Run in parallel using n_core cores
        print("Bootstrapping coefficients in parallel:")

        # Handle case where n_core > os.cpu_count()
        if n_core > os.cpu_count():
            n_core = os.cpu_count()

        # Loop over number of iterations
        indexes = range(n_samples)
        with concurrent.futures.ThreadPoolExecutor(max_workers=n_core) as executor:
            future_to_coef = {executor.submit(bootstrap_coefficient, estimator, fit, extract, X, y): ix for ix in indexes}
            for future in concurrent.futures.as_completed(future_to_coef):
                try:
                    coef = future.result()
                    coefs.append(coef)
                except:
                    coefs.append(None)

                # increment progress bar
                if progress_bar:
                    bar.update(i)
                    i += 1
    else:
        raise ValueError

    # cast to np.ndarray
    coefs = np.array(coefs)

    return coefs


def bootstrap_prediction(estimator, fit, predict, X_train, y_train, X_test):
    # Fit estimator with the training set
    estimator = fit(estimator, X_train, y_train)

    # Generate predictions for training and test sets
    y_train_pred = predict(estimator, X_train)
    y_test_pred = predict(estimator, X_test)

    # Save predictions
    return y_train_pred, y_test_pred


def bootstrap_predictions(estimator, fit, predict, X_train, y_train, X_test,
                          n_samples=1000, progress_bar=True, n_core=1):
    # Initialize progress bar (optional)
    if progress_bar:
        bar = progressbar.ProgressBar(max_value=n_samples)
        i = 0

    # Initialize containers
    y_train_preds = []
    y_test_preds = []

    # Evaluate parallelism
    if n_core == 1:
        # Run sequentially
        print("Bootstrapping predictions:")

        # Loop over number of iterations
        for _ in range(n_samples):
            y_train_pred, y_test_pred = bootstrap_prediction(estimator, fit, predict, X_train, y_train, X_test)
            y_train_preds.append(y_train_pred)
            y_test_preds.append(y_test_pred)

            #increment progress bar
            if progress_bar:
                bar.update(i)
                i += 1

    elif n_core > 1:
        # Run in parallel using n_core cores
        print("Bootstrapping predictions in parallel:")

        # Handle case where n_core > os.cpu_count()
        if n_core > os.cpu_count():
            n_core = os.cpu_count()

        # Loop over number of iterations
        indexes = range(n_samples)
        with concurrent.futures.ThreadPoolExecutor(max_workers=n_core) as executor:
            futures = {executor.submit(bootstrap_prediction(estimator, fit, predict, X_train, y_train, X_test)): ix for ix in indexes}
            for future in concurrent.futures.as_completed(futures):
                try:
                    y_train_pred, y_test_pred = future.result()
                    y_train_preds.append(y_train_pred)
                    y_test_preds.append(y_test_pred)
                except:
                    y_train_preds.append(None)
                    y_test_preds.append(None)

                # increment progress bar
                if progress_bar:
                    bar.update(i)
                    i += 1
    else:
        raise ValueError

    # cast to np.ndarray
    y_train_preds = np.array(y_train_preds)
    y_test_preds = np.array(y_test_preds)

    return y_train_preds, y_test_preds


def bootstrap_metric(estimator, fit, predict, measure, X_train, y_train, X_test, y_test):
    # Fit estimator with the training set
    estimator = fit(estimator, X_train, y_train)

    # Generate predictions for training and test sets
    y_train_pred = predict(estimator, X_train)
    y_test_pred = predict(estimator, X_test)

    # Calculate metric on training and test sets
    train_metric = measure(y_train, y_train_pred)
    test_metric = measure(y_test, y_test_pred)

    # Save metrics
    return train_metric, test_metric


def bootstrap_metrics(estimator, sample, fit, predict, measure, X, y,
                      n_divisions=1000, n_iterations=100, progress_bar=True, n_core=1):
    # Initialize progress bar (optional)
    if progress_bar:
        bar = progressbar.ProgressBar(max_value=n_divisions)
        i = 0

    # Create temporary containers
    all_train_metrics = []
    all_test_metrics = []

    # Loop over number of divisions
    for _ in range(n_divisions):
        # Split data
        X_train, X_test, y_train, y_test = sample(X, y)

        # Create temporary containers
        train_metrics = []
        test_metrics = []

        # Evaluate parallelism
        if n_core == 1:
            # Run sequentially
            print("Bootstrapping metrics:")

            # Loop over number of iterations
            for _ in range(n_iterations):
                train_metric, test_metric = bootstrap_metric(estimator, fit, predict, measure, X_train, y_train, X_test, y_test)
                train_metrics.append(train_metric)
                test_metrics.append(test_metric)

                # increment progress bar
                if progress_bar:
                    bar.update(i)
                    i += 1

        elif n_core > 1:
            # Run in parallel using n_core cores
            print("Bootstrapping predictions in parallel:")

            # Handle case where n_core > os.cpu_count()
            if n_core > os.cpu_count():
                n_core = os.cpu_count()

            # Loop over number of iterations
            indexes = range(n_iterations)
            with concurrent.futures.ThreadPoolExecutor(max_workers=n_core) as executor:
                future_to_metric = {executor.submit(bootstrap_metric(estimator, fit, predict, X_train, y_train, X_test)): ix for ix in indexes}
                for future in concurrent.futures.as_completed(future_to_metric):
                    try:
                        train_metric, test_metric = future.result()
                        train_metrics.append(train_metric)
                        test_metrics.append(test_metric)
                    except:
                        train_metrics.append(None)
                        test_metrics.append(None)

        # increment progress bar
        if progress_bar:
            bar.update(i)
            i += 1
        else:
            raise ValueError

        # Process loop and save in temporary containers
        all_train_metrics.append(np.mean(np.array(train_metrics)))
        all_test_metrics.append(np.mean(np.array(test_metrics)))

    # cast to np.ndarray
    all_train_metrics = np.array(all_train_metrics)
    all_test_metrics = np.array(all_test_metrics)

    return all_train_metrics, all_test_metrics


def bootstrap_aucs(estimator, sample, fit, predict, X, y,
                   n_divisions=1000, n_iterations=100, progress_bar=True, n_core=1):
    return bootstrap_metrics(estimator=estimator, sample=sample,
                             fit=fit, predict=predict, measure=metrics.roc_auc_score,
                             X=X, y=y, n_divisions=n_divisions, n_iterations=n_iterations,
                             progress_bar=progress_bar, n_core=n_core)


def bootstrap_mses(estimator, sample, fit, predict, X, y,
                    n_divisions=1000, n_iterations=100, progress_bar=True, n_core=1):
    return bootstrap_metrics(estimator=estimator, sample=sample,
                             fit=fit, predict=predict, measure=metrics.mean_squared_error,
                             X=X, y=y, n_divisions=n_divisions, n_iterations=n_iterations,
                             progress_bar=progress_bar, n_core=n_core)
