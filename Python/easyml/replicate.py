"""Utility functions for replicating estimates.
"""
import concurrent.futures
import numpy as np
import os
import progressbar
from sklearn import metrics


__all__ = []


def replicate_coefficient(estimator, fit, extract_coefficients, X, y):
    # Fit estimator with the training set
    estimator = fit(estimator, X, y)

    # Extract coefficients
    coef = extract_coefficients(estimator)

    # Save coefficients
    return coef


def replicate_coefficients(estimator, fit, extract_coefficients,
                           preprocessor, X, y,
                           categorical_variables=None,
                           n_samples=1000, progress_bar=True, n_core=1):
    # Initialize progress bar (optional)
    if progress_bar:
        bar = progressbar.ProgressBar(max_value=n_samples)
        i = 0

    # Preprocess data
    X = preprocessor(X, categorical_variables=categorical_variables)

    # Initialize containers
    coefs = []

    # Evaluate parallelism
    if n_core == 1:
        # Run sequentially
        print("Replicating coefficients:")

        # Loop over number of iterations
        for _ in range(n_samples):
            coef = replicate_coefficient(estimator, fit, extract_coefficients, X, y)
            coefs.append(coef)

            # Increment progress bar
            if progress_bar:
                bar.update(i)
                i += 1

    elif n_core > 1:
        # Run in parallel using n_core cores
        print("Replicating coefficients in parallel:")

        # Handle case where n_core > os.cpu_count()
        if n_core > os.cpu_count():
            n_core = os.cpu_count()

        # Loop over number of iterations
        indexes = range(n_samples)
        with concurrent.futures.ProcessPoolExecutor(max_workers=n_core) as executor:
            futures = {executor.submit(replicate_coefficient, estimator, fit, extract_coefficients, X, y): ix for ix in indexes}
            for future in concurrent.futures.as_completed(futures):
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
    coefs = np.asarray(coefs)

    return coefs


def replicate_prediction(estimator, fit_model, predict_model, X_train, y_train, X_test):
    # Fit estimator with the training set
    estimator = fit_model(estimator, X_train, y_train)

    # Generate predictions for training and test sets
    y_train_pred = predict_model(estimator, X_train)
    y_test_pred = predict_model(estimator, X_test)

    # Save predictions
    return y_train_pred, y_test_pred


def replicate_predictions(estimator, fit_model, predict_model, preprocessor, X_train, y_train, X_test,
                          categorical_variables=None, n_samples=1000, progress_bar=True, n_core=1):
    # Initialize progress bar (optional)
    if progress_bar:
        bar = progressbar.ProgressBar(max_value=n_samples)
        i = 0

    # Preprocess data
    X_train, X_test = preprocessor(X_train, X_test, categorical_variables=categorical_variables)

    # Initialize containers
    y_train_preds = []
    y_test_preds = []

    # Evaluate parallelism
    if n_core == 1:
        # Run sequentially
        print("Replicating predictions:")

        # Loop over number of iterations
        for _ in range(n_samples):
            y_train_pred, y_test_pred = replicate_prediction(estimator, fit_model, predict_model, X_train, y_train, X_test)
            y_train_preds.append(y_train_pred)
            y_test_preds.append(y_test_pred)

            # Increment progress bar
            if progress_bar:
                bar.update(i)
                i += 1

    elif n_core > 1:
        # Run in parallel using n_core cores
        print("Replicating predictions in parallel:")

        # Handle case where n_core > os.cpu_count()
        if n_core > os.cpu_count():
            n_core = os.cpu_count()

        # Loop over number of iterations
        indexes = range(n_samples)
        with concurrent.futures.ProcessPoolExecutor(max_workers=n_core) as executor:
            futures = {executor.submit(replicate_prediction, estimator, fit_model, predict_model, X_train, y_train, X_test): ix for ix in indexes}
            for future in concurrent.futures.as_completed(futures):
                try:
                    y_train_pred, y_test_pred = future.result()
                    y_train_preds.append(y_train_pred)
                    y_test_preds.append(y_test_pred)
                except:
                    y_train_preds.append(None)
                    y_test_preds.append(None)

                # Increment progress bar
                if progress_bar:
                    bar.update(i)
                    i += 1
    else:
        raise ValueError

    # cast to np.ndarray
    y_train_preds = np.asarray(y_train_preds)
    y_test_preds = np.asarray(y_test_preds)

    return y_train_preds, y_test_preds


def replicate_metric(estimator, sampler, fit_model, predict_model, preprocessor, measure, X, y,
                     categorical_variables=None, n_iterations=100):
    # Split data
    try:
        X_train, X_test, y_train, y_test = sampler(X, y)
    except Exception as e:
        print('Sample Exception: {}'.format(e))

    # Preprocess data
    X_train, X_test = preprocessor(X_train, X_test, categorical_variables=categorical_variables)

    # Create temporary containers
    train_metrics = []
    test_metrics = []

    # Loop over number of iterations
    for _ in range(n_iterations):
        # Fit estimator with the training set
        try:
            results = fit_model(estimator, X_train, y_train)
        except Exception as e:
            print('Model Exception: {}'.format(e))

        # Generate predictions for training and test sets
        try:
            y_train_pred = predict_model(results, X_train)
            y_test_pred = predict_model(results, X_test)
        except Exception as e:
            print('Predict Exception: {}'.format(e))

        # Calculate metric on training and test sets
        try:
            train_metric = measure(y_train, y_train_pred)
            test_metric = measure(y_test, y_test_pred)
        except Exception as e:
            print('Measure Exception: {}'.format(e))

        # Save metrics
        try:
            train_metrics.append(train_metric)
            test_metrics.append(test_metric)
        except Exception as e:
            print('Append Exception: {}'.format(e))


    # Take mean of metrics
    try:
        mean_train_metric = np.mean(np.asarray(train_metrics))
        mean_test_metric = np.mean(np.asarray(test_metrics))
    except Exception as e:
        print('Mean Exception: {}'.format(e))

    # Save metrics
    return mean_train_metric, mean_test_metric


def replicate_metrics(estimator, sampler, fit_model, predict_model, preprocessor, measure, X, y,
                      categorical_variables=None, n_divisions=1000, n_iterations=100, progress_bar=True, n_core=1):
    # Initialize progress bar (optional)
    if progress_bar:
        bar = progressbar.ProgressBar(max_value=n_divisions)
        i = 0

    # Create temporary containers
    mean_train_metrics = []
    mean_test_metrics = []

    # Evaluate parallelism
    if True:
        # Run sequentially
        print("Replicating metrics (parallelism is currently disabled for this function):")

        # Loop over number of divisions
        for _ in range(n_divisions):
            # Bootstrap metric
            mean_train_metric, mean_test_metric = replicate_metric(estimator, sampler, fit_model,
                                                                   predict_model, preprocessor, measure, X, y,
                                                                   categorical_variables, n_iterations)

            # Process loop and save in temporary containers
            mean_train_metrics.append(mean_train_metric)
            mean_test_metrics.append(mean_test_metric)

            # Increment progress bar
            if progress_bar:
                bar.update(i)
                i += 1

    elif False:
        # Run in parallel using n_core cores
        print("Replicating metrics in parallel:")

        # Handle case where n_core > os.cpu_count()
        if n_core > os.cpu_count():
            n_core = os.cpu_count()

        # Loop over number of iterations
        indexes = range(n_divisions)
        with concurrent.futures.ProcessPoolExecutor(max_workers=n_core) as executor:
            futures = {executor.submit(replicate_metric, estimator, sample, fit_model, predict_model, measure, X, y, n_iterations): ix for ix in indexes}
            for future in concurrent.futures.as_completed(futures):
                try:
                    mean_train_metric, mean_test_metric = future.result()
                except Exception as e:
                    print('Exception: {}'.format(e))
                    mean_train_metric = None
                    mean_test_metric = None

                # Save metrics
                # print(mean_train_metric, mean_test_metric)
                mean_train_metrics.append(mean_train_metric)
                mean_test_metrics.append(mean_test_metric)

                # Increment progress bar
                if progress_bar:
                    bar.update(i)
                    i += 1
    else:
        raise ValueError

    # cast to np.ndarray
    mean_train_metrics = np.asarray(mean_train_metrics)
    mean_test_metrics = np.asarray(mean_test_metrics)

    return mean_train_metrics, mean_test_metrics


def replicate_aucs(estimator, sampler, fit_model, predict_model, preprocessor, X, y,
                   categorical_variables=None, n_divisions=1000, n_iterations=100, progress_bar=True, n_core=1):
    return replicate_metrics(estimator=estimator, sampler=sampler,
                             fit_model=fit_model, predict_model=predict_model,
                             preprocessor=preprocessor, measure=metrics.roc_auc_score,
                             X=X, y=y, categorical_variables=categorical_variables,
                             n_divisions=n_divisions, n_iterations=n_iterations,
                             progress_bar=progress_bar, n_core=n_core)


def replicate_mses(estimator, sampler, fit_model, predict_model, preprocessor, X, y,
                   categorical_variables=None, n_divisions=1000, n_iterations=100, progress_bar=True, n_core=1):
    return replicate_metrics(estimator=estimator, sampler=sampler,
                             fit_model=fit_model, predict_model=predict_model,
                             preprocessor=preprocessor, measure=metrics.mean_squared_error,
                             X=X, y=y, categorical_variables=categorical_variables,
                             n_divisions=n_divisions, n_iterations=n_iterations,
                             progress_bar=progress_bar, n_core=n_core)
