"""Factory functions for quick and easy analysis.
"""
from glmnet import ElasticNet, LogitNet
import matplotlib.pyplot as plt
import numpy as np
from os import path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from .bootstrap import bootstrap_aucs, bootstrap_coefficients, bootstrap_mses, bootstrap_predictions
from .plot import plot_auc_histogram, plot_gaussian_predictions, plot_mse_histogram, plot_roc_curve
from .utils import process_coefficients
from .sample import sample_equal_proportion


__all__ = ['easy_glmnet']


def easy_glmnet(data, dependent_variable, family='gaussian', sample=None,
                exclude_variables=None, categorical_variables=None,
                standardize_data=True, train_size=0.667, survival_rate_cutoff=0.05,
                n_samples=1000, n_divisions=1000, n_iterations=10,
                out_directory='.', random_state=None, progress_bar=True,
                n_core=1, **kwargs):
    # Handle random state
    if random_state is not None:
        np.random.seed(random_state)

    # Handle columns
    column_names = data.columns
    column_names = [c for c in column_names if c != dependent_variable]
    if exclude_variables is not None:
        column_names = [c for c in column_names if c not in exclude_variables]

    # Exclude certain variables and y
    y = data[dependent_variable].values
    data = data.drop(dependent_variable, axis=1)
    if exclude_variables is not None:
        data = data.drop(exclude_variables, axis=1)

    # If True, standardize the data
    if standardize_data:
        stdsc = StandardScaler()
        if categorical_variables is None:
            X = data.values
            X = stdsc.fit_transform(X)
        else:
            X_categorical = data[categorical_variables].values
            X_numeric = data.drop(categorical_variables, axis=1).values
            X_std = stdsc.fit_transform(X_numeric)
            X = np.concatenate([X_categorical, X_std], axis=1)
            column_names = [c for c in column_names if c not in categorical_variables]
            column_names = categorical_variables + column_names

    # Set glmnet specific handlers
    def fit(e, X, y):
        return e.fit(X, y)

    if family == 'gaussian':
        # Set gaussian specific functions
        model = ElasticNet(**kwargs)

        def extract(e):
            return e.coef_

        def predict(e, X):
            return e.predict(X)

        if sample is None:
            sample = train_test_split

        # Bootstrap coefficients
        coefs = bootstrap_coefficients(model, fit, extract, X, y, n_samples=n_samples,
                                       progress_bar=progress_bar, n_core=n_core)

        # Process coefficients
        betas = process_coefficients(coefs, column_names, survival_rate_cutoff=survival_rate_cutoff)
        betas.to_csv(path.join(out_directory, 'betas.csv'), index=False)

        # Split data
        X_train, X_test, y_train, y_test = sample(X, y, train_size=train_size)

        # Bootstrap predictions
        y_train_pred, y_test_pred = bootstrap_predictions(model, fit, predict, X_train, y_train, X_test,
                                                          n_samples=n_samples, progress_bar=progress_bar,
                                                          n_core=n_core)

        # Take average of predictions for training and test sets
        y_train_pred_mean = np.mean(y_train_pred, axis=0)
        y_test_pred_mean = np.mean(y_test_pred, axis=0)

        # Plot the gaussian predictions for training
        plot_gaussian_predictions(y_train, y_train_pred_mean)
        plt.savefig(path.join(out_directory, 'train_predictions.png'))

        # Plot the gaussian predictions for test
        plot_gaussian_predictions(y_test, y_test_pred_mean)
        plt.savefig(path.join(out_directory, 'test_predictions.png'))

        # Bootstrap training and test MSEs
        train_mses, test_mses = bootstrap_mses(model, sample, fit, predict, X, y,
                                               n_divisions=n_divisions, n_iterations=n_iterations,
                                               progress_bar=progress_bar, n_core=n_core)

        # Plot histogram of training MSEs
        plot_mse_histogram(train_mses)
        plt.savefig(path.join(out_directory, 'train_mse_distribution.png'))

        # Plot histogram of test MSEs
        plot_mse_histogram(test_mses)
        plt.savefig(path.join(out_directory, 'test_mse_distribution.png'))

    elif family == 'binomial':
        # Set binomial specific functions
        model = LogitNet(**kwargs)

        def extract(e):
            return e.coef_[0]

        def predict(e, X):
            return e.predict_proba(X)[:, 1]

        if sample is None:
            sample = sample_equal_proportion

        # Bootstrap coefficients
        coefs = bootstrap_coefficients(model, fit, extract, X, y, n_samples=n_samples,
                                       progress_bar=progress_bar, n_core=n_core)

        # Process coefficients
        betas = process_coefficients(coefs, column_names, survival_rate_cutoff=survival_rate_cutoff)
        betas.to_csv(path.join(out_directory, 'betas.csv'), index=False)

        # Split data
        X_train, X_test, y_train, y_test = sample(X, y, train_size=train_size)

        # Bootstrap predictions
        y_train_pred, y_test_pred = bootstrap_predictions(model, fit, predict, X_train, y_train, X_test,
                                                          n_samples=n_samples, progress_bar=progress_bar,
                                                          n_core=n_core)

        # Take average of predictions for training and test sets
        y_train_pred_mean = np.mean(y_train_pred, axis=0)
        y_test_pred_mean = np.mean(y_test_pred, axis=0)

        # Plot the ROC curve for training
        plot_roc_curve(y_train, y_train_pred_mean)
        plt.savefig(path.join(out_directory, 'train_roc_curve.png'))

        # Plot the ROC curve for test
        plot_roc_curve(y_test, y_test_pred_mean)
        plt.savefig(path.join(out_directory, 'test_roc_curve.png'))

        # Bootstrap training and test AUCSs
        train_aucs, test_aucs = bootstrap_aucs(model, sample, fit, predict, X, y,
                                               n_divisions=n_divisions, n_iterations=n_iterations,
                                               progress_bar=progress_bar, n_core=n_core)

        # Plot histogram of training AUCs
        plot_auc_histogram(train_aucs)
        plt.savefig(path.join(out_directory, 'train_auc_distribution.png'))

        # Plot histogram of test AUCs
        plot_auc_histogram(test_aucs)
        plt.savefig(path.join(out_directory, 'test_auc_distribution.png'))

    else:
        raise ValueError
