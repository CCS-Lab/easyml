"""Functions for random forest analysis.
"""
import matplotlib.pyplot as plt
import numpy as np
from os import path
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split

from .replicate import replicate_aucs, replicate_mses, replicate_predictions
from .plot import plot_auc_histogram, plot_gaussian_predictions, plot_mse_histogram, plot_roc_curve
from .preprocess import preprocess_identity
from .resample import sample_equal_proportion
from .utils import set_random_state


__all__ = ['easy_random_forest']


def easy_random_forest(data, dependent_variable, family='gaussian',
                       sampler=None, preprocessor=None,
                       exclude_variables=None, categorical_variables=None, train_size=0.667,
                       n_samples=1000, n_divisions=1000, n_iterations=10,
                       out_directory='.', random_state=None, progress_bar=True,
                       n_core=1, **kwargs):
    # Make it run in sequential for now
    n_core = 1

    # Set random state
    set_random_state(random_state)

    # Set columns
    column_names = data.columns

    # Exclude variables
    if exclude_variables is not None:
        data = data.drop(exclude_variables, axis=1)
        column_names = [c for c in column_names if c not in exclude_variables]

    # Isolate y
    y = data[dependent_variable].values

    # Remove y column name from column names
    column_names = [c for c in column_names if c != dependent_variable]
    data = data.drop(dependent_variable, axis=1)

    # Isolate X
    X = data.values

    # Move categorical names to the front when there are categorical variables
    if categorical_variables is not None and preprocessor is not None:
        column_names = [c for c in column_names if c not in categorical_variables]
        column_names = categorical_variables + column_names
        categorical_variables = np.array([True if c in categorical_variables else False for c in column_names])

    # Set preprocessor function
    if preprocessor is None:
        preprocessor = preprocess_identity

    # Set random_forest specific handlers
    def fit_model(e, X, y):
        return e.fit(X, y)

    if family == 'gaussian':
        # Set gaussian specific functions
        model = RandomForestRegressor(**kwargs)

        def predict_model(e, X):
            return e.predict(X)

        if sampler is None:
            sampler = train_test_split

        # Split data
        X_train, X_test, y_train, y_test = sampler(X, y, train_size=train_size)

        # Replicate predictions
        y_train_pred, y_test_pred = replicate_predictions(model, fit_model, predict_model, preprocessor,
                                                          X_train, y_train, X_test,
                                                          categorical_variables=categorical_variables,
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

        # Replicate training and test MSEs
        train_mses, test_mses = replicate_mses(model, sampler, fit_model, predict_model, preprocessor, X, y,
                                               categorical_variables=categorical_variables,
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
        model = RandomForestClassifier(**kwargs)

        def predict_model(e, X):
            return e.predict_proba(X)[:, 1]

        if sampler is None:
            sampler = sample_equal_proportion

        # Split data
        X_train, X_test, y_train, y_test = sampler(X, y, train_size=train_size)

        # Replicate predictions
        y_train_pred, y_test_pred = replicate_predictions(model, fit_model, predict_model, preprocessor,
                                                          X_train, y_train, X_test,
                                                          categorical_variables=categorical_variables,
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

        # Replicate training and test AUCSs
        train_aucs, test_aucs = replicate_aucs(model, sampler, fit_model, predict_model, preprocessor, X, y,
                                               categorical_variables=categorical_variables,
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
