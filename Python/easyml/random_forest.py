"""Functions for random forest analysis.
"""
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from . import replicate
from . import plot
from . import utils


__all__ = ['easy_random_forest']


def easy_random_forest(data, dependent_variable, family='gaussian',
                       sampler=None, preprocessor=None,
                       exclude_variables=None, categorical_variables=None, train_size=0.667,
                       n_samples=1000, n_divisions=1000, n_iterations=10,
                       out_directory='.', random_state=None, progress_bar=True,
                       n_core=1, **kwargs):
    # Make it run in sequential for now
    n_core = 1

    # Set sampler function
    sampler = utils.set_sampler(sampler, family)

    # Set preprocessor function
    preprocessor = utils.set_preprocessor(preprocessor)

    # Set random state
    utils.set_random_state(random_state)

    # Set columns
    column_names = list(data.columns.values)
    column_names = utils.set_column_names(column_names, dependent_variable,
                                          exclude_variables, preprocessor, categorical_variables)

    # Remove variables
    data = utils.remove_variables(data, exclude_variables)

    # Set categorical variables
    categorical_variables = utils.set_categorical_variables(column_names, categorical_variables)

    # Isolate y
    y = utils.isolate_dependent_variable(data, dependent_variable)

    # Isolate X
    X = utils.isolate_independent_variables(data, dependent_variable)

    # Instantiate output
    output = dict()

    # assess family of regression
    if family == 'gaussian':
        # Set gaussian specific functions
        model = RandomForestRegressor(**kwargs)

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
