"""Functions for glmnet analysis.
"""
from glmnet import ElasticNet, LogitNet
import numpy as np

from . import replicate
from . import plot
from . import utils


__all__ = ['easy_glmnet']


def glmnet_fit_model(e, X, y):
    return e.fit(X, y)


def glmnet_extract_coefficients_gaussian(e):
    return e.coef_


def glmnet_extract_coefficients_binomial(e):
    return e.coef_[0]


def glmnet_predict_model_gaussian(e, X):
    return e.predict(X)


def glmnet_predict_model_binomial(e, X):
    return e.predict_proba(X)[:, 1]


def easy_glmnet(data, dependent_variable, family='gaussian',
                sampler=None, preprocessor=None,
                exclude_variables=None, categorical_variables=None,
                train_size=0.667, survival_rate_cutoff=0.05,
                n_samples=1000, n_divisions=1000, n_iterations=10,
                random_state=None, progress_bar=True, n_core=1, **kwargs):
    # Set sampler function
    sampler = utils.set_sampler(sampler, family)

    # Set preprocessor function
    preprocessor = utils.set_preprocessor(preprocessor)

    # Set random state
    utils.set_random_state(random_state)

    # Set columns
    column_names = data.columns
    column_names = utils.set_column_names(column_names, dependent_variable,
                                          exclude_variables=None, preprocessor=None, categorical_variables=None)

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
        model = ElasticNet(**kwargs)
        output.update({'model': model})

        # Replicate coefficients
        coefs = replicate.replicate_coefficients(model, glmnet_fit_model, glmnet_extract_coefficients_gaussian,
                                                 preprocessor, X, y, categorical_variables=categorical_variables,
                                                 n_samples=n_samples, progress_bar=progress_bar, n_core=n_core)
        output.update({'coefs': coefs})

        # Process coefficients
        betas = utils.process_coefficients(coefs, column_names, survival_rate_cutoff=survival_rate_cutoff)

        # Write coefficients
        output.update({'betas': betas})

        # Split data
        X_train, X_test, y_train, y_test = sampler(X, y, train_size=train_size)

        # Replicate predictions
        y_train_pred, y_test_pred = replicate.replicate_predictions(model, glmnet_fit_model,
                                                                    glmnet_predict_model_gaussian, preprocessor,
                                                                    X_train, y_train, X_test,
                                                                    categorical_variables=categorical_variables,
                                                                    n_samples=n_samples, progress_bar=progress_bar,
                                                                    n_core=n_core)

        # Take average of predictions for training and test sets
        # Process predictions
        # y_train_pred_mean, y_test_pred_mean = process_predictions(y_train_pred, y_test_pred)
        y_train_pred_mean = np.mean(y_train_pred, axis=0)
        y_test_pred_mean = np.mean(y_test_pred, axis=0)

        # Plot the gaussian predictions for training
        plot.plot_gaussian_predictions(y_train, y_train_pred_mean)
        # plt.savefig(path.join(out_directory, 'train_predictions.png'))

        # Plot the gaussian predictions for test
        plot.plot_gaussian_predictions(y_test, y_test_pred_mean)
        # plt.savefig(path.join(out_directory, 'test_predictions.png'))

        # Replicate training and test MSEs
        train_mses, test_mses = replicate.replicate_mses(model, sampler, glmnet_fit_model,
                                                         glmnet_predict_model_gaussian, preprocessor, X, y,
                                                         categorical_variables=categorical_variables,
                                                         n_divisions=n_divisions, n_iterations=n_iterations,
                                                         progress_bar=progress_bar, n_core=n_core)

        # Plot histogram of training MSEs
        plot.plot_mse_histogram(train_mses)
        # plt.savefig(path.join(out_directory, 'train_mse_distribution.png'))

        # Plot histogram of test MSEs
        plot.plot_mse_histogram(test_mses)
        # plt.savefig(path.join(out_directory, 'test_mse_distribution.png'))

    elif family == 'binomial':
        # Set binomial specific functions
        model = LogitNet(**kwargs)

        # Replicate coefficients
        coefs = replicate.replicate_coefficients(model, glmnet_fit_model, glmnet_extract_coefficients_binomial,
                                                 preprocessor, X, y, categorical_variables=categorical_variables,
                                                 n_samples=n_samples, progress_bar=progress_bar, n_core=n_core)

        # Process coefficients
        # betas = utils.process_coefficients(coefs, column_names, survival_rate_cutoff=survival_rate_cutoff)

        # Split data
        X_train, X_test, y_train, y_test = sampler(X, y, train_size=train_size)

        # Replicate predictions
        y_train_pred, y_test_pred = replicate.replicate_predictions(model, glmnet_fit_model,
                                                                    glmnet_predict_model_binomial, preprocessor,
                                                                    X_train, y_train, X_test,
                                                                    categorical_variables=categorical_variables,
                                                                    n_samples=n_samples, progress_bar=progress_bar,
                                                                    n_core=n_core)

        # Take average of predictions for training and test sets
        y_train_pred_mean = np.mean(y_train_pred, axis=0)
        y_test_pred_mean = np.mean(y_test_pred, axis=0)

        # Plot the ROC curve for training
        plot.plot_roc_curve(y_train, y_train_pred_mean)

        # Plot the ROC curve for test
        plot.plot_roc_curve(y_test, y_test_pred_mean)

        # Replicate training and test AUCSs
        train_aucs, test_aucs = replicate.replicate_aucs(model, sampler, glmnet_fit_model,
                                                         glmnet_predict_model_binomial, preprocessor, X, y,
                                                         categorical_variables=categorical_variables,
                                                         n_divisions=n_divisions, n_iterations=n_iterations,
                                                         progress_bar=progress_bar, n_core=n_core)

        # Plot histogram of training AUCs
        plot.plot_auc_histogram(train_aucs)

        # Plot histogram of test AUCs
        plot.plot_auc_histogram(test_aucs)
    else:
        raise ValueError

    return output
