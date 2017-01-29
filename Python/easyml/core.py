"""Functions for basic analysis.
"""
import numpy as np

from . import replicate
from . import setters
from . import utils


__all__ = ['easy_analysis']


def easy_analysis(data, dependent_variable,
                  algorithm=None, family='gaussian',
                  resample=None, preprocess=None, measure=None,
                  exclude_variables=None, categorical_variables=None,
                  train_size=0.667, survival_rate_cutoff=0.05,
                  n_samples=1000, n_divisions=1000, n_iterations=10,
                  random_state=None, progress_bar=True, n_core=1, **kwargs):
    # Instantiate output
    output = dict()

    # Set random state
    setters.set_random_state(random_state)

    # Set coefficients boolean
    coefficients_boolean = setters.set_coefficients_boolean(algorithm)

    # Set predictions boolean
    predictions_boolean = setters.set_predictions_boolean(algorithm)

    # Set metrics boolean
    metrics_boolean = setters.set_metrics_boolean(algorithm)

    # Set resample function
    resample = setters.set_resample(resample, family)
    output.update({'resample': resample})

    # Set preprocess function
    preprocess = setters.set_preprocess(preprocess)
    output.update({'preprocess': preprocess})

    # Set measure function
    measure = setters.set_measure(measure, algorithm, family)
    output.update({'measure': measure})

    # Set fit_model function
    fit_model = setters.set_fit_model(algorithm, family)
    output.update({'fit_model': fit_model})

    # Set extract_coefficients function
    extract_coefficients = setters.set_extract_coefficients(algorithm, family)
    output.update({'extract_coefficients': extract_coefficients})

    # Set predict_model function
    predict_model = setters.set_predict_model(algorithm, family)
    output.update({'predict_model': predict_model})

    # Set plot_predictions function
    plot_predictions = setters.set_plot_predictions(algorithm, family)
    output.update({'plot_predictions': plot_predictions})

    # Set plot_metrics function
    plot_metrics = setters.set_plot_metrics(measure)
    output.update({'plot_metrics': plot_metrics})

    # Set column names
    column_names = list(data.columns.values)
    column_names = setters.set_column_names(column_names, dependent_variable,
                                            exclude_variables, preprocess, categorical_variables)

    # Remove variables
    data = utils.remove_variables(data, exclude_variables)

    # Set categorical variables
    categorical_variables = setters.set_categorical_variables(column_names, categorical_variables)

    # Set dependent variable
    y = setters.set_dependent_variable(data, dependent_variable)
    output.update({'y': y})

    # Set independent variables
    X = setters.set_independent_variables(data, dependent_variable)
    output.update({'X': X})

    # Resample data
    X_train, X_test, y_train, y_test = resample(X, y, train_size=train_size)
    output.update({'X_train': X_train, 'X_test': X_test, 'y_train': y_train, 'y_test': y_test})

    # Assess if coefficients should be replicated for this algorithm
    if coefficients_boolean:
        # Replicate coefficients
        coefficients = replicate.replicate_coefficients(fit_model, extract_coefficients,
                                                        preprocess, X, y, categorical_variables=categorical_variables,
                                                        n_samples=n_samples, progress_bar=progress_bar, n_core=n_core,
                                                        **kwargs)
        output.update({'coefficients': coefficients})

        # Process coefficients
        coefficients_processed = utils.process_coefficients(coefficients, column_names,
                                                            survival_rate_cutoff=survival_rate_cutoff)
        output.update({'coefficients_processed': coefficients_processed})

    # Assess if predictions should be replicated for this algorithm
    if predictions_boolean:
        # Replicate predictions
        predictions = replicate.replicate_predictions(fit_model, predict_model, preprocess,
                                                      X_train, y_train, X_test,
                                                      categorical_variables=categorical_variables,
                                                      n_samples=n_samples,
                                                      progress_bar=progress_bar,
                                                      n_core=n_core, **kwargs)
        predictions_train, predictions_test = predictions
        output.update({'predictions_train': predictions_train, 'predictions_test': predictions_test})

        # Process predictions
        predictions_train_mean = np.mean(predictions_train, axis=0)
        predictions_test_mean = np.mean(predictions_test, axis=0)
        output.update({'predictions_train_mean': predictions_train_mean, 'predictions_test_mean': predictions_test_mean})

        # Save predictions plots
        output.update({'plot_predictions_train': plot_predictions(y_train, predictions_train_mean)})
        output.update({'plot_predictions_test': plot_predictions(y_test, predictions_test_mean)})

    # Assess if metrics should be replicated for this algorithm
    if metrics_boolean:
        # Replicate metrics
        metrics = replicate.replicate_metrics(fit_model, predict_model, resample, preprocess, X, y,
                                              categorical_variables=categorical_variables,
                                              n_divisions=n_divisions, n_iterations=n_iterations,
                                              progress_bar=progress_bar, n_core=n_core, **kwargs)
        metrics_train_mean, metrics_test_mean = metrics
        output.update({'metrics_train_mean': metrics_train_mean, 'metrics_test_mean': metrics_test_mean})

        # Save metrics plots
        output.update({'plot_metrics_train_mean': plot_metrics(metrics_train_mean)})
        output.update({'plot_metrics_test_mean': plot_metrics(metrics_test_mean)})

    return output
