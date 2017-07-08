"""
Functions for setting certain functions and parameters.
"""
import numpy as np

from . import measure as meas
from . import preprocess as prep
from . import plot as plt
from . import resample as res


__all__ = ['set_random_state', 'set_parallel', 'set_resample',
           'set_categorical_variables', 'set_column_names',
           'set_dependent_variable', 'set_independent_variables',
           'set_measure', 'set_plot_model_performance',
           'set_plot_predictions', 'set_preprocess']


def set_random_state(random_state=None):
    """
    Set random state.

    Sets the random state to a specific seed. Please note this function affects global state.

    :param random_state: An integer; specifies the seed to be used for the analysis. Defaults to None.
    :return: None.
    """
    if random_state:
        np.random.seed(random_state)
    return None


def set_parallel(n_core):
    """
    Set parallel.

    This helper function decides whether the analysis should be run in parallel based on the number of cores specified.

    :param n_core: An integer; specifies the number of cores to use for this analysis.
    :return: A boolean; whether analysis should be run in parallel or not.
    """
    n_core = int(n_core)
    if n_core == 1:
        parallel = False
    elif n_core > 1:
        parallel = True
    else:
        raise ValueError
    return parallel


def set_resample(resample=None, family=None):
    """
    Set resample function.
    
    Sets the function responsible for resampling the data.
    
    :param resample: A function; the function for resampling the data. Defaults to None.
    :param family: A string; the type of regression to run on the data. Choices are either 'gaussian' or 'binomial'.
    :return: A function; the function for resampling the data.
    """
    if not resample:
        if family == 'gaussian':
            resample = res.resample_simple_train_test_split
        elif family == 'binomial':
            resample = res.resample_stratified_class_train_test_split
        else:
            raise ValueError
    return resample


def set_preprocess(preprocess=None):
    """
    Set preprocess function.
    
    Sets the function responsible for preprocessing the data.
    
    :param preprocess: A function; the function for preprocessing the data. Defaults to None.
    :return: A function; the function for preprocessing the data.
    """
    if not preprocess:
        preprocess = prep.preprocess_identity
    return preprocess


def set_measure(measure=None, family=None):
    """
    Set measure function.
    
    Sets the function responsible for measuring the results.
    
    :param measure: A function; the function for measuring the results. Defaults to None.
    :param family: A string; the type of regression to run on the data. Choices are either 'gaussian' or 'binomial'.
    :return: A function; the function for measuring the results.
    """
    if not measure:
        if family == 'gaussian':
            measure = meas.measure_cor_score
        elif family == 'binomial':
            measure = meas.measure_area_under_curve
        else:
            raise ValueError
    return measure


def set_column_names(column_names, dependent_variable,
                     exclude_variables=None, preprocess=None,
                     categorical_variables=None):
    """
    Set column names.

    This functions helps decide what the updated column names of a data.frame should be within
    the easyml framework based on the dependent variable, preprocessing function,
    exclusionary variables, and categorical variables.
    
    :param column_names: A list of strings; the column names of the data for this analysis.
    :param dependent_variable: A string; the dependent variable for this analysis.
    :param preprocess: A function; the function for preprocessing the data. Defaults to None.
    :param exclude_variables: A list of strings; the variables from the data set to exclude. Defaults to None.
    :param categorical_variables: A list of strings; the variables that are categorical. Defaults to None.
    :return: The updated columns, in the correct order for preprocessing.
    """
    column_names = [c for c in column_names if c != dependent_variable]
    if exclude_variables:
        column_names = [c for c in column_names if c not in exclude_variables]
    if categorical_variables and preprocess is prep.preprocess_scale:
        column_names = [c for c in column_names if c not in categorical_variables]
        column_names = categorical_variables + column_names
    return column_names


def set_categorical_variables(column_names, categorical_variables=None):
    """
    Set categorical variables.
    
    This helper functions determines a logical boolean vector based on the column names
    and the designation for which ones are categorical variables.
    
    :param column_names: A list of strings; the column names of the data for this analysis.
    :param categorical_variables: A list of strings; the variables that are categorical. Defaults to None.
    :return: None, or if `categorical_variables` is not None, then a list of booleans of length len(column_names) where True represents that column is a categorical variable.
    """
    if categorical_variables:
        categorical_variables = np.in1d(column_names, categorical_variables)
    return categorical_variables


def set_dependent_variable(data, dependent_variable):
    """
    Set dependent variable.
    
    This helper functions isolates the dependent variable in a data.frame.
    
    :param data: An object of class pandas.DataFrame; the data to be analyzed.
    :param dependent_variable: A string; the dependent variable for this analysis.
    :return: An ndarray, the dependent variable of the analysis.
    """
    y = data[dependent_variable].values
    return y


def set_independent_variables(data, dependent_variable):
    """
    Set independent variables.
    
    This helper functions isolates the independent variables in a data.frame.
    
    :param data: An object of class pandas.DataFrame; the data to be analyzed.
    :param dependent_variable: A string; the dependent variable for this analysis.
    :return: An object of class pandas.DataFrame; the independent variables of the analysis.
    """
    X = data.drop(dependent_variable, axis=1).values
    return X


def set_plot_predictions(family=None):
    """
    Set plot predictions function.
    
    Sets the function responsible for plotting the predictions generated from a fitted model.
    
    :param family: A string; the type of regression to run on the data. Choices are either 'gaussian' or 'binomial'.
    :return: A function; the function for plotting the predictions generated from a fitted model.
    """
    if family == 'gaussian':
        plot_predictions = plt.plot_predictions_gaussian
    elif family == 'binomial':
        plot_predictions = plt.plot_predictions_binomial
    else:
        raise ValueError
    return plot_predictions


def set_plot_model_performance(measure):
    """
    Set plot model performance function.
    
    Sets the function responsible for plotting the measures of model performance generated from the predictions generated from a fitted model.
    
    :param measure: A function; the function for measuring the results. Defaults to None.
    :return: A function; the function for plotting the measures of model performance generated from the predictions generated from a fitted model.
    """
    if measure == meas.measure_mean_squared_error:
        plot_model_performance = plt.plot_model_performance_gaussian_mean_squared_error
    elif measure == meas.measure_cor_score:
        plot_model_performance = plt.plot_model_performance_gaussian_cor_score
    elif measure == meas.measure_r2_score:
        plot_model_performance = plt.plot_model_performance_gaussian_r2_score
    elif measure == meas.measure_area_under_curve:
        plot_model_performance = plt.plot_model_performance_binomial_area_under_curve
    else:
        raise ValueError
    return plot_model_performance
