"""Tests for utility functions.
"""
import numpy as np
import pandas as pd

from easyml import measure, plot, preprocess, resample, setters


def test_set_random_state():
    assert setters.set_random_state() is None


def test_set_parallel():
    assert setters.set_parallel(1) is False
    assert setters.set_parallel(2) is True


def test_set_resample():
    assert setters.set_resample(resample.resample_simple_train_test_split) is resample.resample_simple_train_test_split


def test_set_preprocess():
    assert setters.set_preprocess(preprocess=None) is preprocess.preprocess_identity
    assert setters.set_preprocess(preprocess.preprocess_scale) is preprocess.preprocess_scale


def test_set_measure():
    assert setters.set_measure(measure.measure_mean_squared_error) is measure.measure_mean_squared_error


def test_set_column_names():
    value = setters.set_column_names(['y', 'a', 'b', 'c'], 'y',
                                     preprocess=None, exclude_variables=None, categorical_variables=None)
    assert value == ['a', 'b', 'c']
    value = setters.set_column_names(['y', 'a', 'b', 'c'], 'y',
                                   preprocess=None, exclude_variables=['a'], categorical_variables=None)
    assert value == ['b', 'c']
    value = setters.set_column_names(['y', 'a', 'b', 'c'], 'y',
                                   preprocess=None, exclude_variables=None, categorical_variables=['c'])
    assert value == ['a', 'b', 'c']
    value = setters.set_column_names(['y', 'a', 'b', 'c'], 'y',
                                   preprocess=None, exclude_variables=['a'], categorical_variables=['c'])
    assert value == ['b', 'c']
    value = setters.set_column_names(['y', 'a', 'b', 'c'], 'y',
                                   preprocess=preprocess.preprocess_identity,
                                   exclude_variables=None, categorical_variables=None)
    assert value == ['a', 'b', 'c']
    value = setters.set_column_names(['y', 'a', 'b', 'c'], 'y',
                                   preprocess=preprocess.preprocess_identity,
                                   exclude_variables=['a'], categorical_variables=None)
    assert value == ['b', 'c']
    value = setters.set_column_names(['y', 'a', 'b', 'c'], 'y',
                                   preprocess=preprocess.preprocess_identity,
                                   exclude_variables=None, categorical_variables=['c'])
    assert value == ['a', 'b', 'c']
    value = setters.set_column_names(['y', 'a', 'b', 'c'], 'y',
                                   preprocess=preprocess.preprocess_identity,
                                   exclude_variables=['a'], categorical_variables=['c'])
    assert value == ['b', 'c']
    value = setters.set_column_names(['y', 'a', 'b', 'c'], 'y',
                                   preprocess=preprocess.preprocess_scale,
                                   exclude_variables=None, categorical_variables=None)
    assert value == ['a', 'b', 'c']
    value = setters.set_column_names(['y', 'a', 'b', 'c'], 'y',
                                   preprocess=preprocess.preprocess_scale,
                                   exclude_variables=['a'], categorical_variables=None)
    assert value == ['b', 'c']
    value = setters.set_column_names(['y', 'a', 'b', 'c'], 'y',
                                   preprocess=preprocess.preprocess_scale,
                                   exclude_variables=None, categorical_variables=['c'])
    assert value == ['c', 'a', 'b']
    value = setters.set_column_names(['y', 'a', 'b', 'c'], 'y',
                                   preprocess=preprocess.preprocess_scale,
                                   exclude_variables=['a'], categorical_variables=['c'])
    assert value == ['c', 'b']


def test_set_categorical_variables():
    assert setters.set_categorical_variables(['a', 'b', 'c']) is None
    assert all(setters.set_categorical_variables(['a', 'b', 'c'], ['a']) == np.array([True, False, False]))

df = pd.DataFrame({'a': [0, 1, 2], 'b': [3, 4, 5], 'c': [6, 7, 8]})
def test_set_dependent_variable():
    assert all(setters.set_dependent_variable(df, 'a') == df['a'])


def test_set_independent_variables():
    assert all(setters.set_independent_variables(df, 'a') == df.drop('a', axis=1))


def test_set_plot_predictions():
    assert setters.set_plot_predictions('gaussian') == plot.plot_predictions_gaussian
    assert setters.set_plot_predictions('binomial') == plot.plot_predictions_binomial


def test_set_plot_metrics():
    assert setters.set_plot_metrics(measure.mean_squared_error) == plot.plot_metrics_gaussian_mean_squared_error
    assert setters.set_plot_metrics(measure.measure_cor_score) == plot.plot_metrics_gaussian_cor_score
    assert setters.set_plot_metrics(measure.measure_r2_score) == plot.plot_metrics_gaussian_r2_score
    assert setters.set_plot_metrics(measure.measure_area_under_curve) == plot.plot_metrics_binomial_area_under_curve
