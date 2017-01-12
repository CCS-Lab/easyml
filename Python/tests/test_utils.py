"""Tests for utility functions.
"""
import matplotlib as mpl; mpl.use('TkAgg')
from sklearn.model_selection import train_test_split


from easyml import resample, preprocess, utils


def test_check_args():
    assert utils.check_args() == 1


def test_identify_parallel():
    assert utils.identify_parallel(1) is False
    assert utils.identify_parallel(2) is True


def test_reduce_cores():
    assert utils.reduce_cores(2, 4) == 2
    assert utils.reduce_cores(4, 4) == 4
    assert utils.reduce_cores(8, 4) == 4


def test_set_column_names():
    value = utils.set_column_names(['y', 'a', 'b', 'c'], 'y',
                                   preprocessor=None, exclude_variables=None, categorical_variables=None)
    assert value == ['a', 'b', 'c']
    value = utils.set_column_names(['y', 'a', 'b', 'c'], 'y',
                                   preprocessor=None, exclude_variables=['a'], categorical_variables=None)
    assert value == ['b', 'c']
    value = utils.set_column_names(['y', 'a', 'b', 'c'], 'y',
                                   preprocessor=None, exclude_variables=None, categorical_variables=['c'])
    assert value == ['a', 'b', 'c']
    value = utils.set_column_names(['y', 'a', 'b', 'c'], 'y',
                                   preprocessor=None, exclude_variables=['a'], categorical_variables=['c'])
    assert value == ['b', 'c']
    value = utils.set_column_names(['y', 'a', 'b', 'c'], 'y',
                                   preprocessor=preprocess.preprocess_identity,
                                   exclude_variables=None, categorical_variables=None)
    assert value == ['a', 'b', 'c']
    value = utils.set_column_names(['y', 'a', 'b', 'c'], 'y',
                                   preprocessor=preprocess.preprocess_identity,
                                   exclude_variables=['a'], categorical_variables=None)
    assert value == ['b', 'c']
    value = utils.set_column_names(['y', 'a', 'b', 'c'], 'y',
                                   preprocessor=preprocess.preprocess_identity,
                                   exclude_variables=None, categorical_variables=['c'])
    assert value == ['a', 'b', 'c']
    value = utils.set_column_names(['y', 'a', 'b', 'c'], 'y',
                                   preprocessor=preprocess.preprocess_identity,
                                   exclude_variables=['a'], categorical_variables=['c'])
    assert value == ['b', 'c']
    value = utils.set_column_names(['y', 'a', 'b', 'c'], 'y',
                                   preprocessor=preprocess.preprocess_scaler,
                                   exclude_variables=None, categorical_variables=None)
    assert value == ['a', 'b', 'c']
    value = utils.set_column_names(['y', 'a', 'b', 'c'], 'y',
                                   preprocessor=preprocess.preprocess_scaler,
                                   exclude_variables=['a'], categorical_variables=None)
    assert value == ['b', 'c']
    value = utils.set_column_names(['y', 'a', 'b', 'c'], 'y',
                                   preprocessor=preprocess.preprocess_scaler,
                                   exclude_variables=None, categorical_variables=['c'])
    assert value == ['c', 'a', 'b']
    value = utils.set_column_names(['y', 'a', 'b', 'c'], 'y',
                                   preprocessor=preprocess.preprocess_scaler,
                                   exclude_variables=['a'], categorical_variables=['c'])
    assert value == ['c', 'b']


def test_set_categorical_variables():
    assert utils.set_categorical_variables(['a', 'b', 'c']) is None
    assert utils.set_categorical_variables(['a', 'b', 'c'], ['a']) == [True, False, False]


def test_set_preprocessor():
    assert utils.set_preprocessor(preprocessor=None) is preprocess.preprocess_identity
    assert utils.set_preprocessor(preprocess.preprocess_scaler) is preprocess.preprocess_scaler


def test_set_sampler():
    assert utils.set_sampler(sampler=resample.sample_equal_proportion) is resample.sample_equal_proportion
    assert utils.set_sampler(sampler=None, family="gaussian") is train_test_split
    assert utils.set_sampler(sampler=None, family="binomial") is resample.sample_equal_proportion


def test_isolate_dependent_variable():
    assert 1 == 1


def test_isolate_independent_variables():
    assert 1 == 1


def test_remove_variables():
    assert 1 == 1


def test_set_random_state():
    assert 1 == 1
