"""Tests for utility functions.
"""
from sklearn.model_selection import train_test_split


from easyml import resample, preprocess, utils, setters


# def test_check_args():
#     assert utils.check_args() == 1


def test_identify_parallel():
    assert setters.set_parallel(1) is False
    assert setters.set_parallel(2) is True


def test_reduce_cores():
    assert utils.reduce_cores(2, 4) == 2
    assert utils.reduce_cores(4, 4) == 4
    assert utils.reduce_cores(8, 4) == 4


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
                                   preprocess=preprocess.preprocess_scaler,
                                   exclude_variables=None, categorical_variables=None)
    assert value == ['a', 'b', 'c']
    value = setters.set_column_names(['y', 'a', 'b', 'c'], 'y',
                                   preprocess=preprocess.preprocess_scaler,
                                   exclude_variables=['a'], categorical_variables=None)
    assert value == ['b', 'c']
    value = setters.set_column_names(['y', 'a', 'b', 'c'], 'y',
                                   preprocess=preprocess.preprocess_scaler,
                                   exclude_variables=None, categorical_variables=['c'])
    assert value == ['c', 'a', 'b']
    value = setters.set_column_names(['y', 'a', 'b', 'c'], 'y',
                                   preprocess=preprocess.preprocess_scaler,
                                   exclude_variables=['a'], categorical_variables=['c'])
    assert value == ['c', 'b']


def test_set_categorical_variables():
    assert setters.set_categorical_variables(['a', 'b', 'c']) is None
    assert setters.set_categorical_variables(['a', 'b', 'c'], ['a']) == [True, False, False]


def test_set_preprocess():
    assert setters.set_preprocess(preprocess=None) is preprocess.preprocess_identity
    assert setters.set_preprocess(preprocess.preprocess_scaler) is preprocess.preprocess_scaler


# def test_set_resample():
#     assert setters.set_resample(resample=resample.sample_equal_proportion) is resample.sample_equal_proportion
#     assert setters.set_resample(resample=None, family="gaussian") is train_test_split
#     assert setters.set_resample(resample=None, family="binomial") is resample.sample_equal_proportion


def test_isolate_dependent_variable():
    assert 1 == 1


def test_isolate_independent_variables():
    assert 1 == 1


def test_remove_variables():
    assert 1 == 1


def test_set_random_state():
    assert 1 == 1
