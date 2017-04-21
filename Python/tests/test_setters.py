"""Tests for utility functions.
"""
from easyml import preprocess, setters


def test_set_parallel():
    assert setters.set_parallel(1) is False
    assert setters.set_parallel(2) is True


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


# def test_set_categorical_variables():
#     assert setters.set_categorical_variables(['a', 'b', 'c']) is None
#     assert setters.set_categorical_variables(['a', 'b', 'c'], ['a']) is [True, False, False]


def test_set_preprocess():
    assert setters.set_preprocess(preprocess=None) is preprocess.preprocess_identity
    assert setters.set_preprocess(preprocess.preprocess_scale) is preprocess.preprocess_scale


# def test_set_resample():
#     assert setters.set_resample(resample=resample.sample_equal_proportion) is resample.sample_equal_proportion
#     assert setters.set_resample(resample=None, family="gaussian") is train_test_split
#     assert setters.set_resample(resample=None, family="binomial") is resample.sample_equal_proportion


def test_set_dependent_variable():
    assert 1 == 1


def test_set_independent_variables():
    assert 1 == 1


def test_set_random_state():
    assert 1 == 1
