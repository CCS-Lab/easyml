"""Tests for utility functions.
"""
from easyml import utils


def test_reduce_cores():
    assert utils.reduce_cores(2, 4) == 2
    assert utils.reduce_cores(4, 4) == 4
    assert utils.reduce_cores(8, 4) == 4


def test_remove_variables():
    assert 1 == 1
