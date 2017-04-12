"""Utility functions.
"""
import os


__all__ = []


def reduce_cores(n_core, cpu_count=None):
    """    
    Reduces cores.
    
    :param n_core: foo
    :param cpu_count: bar
    :return: number
    """
    if cpu_count is None:
        cpu_count = os.cpu_count()
    n_core = min(n_core, cpu_count)
    return n_core


def remove_variables(data, exclude_variables=None):
    if exclude_variables is not None:
        data = data.drop(exclude_variables, axis=1, inplace=False)
    return data
