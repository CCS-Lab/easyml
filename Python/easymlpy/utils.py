"""
Utility functions.
"""
import os


__all__ = ['reduce_cores', 'remove_variables']


def reduce_cores(n_core, cpu_count=None):
    """    
    Reduces cores.
    
    If the number of cores exceeds the number of cores on the OS
    then n_core is reduced to the number of cores on the OS.
    
    :param n_core: integer The number of cores to use for the analysis.
    :param cpu_count: integer, None The number of CPUs available on the machine. Defaults to 
    os.cpu_count() if None.
    :return: number of cores.
    """
    if cpu_count is None:
        cpu_count = os.cpu_count()
    n_core = min(n_core, cpu_count)
    return n_core


def remove_variables(data, exclude_variables=None):
    """    
    Removes variables from the data set.

    If passed a list of variable names to exclude, remove_variables
    will drop those variables from the dataset.

    :param data: A pandas.DataFrame.
    :param exclude_variables: A list of strings.
    :return: A pandas.DataFrame.
    """
    if exclude_variables is not None:
        data = data.drop(exclude_variables, axis=1)
    return data
