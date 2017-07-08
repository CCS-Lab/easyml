"""
Helper functions for datasets.
"""
import io
import pandas as pd
import requests


__all__ = ['load_cocaine_dependence', 'load_prostate']


def load_cocaine_dependence():
    """
    Loads the cocaine dependence dataset.

    :return: An object of class pandas.DataFrame.
    """
    url = 'https://raw.githubusercontent.com/CCS-Lab/easyml/master/Python/datasets/cocaine_dependence.csv'
    s = requests.get(url).content
    dataset = pd.read_csv(io.StringIO(s.decode('utf-8')))
    return dataset


def load_prostate():
    """
    Loads the prostate cancer dataset.

    :return: An object of class pandas.DataFrame.
    """
    url = 'https://raw.githubusercontent.com/CCS-Lab/easyml/master/Python/datasets/prostate.csv'
    s = requests.get(url).content
    dataset = pd.read_csv(io.StringIO(s.decode('utf-8')))
    return dataset
