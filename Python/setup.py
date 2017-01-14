from setuptools import setup, find_packages
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='easyml',
    version='0.1.0',
    description='A Python toolkit for easily building and evaluating penalized regression models.',
    long_description=long_description,
    url='https://github.com/CCS-Lab/easyml',
    author='OSU CCSL',
    author_email='https://github.com/CCS-Lab/',
    license='MIT',
    keywords='easy machine learning glmnet penalized regression random forest penalized regression',
    packages=['easyml'],
    test_suite='nose2.collector.collector',
)
