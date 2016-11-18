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
    url='https://github.com/CCS-Lab/easyML',
    author='OSU CCSL',
    author_email='https://github.com/CCS-Lab/',
    license='MIT',
    keywords='glmnet penalized regression',
    packages=find_packages(exclude=['docs', 'examples', 'scripts', 'tests']),
    test_suite='nose2.collector.collector',
)
