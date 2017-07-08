from setuptools import setup

setup(
    name='easymlpy',
    version='0.1.2',
    description='A Python toolkit for easily building and evaluating machine learning models.',
    url='https://github.com/CCS-Lab/easyml',
    author='OSU CCSL',
    author_email='paul.hendricks.2013@owu.edu',
    license='MIT',
    keywords='easy machine learning glmnet penalized regression random forest penalized regression',
    packages=['easymlpy'],
    test_suite='nose2.collector.collector',
)
