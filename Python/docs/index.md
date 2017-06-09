easyml
======

[![Project Status: WIP - Initial development is in progress, but there has not yet been a stable, usable release suitable for the public.](http://www.repostatus.org/badges/latest/wip.svg)](http://www.repostatus.org/#wip)[![DOI](https://zenodo.org/badge/71721801.svg)](https://zenodo.org/badge/latestdoi/71721801)[![Documentation Status](https://readthedocs.org/projects/easyml/badge/?version=master)](http://easyml.readthedocs.io/en/master/?badge=master)[![Build Status](https://travis-ci.org/CCS-Lab/easyml.svg?branch=master)](https://travis-ci.org/CCS-Lab/easyml)

A toolkit for easily building and evaluating machine learning models.

Installation
------------

You can install the latest development version from github with:

```bash
git clone https://github.com/CCS-Lab/easyml.git
cd easyml/Python
pip install .
pip install -r requirements.txt
```

If you encounter a clear bug, please file a [minimal reproducible example](http://stackoverflow.com/questions/5963269/how-to-make-a-great-r-reproducible-example) on [github](https://github.com/CCS-Lab/easyml/issues).

Examples
--------

Load the `easyml` library:

``` python
import pandas as pd
from easyml.glmnet import easy_glmnet
```

For a dataset with a continuous dependent variable:

``` python
# Load data
prostate = pd.read_table('./prostate.txt')

# Analyze data
output = easy_glmnet(prostate, 'lpsa',
            random_state=1, progress_bar=True, n_core=1,
            n_samples=100, n_divisions=10, n_iterations=5,
            model_args={'alpha': 1, 'n_lambda': 200})
```

For a dataset with a binary dependent variable:

``` python
# Load data
cocaine_depedence = pd.read_table('./cocaine_depedence.txt')

# Analyze data
results = easy_glmnet(cocaine_dependence, 'DIAGNOSIS',
                      family='binomial',
                      exclude_variables=['subject'],
                      categorical_variables=['Male'],
                      random_state=12345, progress_bar=True, n_core=1,
                      n_samples=5, n_divisions=5, n_iterations=2,
                      model_args={'alpha': 1, 'n_lambda': 200})
```

Citation
--------

A whitepaper for easyml is available at https://doi.org/10.1101/137240. If you find this code useful please cite us in your work:

```
@article {Hendricks137240,
	author = {Hendricks, Paul and Ahn, Woo-Young},
	title = {Easyml: Easily Build And Evaluate Machine Learning Models},
	year = {2017},
	doi = {10.1101/137240},
	publisher = {Cold Spring Harbor Labs Journals},
	URL = {http://biorxiv.org/content/early/2017/05/12/137240},
	journal = {bioRxiv}
}
```

References
----------
Hendricks, P., & Ahn, W.-Y. (2017). Easyml: Easily Build And Evaluate Machine Learning Models. bioRxiv, 137240. http://doi.org/10.1101/137240
