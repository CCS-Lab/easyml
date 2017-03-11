<!-- README.md is generated from README.Rmd. Please edit that file -->
easyml
======

[![Project Status: Active - The project has reached a stable, usable state and is being actively developed.](http://www.repostatus.org/badges/latest/active.svg)](http://www.repostatus.org/#active)[![DOI](https://zenodo.org/badge/71721801.svg)](https://zenodo.org/badge/latestdoi/71721801)[![Documentation Status](https://img.shields.io/badge/docs-latest-brightgreen.svg)](https://ccs-lab.github.io/easyml/)[![Build Status](https://travis-ci.org/CCS-Lab/easyml.svg?branch=master)](https://travis-ci.org/CCS-Lab/easyml)[![AppVeyor Build Status](https://ci.appveyor.com/api/projects/status/github/CCS-Lab/easyml?branch=master&svg=true)](https://ci.appveyor.com/project/CCS-Lab/easyml)[![codecov](https://codecov.io/gh/CCS-Lab/easyml/branch/master/graph/badge.svg)](https://codecov.io/gh/CCS-Lab/easyml)

A toolkit for easily building and evaluating machine learning models.

Installation
------------

You can install the latest development version from github with:

``` r
if (packageVersion("devtools") < 1.6) {
  install.packages("devtools")
}
devtools::install_github("CCS-Lab/easyml", subdir = "R")
```

If you encounter a clear bug, please file a [minimal reproducible example](http://stackoverflow.com/questions/5963269/how-to-make-a-great-r-reproducible-example) on [github](https://github.com/CCS-Lab/easyml/issues).

Examples
--------

Load the `easyml` library:

``` r
library(easyml)
```

For a dataset with a continuous dependent variable:

``` r
data("prostate", package = "easyml")
results <- easy_glmnet(prostate, "lpsa")
```

For a dataset with a binary dependent variable:

``` r
data("cocaine_dependence", package = "easyml")
results <- easy_glmnet(cocaine_dependence, "diagnosis", 
                       family = "binomial", exclude_variables = c("subject", "age"), 
                       categorical_variables = c("male"))
```

Citation
--------

A whitepaper for easyml is available at <http://arxiv.org/abs/TOBEEDITED>. If you find this code useful please cite us in your work:

    @inproceedings{TOBEEDITED,
        title = {easyml: A toolkit for easily building and evaluating machine learning models},
        author = {Paul Hendricks and Woo-Young Ahn},
        eprint = {arXiv:TOBEEDITED},
        year = {2017},
    }

References
----------

Ahn, W.-Y.∗, Ramesh∗, D., Moeller, F. G., & Vassileva, J. (2016) Utility of machine learning approaches to identify behavioral markers for substance use disorders: Impulsivity dimensions as predictors of current cocaine dependence. Frontiers in Psychiatry, 7: 34. [PDF](https://u.osu.edu/ccsl/files/2015/08/Ahn2016_Frontiers-26g6nye.pdf) ∗Co-first authors

Ahn, W.-Y. & Vassileva, J. (2016) Machine-learning identifies substance-specific behavioral markers for opiate and stimulant dependence. Drug and Alcohol Dependence, 161 (1), 247–257. [PDF](https://u.osu.edu/ccsl/files/2016/02/Ahn2016_DAD-oftlf3.pdf)

Ahn, W.-Y., Kishida, K. T., Gu, X., Lohrenz, T., Harvey, A. H., Alford, J. R., Smith, K. B., Yaffe, G., Hibbing, J. R., Dayan, P., & Montague, P. R. (2014) Nonpolitical images evoke neural predictors of political ideology. Current Biology, 24(22), 2693-2599. [PDF](https://u.osu.edu/ccsl/files/2015/11/Ahn2014_CB-1l5475k.pdf) [SOM](https://u.osu.edu/ccsl/files/2015/11/Ahn2014_CB_SOM-1xag1ph.pdf)
