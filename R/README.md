
<!-- README.md is generated from README.Rmd. Please edit that file -->
easyml
======

[![Project Status: Active - The project has reached a stable, usable state and is being actively developed.](http://www.repostatus.org/badges/latest/active.svg)](http://www.repostatus.org/#active)[![DOI](https://zenodo.org/badge/71721801.svg)](https://zenodo.org/badge/latestdoi/71721801)[![Build Status](https://travis-ci.org/CCS-Lab/easyml.svg?branch=master)](https://travis-ci.org/CCS-Lab/easyml)[![AppVeyor Build Status](https://ci.appveyor.com/api/projects/status/github/CCS-Lab/hBayesDM?branch=master&svg=true)](https://ci.appveyor.com/project/CCS-Lab/hBayesDM)[![codecov](https://codecov.io/gh/CCS-Lab/easyml/branch/master/graph/badge.svg)](https://codecov.io/gh/CCS-Lab/easyml)

A toolkit for easily building and evaluating machine learning models.

R tutorial: <https://ccs-lab.github.io/easyml/>

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

A whitepaper for easyml is available at <https://doi.org/10.1101/137240>. If you find this code useful please cite us in your work:

    @article {Hendricks137240,
        author = {Hendricks, Paul and Ahn, Woo-Young},
        title = {Easyml: Easily Build And Evaluate Machine Learning Models},
        year = {2017},
        doi = {10.1101/137240},
        publisher = {Cold Spring Harbor Labs Journals},
        URL = {http://biorxiv.org/content/early/2017/05/12/137240},
        journal = {bioRxiv}
    }

References
----------

Hendricks, P., & Ahn, W.-Y. (2017). Easyml: Easily Build And Evaluate Machine Learning Models. bioRxiv, 137240. <http://doi.org/10.1101/137240>
