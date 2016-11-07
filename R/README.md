<!-- README.md is generated from README.Rmd. Please edit that file -->
easyml
======

[![Project Status: WIP - Initial development is in progress, but there has not yet been a stable, usable release suitable for the public.](http://www.repostatus.org/badges/latest/wip.svg)](http://www.repostatus.org/#wip)

A toolkit for easily building and evaluation machine learning models.

A whitepaper for easyml is available at <http://arxiv.org/abs/TOBEEDITED>, and here's a BibTeX entry that you can use to cite it in a publication::

    @misc{TOBEEDITED,
        Author = {Woo-Young Ahn and Paul Hendricks},
        Title = {easyml: A toolkit for easily building and evaluation machine learning models.},
        Year = {2016},
        Eprint = {arXiv:TOBEEDITED},
    }

Installation
------------

You can install the latest development version from github with:

``` r
if (packageVersion("devtools") < 1.6) {
  install.packages("devtools")
}
devtools::install_github("CCS-Lab/easyml", subdir = "R")
```

If you encounter a clear bug, please file a minimal reproducible example on [github](https://github.com/CCS-Lab/easyml/issues).

Examples
--------

For a dataset with a continuous dependent variable:

``` r
data("prostate", package = "easyml")
output <- easy_glmnet(data = prostate, dependent_variable = "lpsa")
```

For a dataset with a binary dependent variable:

``` r
data("cocaine", package = "easyml")
output <- easy_glmnet(data = cocaine, dependent_variable = "DIAGNOSIS", family = "binomial")
```

References
----------

Ahn, W.-Y.∗, Ramesh∗, D., Moeller, F. G., & Vassileva, J. (2016) Utility of machine learning approaches to identify behavioral markers for substance use disorders: Impulsivity dimensions as predictors of current cocaine dependence. Frontiers in Psychiatry, 7: 34. [PDF](https://u.osu.edu/ccsl/files/2015/08/Ahn2016_Frontiers-26g6nye.pdf) ∗Co-first authors

Ahn, W.-Y. & Vassileva, J. (2016) Machine-learning identifies substance-specific behavioral markers for opiate and stimulant dependence. Drug and Alcohol Dependence, 161 (1), 247–257. [PDF](https://u.osu.edu/ccsl/files/2016/02/Ahn2016_DAD-oftlf3.pdf)

Ahn, W.-Y., Kishida, K. T., Gu, X., Lohrenz, T., Harvey, A. H., Alford, J. R., Smith, K. B., Yaffe, G., Hibbing, J. R., Dayan, P., & Montague, P. R. (2014) Nonpolitical images evoke neural predictors of political ideology. Current Biology, 24(22), 2693-2599. [PDF](https://u.osu.edu/ccsl/files/2015/11/Ahn2014_CB-1l5475k.pdf) [SOM](https://u.osu.edu/ccsl/files/2015/11/Ahn2014_CB_SOM-1xag1ph.pdf)
