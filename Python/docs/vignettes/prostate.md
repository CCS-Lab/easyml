
# Prostate Cancer

Paul Hendricks
2017-06-25

## Overview

In this vignette, we demonstrate the power of easyml using a Prostate Cancer dataset.

## Load the data

First we load the easymlpy package and the Prostate Cancer dataset.



```python
from easymlpy.datasets import load_prostate
from easymlpy import support_vector_machine

%matplotlib inline
```


```python
prostate = load_prostate()
print(prostate.head())
```

         lcavol   lweight  age      lbph  svi       lcp  gleason  pgg45      lpsa
    0 -0.579818  2.769459   50 -1.386294    0 -1.386294        6      0 -0.430783
    1 -0.994252  3.319626   58 -1.386294    0 -1.386294        6      0 -0.162519
    2 -0.510826  2.691243   74 -1.386294    0 -1.386294        7     20 -0.162519
    3 -1.203973  3.282789   58 -1.386294    0 -1.386294        6      0 -0.162519
    4  0.751416  3.432373   62 -1.386294    0 -1.386294        6      0  0.371564


## Train a supprt vector machine model

To run an `easy_support_vector_machine` model, we pass in the following parameters:

* the data set `prostate`,
* the name of the dependent variable e.g. `lpsa`,
* whether to run a gaussian or a binomial model,
* which variables to exclude from the analysis,
* which variables are categorical variables; these variables are not scaled, if `preprocess_scale` is used,
* the random state,
* whether to display a progress bar,
* how many cores to run the analysis on in parallel.


```python
# Analyze data
results = support_vector_machine.easy_support_vector_machine(prostate, 'lpsa',
                                                             n_samples=10, n_divisions=10, 
                                                             n_iterations=10, progress_bar=False, 
                                                             random_state=12345, n_core=1)
```

    Generating predictions for a single train test split:
    Generating measures of model performance over multiple train test splits:


## Assess results

Now let’s assess the results of the `easy_support_vector_machine` model.


```python

```
