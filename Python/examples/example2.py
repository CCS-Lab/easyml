from glmnet import ElasticNet
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


# Load data
prostate = pd.read_table('./Python/examples/prostate.txt')

# Generate coefficients from data by hand
X, y = prostate.drop('lpsa', axis=1).values, prostate['lpsa'].values
sclr = StandardScaler()
X_preprocessed = sclr.fit_transform(X)

# seed set at outer level
np.random.seed(43210)
coefficients = []
for i in range(10):
    print(id(model))
    model = ElasticNet(alpha=1, standardize=False, cut_point=0.0, n_lambda=200)
    model.fit(X_preprocessed, y)
    coefficients.append(np.asarray(model.coef_))
print(coefficients)

# seed set at inner level
coefficients = []
for i in range(10):
    np.random.seed(43210)
    print(id(model))
    model = ElasticNet(alpha=1, standardize=False, cut_point=0.0, n_lambda=200)
    model.fit(X_preprocessed, y)
    coefficients.append(np.asarray(model.coef_))
print(coefficients)

# seed set at function level
coefficients = []
for i in range(10):
    print(id(model))
    random_state = np.random.RandomState(i)
    model = ElasticNet(alpha=1, standardize=False, cut_point=0.0, n_lambda=200, random_state=random_state)
    model.fit(X_preprocessed, y)
    coefficients.append(np.asarray(model.coef_))
print(coefficients)

coefficients = []
random_state = np.random.RandomState(43210)
for i in range(10):
    print(id(model))
    model = ElasticNet(alpha=1, standardize=False, cut_point=0.0, n_lambda=200, random_state=random_state)
    model.fit(X_preprocessed, y)
    coefficients.append(np.asarray(model.coef_))
print(coefficients)
