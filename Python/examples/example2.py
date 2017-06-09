from glmnet import ElasticNet
import io
import numpy as np
import pandas as pd
import requests
from sklearn.preprocessing import StandardScaler


# Load data
url = 'https://raw.githubusercontent.com/CCS-Lab/easyml/master/Python/datasets/prostate.csv'
s = requests.get(url).content
prostate = pd.read_csv(io.StringIO(s.decode('utf-8')))

# Generate coefficients from data by hand
X, y = prostate.drop('lpsa', axis=1).values, prostate['lpsa'].values
sclr = StandardScaler()
X_preprocessed = sclr.fit_transform(X)

# no random state
coefficients = []
for i in range(10):
    model = ElasticNet(alpha=1, standardize=False, cut_point=0.0, n_lambda=200)
    print(id(model))
    model.fit(X_preprocessed, y)
    coefficients.append(np.asarray(model.coef_))
print(coefficients)

# seed set at outer level
np.random.seed(43210)
coefficients = []
for i in range(10):
    model = ElasticNet(alpha=1, standardize=False, cut_point=0.0, n_lambda=200)
    print(id(model))
    model.fit(X_preprocessed, y)
    coefficients.append(np.asarray(model.coef_))
print(coefficients)

# seed set at inner level
coefficients = []
for i in range(10):
    np.random.seed(43210)
    model = ElasticNet(alpha=1, standardize=False, cut_point=0.0, n_lambda=200)
    print(id(model))
    model.fit(X_preprocessed, y)
    coefficients.append(np.asarray(model.coef_))
print(coefficients)

# seed set at function level
coefficients = []
for i in range(10):
    random_state = np.random.RandomState(i)
    model = ElasticNet(alpha=1, standardize=False, cut_point=0.0, n_lambda=200, random_state=random_state)
    print(id(model))
    model.fit(X_preprocessed, y)
    coefficients.append(np.asarray(model.coef_))
print(coefficients)

coefficients = []
random_state = np.random.RandomState(43210)
for i in range(10):
    model = ElasticNet(alpha=1, standardize=False, cut_point=0.0, n_lambda=200, random_state=random_state)
    print(id(model))
    model.fit(X_preprocessed, y)
    coefficients.append(np.asarray(model.coef_))
print(coefficients)
