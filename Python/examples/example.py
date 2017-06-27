from easymlpy import glmnet
from glmnet import ElasticNet
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


# Load data
prostate = pd.read_table('./Python/examples/prostate.txt')

# Generate coefficients from data using easy_glmnet
output = glmnet.easy_glmnet(prostate, 'lpsa',
                            random_state=1, progress_bar=True, n_core=1,
                            n_samples=10, n_divisions=10, n_iterations=2,
                            model_args={'alpha': 1})
print(output.coefficients)

# Generate coefficients from data by hand
X, y = prostate.drop('lpsa', axis=1).values, prostate['lpsa'].values
sclr = StandardScaler()
X_preprocessed = sclr.fit_transform(X)

# data is the same in both - check
assert np.all(X_preprocessed == output.X_preprocessed)

coefficients = []
for i in range(10):
    model = ElasticNet(alpha=1, standardize=False, cut_point=0.0, n_lambda=200)
    model.fit(X_preprocessed, y)
    coefficients.append(np.asarray(model.coef_))

print(coefficients)
# coefficients are the same in both - check
assert np.all(output.coefficients == np.asarray(coefficients))
