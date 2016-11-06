from glmnet import LogitNet
import matplotlib as mpl
import numpy as np
import pandas as pd

# Set matplotlib settings
mpl.get_backend()
mpl.use('TkAgg')
import matplotlib.pyplot as plt
plt.style.use('ggplot')

from easyml.bootstrap import bootstrap_aucs, bootstrap_coefficients, bootstrap_predictions
from easyml.datasets import cocaine
from easyml.plot import plot_auc_histogram, plot_roc_curve
from easyml.utils import process_coefficients, process_data
from easyml.sample import sample_equal_proportion


# Constants
EXCLUDE_AGE = False
TRAIN_SIZE = 0.667
MAX_ITER = 1e6
ALPHA = 1
N_LAMBDA = 200
N_FOLDS = 5
N_DIVISIONS = 1000
N_ITERATIONS = 10
CUT_POINT = 0  # use 0 for minimum, 1 for within 1 SE
SURVIVAL_RATE_CUTOFF = 0.05
SHOW = False
SAVE = True

# Load data
# data = cocaine.load_data()
data = pd.read_table('./cocaine.txt')

# Exclude certain variables
variables = ['subject']

if EXCLUDE_AGE:
    variables.append('AGE')

# Process the data
X, y = process_data(data, dependent_variables='DIAGNOSIS', exclude_variables=variables)

##############################################################################
# Replicating figure 1 - Done!
##############################################################################
# Bootstrap coefficients
lr = LogitNet(alpha=ALPHA, n_lambda=N_LAMBDA, standardize=False, cut_point=CUT_POINT, max_iter=MAX_ITER)
coefs = bootstrap_coefficients(lr, X, y)

# Process coefficients
betas = process_coefficients(coefs)
betas.to_csv('./results/betas.csv', index=False)

##############################################################################
# Replicating figure 2 - Done!
##############################################################################
# Split data
mask = sample_equal_proportion(y, proportion=TRAIN_SIZE, random_state=43210)
y_train = y[mask]
y_test = y[np.logical_not(mask)]
X_train = X[mask, :]
X_test = X[np.logical_not(mask), :]

# Bootstrap predictions
lr = LogitNet(alpha=ALPHA, n_lambda=N_LAMBDA, standardize=False, cut_point=CUT_POINT, max_iter=MAX_ITER)
all_y_train_scores, all_y_test_scores = bootstrap_predictions(lr, X_train, y_train, X_test, y_test, n_samples=1000)

# Generate scores for training and test sets
y_train_scores_mean = np.mean(all_y_train_scores, axis=0)
y_test_scores_mean = np.mean(all_y_test_scores, axis=0)

# Compute ROC curve and ROC area for training
plot_roc_curve(y_train, y_train_scores_mean)
plt.savefig('./results/train_roc_curve.png')

# Compute ROC curve and ROC area for test
plot_roc_curve(y_test, y_test_scores_mean)
plt.savefig('./results/test_roc_curve.png')

##############################################################################
# Replicating figure 4 - Done!
##############################################################################
all_train_aucs, all_test_aucs = bootstrap_aucs(lr, X, y, n_divisions=1000, n_iterations=5)

# Plot histogram of training AUCS
plot_auc_histogram(all_train_aucs)
plt.savefig('./results/train_auc_distribution.png')

# Plot histogram of test AUCS
plot_auc_histogram(all_test_aucs)
plt.savefig('./results/test_auc_distribution.png')