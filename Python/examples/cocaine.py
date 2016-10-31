from glmnet import LogitNet
import matplotlib as mpl
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from easyml.bootstrap import bootstrap_aucs, bootstrap_coefficients, bootstrap_predictions
from easyml.plot import plot_auc_histogram, plot_roc_curve
from easyml.process import process_coefficients
from easyml.sample import sample_equal_proportion


# Set matplotlib settings
mpl.get_backend()
mpl.use('TkAgg')
import matplotlib.pyplot as plt
plt.style.use('ggplot')

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
data = pd.read_table('../data/cocaine.txt')

# Drop subjects column
data = data.drop('subject', axis=1)

# Possibly exclude age
if EXCLUDE_AGE:
    data = data.drop('AGE', axis=1)

# Handle dependent variables
y = data['DIAGNOSIS'].values
data = data.drop('DIAGNOSIS', axis=1)

# Handle categorical variable
male = np.array([data['Male'].values]).T
data = data.drop('Male', axis=1)
X_raw = data.values

# Handle numeric variables
stdsc = StandardScaler()
X_std = stdsc.fit_transform(X_raw)

# Combine categorical variables and continuous variables
X = np.concatenate([male, X_std], axis=1)

##############################################################################
# Replicating figure 1 - Done!
##############################################################################
# Bootstrap coefficients
lr = LogitNet(alpha=ALPHA, n_lambda=N_LAMBDA, standardize=False, cut_point=CUT_POINT, max_iter=MAX_ITER)
coefs = bootstrap_coefficients(lr, X, y)

# Process coefficients
betas = process_coefficients(coefs)
betas.to_csv('./imgs/betas.csv', index=False)

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
all_y_train_scores = np.array(all_y_train_scores)
y_train_scores_mean = np.mean(all_y_train_scores, axis=0)
all_y_test_scores = np.array(all_y_test_scores)
y_test_scores_mean = np.mean(all_y_test_scores, axis=0)

# Compute ROC curve and ROC area for each training
plot_roc_curve(y_train, y_train_scores_mean)

# Compute ROC curve and ROC area for each test
plot_roc_curve(y_test, y_test_scores_mean)

##############################################################################
# Replicating figure 4 - Done!
##############################################################################
all_train_aucs, all_test_aucs = bootstrap_aucs(lr, X, y, n_divisions=1000, n_iterations=100)

all_train_aucs = np.array(all_train_aucs)
all_train_auc_mean = np.mean(all_train_aucs)
all_test_aucs = np.array(all_test_aucs)
all_test_auc_mean = np.mean(all_test_aucs)
bins = np.arange(0, 1, 0.02)

# Plot histogram of training AUCS
plot_auc_histogram(all_train_aucs, bins)

# Plot histogram of test AUCS
plot_auc_histogram(all_test_aucs, bins)
