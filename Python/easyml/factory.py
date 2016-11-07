"""Factory functions for quick and easy analysis.
"""
from glmnet import LogitNet
import matplotlib as mpl
import numpy as np
from os import path

# Set matplotlib settings
mpl.get_backend()
mpl.use('TkAgg')
import matplotlib.pyplot as plt
plt.style.use('ggplot')

from .bootstrap import bootstrap_aucs, bootstrap_coefficients, bootstrap_predictions
from .plot import plot_auc_histogram, plot_roc_curve
from .utils import process_coefficients, process_data
from .sample import sample_equal_proportion

__all__ = ['easy_glmnet']


def easy_glmnet(data):
    # Analysis constants
    EXCLUDE_VARIABLES = ['subject', 'AGE']
    DEPENDENT_VARIABLE = 'DIAGNOSIS'
    TRAIN_SIZE = 0.667
    N_DIVISIONS = 1000
    N_ITERATIONS = 10
    N_SAMPLES = 1000
    OUT_DIRECTORY = './results'

    # Model constants
    ALPHA = 1
    CUT_POINT = 0  # use 0 for minimum, 1 for within 1 SE
    MAX_ITER = 1e6
    N_FOLDS = 5
    N_LAMBDA = 200
    STANDARDIZE = False

    # Create model
    lr = LogitNet(alpha=ALPHA, cut_point=CUT_POINT, max_iter=MAX_ITER, n_folds=N_FOLDS, n_lambda=N_LAMBDA,
                  standardize=STANDARDIZE)

    # Process the data
    X, y = process_data(data, dependent_variable=DEPENDENT_VARIABLE, exclude_variables=EXCLUDE_VARIABLES)

    # Bootstrap coefficients
    coefs = bootstrap_coefficients(lr, X, y)

    # Process coefficients
    betas = process_coefficients(coefs)
    betas.to_csv(path.join(OUT_DIRECTORY, 'betas.csv'), index=False)

    # Split data
    mask = sample_equal_proportion(y, proportion=TRAIN_SIZE, random_state=43210)
    y_train = y[mask]
    y_test = y[np.logical_not(mask)]
    X_train = X[mask, :]
    X_test = X[np.logical_not(mask), :]

    # Bootstrap predictions
    all_y_train_scores, all_y_test_scores = bootstrap_predictions(lr, X_train, y_train, X_test, n_samples=N_SAMPLES)

    # Generate scores for training and test sets
    y_train_scores_mean = np.mean(all_y_train_scores, axis=0)
    y_test_scores_mean = np.mean(all_y_test_scores, axis=0)

    # Compute ROC curve and ROC area for training
    plot_roc_curve(y_train, y_train_scores_mean)
    plt.savefig(path.join(OUT_DIRECTORY, 'train_roc_curve.png'))

    # Compute ROC curve and ROC area for test
    plot_roc_curve(y_test, y_test_scores_mean)
    plt.savefig(path.join(OUT_DIRECTORY, 'test_roc_curve.png'))

    # Bootstrap training and test AUCS
    all_train_aucs, all_test_aucs = bootstrap_aucs(lr, X, y, n_divisions=N_DIVISIONS, n_iterations=N_ITERATIONS)

    # Plot histogram of training AUCS
    plot_auc_histogram(all_train_aucs)
    plt.savefig(path.join(OUT_DIRECTORY, 'train_auc_distribution.png'))

    # Plot histogram of test AUCS
    plot_auc_histogram(all_test_aucs)
    plt.savefig(path.join(OUT_DIRECTORY, 'test_auc_distribution.png'))

