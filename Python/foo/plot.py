"""Utility functions for plotting.
"""
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics


__all__ = ['plot_auc_histogram', 'plot_betas', 'plot_roc_curve']


def plot_auc_histogram(x, bins):
    x_mean = np.mean(x)
    plt.figure()
    plt.hist(x, bins=bins, color='white', edgecolor='black')
    plt.axvline(x=x_mean, color='black', linestyle='--')
    plt.annotate('Mean AUC = %.3f' % x_mean, xy=(150, 200), xycoords='figure pixels', size=28)
    plt.xlim([0.0, 1.0])
    plt.xlabel('AUC')
    plt.ylabel('Frequency')
    plt.title('Distribution of AUCs (Training Set)')


def plot_betas(betas):
    return 1


def plot_roc_curve(y_true, y_pred):
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(2):
        fpr[i], tpr[i], _ = metrics.roc_curve(y_true, y_pred)
        roc_auc[i] = metrics.auc(fpr[i], tpr[i])

    # Compute train ROC curve
    plt.figure()
    plt.plot(fpr[1], tpr[1], color='black',
             lw=2, label='AUC = %.3f' % roc_auc[1])
    plt.plot([0, 1], [0, 1], color='black', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('1 - Specificity')
    plt.ylabel('Sensitivity')
    plt.title('ROC Curve (Training Set)')
    plt.legend(loc="lower right")
