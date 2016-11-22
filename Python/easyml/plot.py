"""Utility functions for plotting.
"""
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics


__all__ = []


def plot_auc_histogram(x):
    bins = np.arange(0, 1, 0.02)
    x_mean = np.mean(x)
    plt.figure()
    plt.hist(x, bins=bins, color='white', edgecolor='black')
    plt.axvline(x=x_mean, color='black', linestyle='--')
    plt.annotate('Mean AUC = %.3f' % x_mean, xy=(150, 200), xycoords='figure pixels', size=28)
    plt.xlim([0.0, 1.0])
    plt.xlabel('AUC')
    plt.ylabel('Frequency')
    plt.title('Distribution of AUCs')


def plot_mse_histogram(x):
    bins = np.linspace(0, np.max(x), 100)
    x_mean = np.mean(x)
    plt.figure()
    plt.hist(x, bins=bins, color='white', edgecolor='black')
    plt.axvline(x=x_mean, color='black', linestyle='--')
    plt.annotate('Mean MSE = %.3f' % x_mean, xy=(150, 200), xycoords='figure pixels', size=28)
    plt.xlabel('MSE')
    plt.ylabel('Frequency')
    plt.title('Distribution of MSEs')


def plot_roc_curve(y_true, y_pred):
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(2):
        fpr[i], tpr[i], _ = metrics.roc_curve(y_true, y_pred)
        roc_auc[i] = metrics.auc(fpr[i], tpr[i])

    # Plot ROC curve
    plt.figure()
    plt.plot(fpr[1], tpr[1], color='black',
             lw=2, label='AUC = %.3f' % roc_auc[1])
    plt.plot([0, 1], [0, 1], color='black', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('1 - Specificity')
    plt.ylabel('Sensitivity')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")


def plot_gaussian_predictions(y_true, y_pred):
    plt.figure()
    plt.plot(y_pred, y_true, "o")
    plt.xlabel('Predicted y values')
    plt.ylabel('True y values')
    plt.title('')
