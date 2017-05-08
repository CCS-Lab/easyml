"""
Functions for plotting.
"""
import matplotlib.pyplot as plt
import numpy as np
import scikitplot.plotters as skplt


__all__ = []

# Settings
plt.style.use('ggplot')


def plot_metrics_gaussian_mean_squared_error(x):
    bins = np.linspace(0, np.max(x), 100)
    x_mean = np.mean(x)
    fig, ax = plt.figure(), plt.gca()
    ax.hist(x, bins=bins, color='white', edgecolor='black')
    ax.axvline(x=x_mean, color='black', linestyle='--')
    ax.annotate('Mean MSE = %.3f' % x_mean, xy=(150, 200), xycoords='figure pixels', size=28)
    ax.set_xlabel('MSE')
    ax.set_ylabel('Frequency')
    ax.set_title('Distribution of MSEs')
    return fig


def plot_model_performance_gaussian_cor_score(x):
    bins = np.arange(0, 1, 0.02)
    x_mean = np.mean(x)
    fig, ax = plt.figure(), plt.gca()
    ax.hist(x, bins=bins, color='white', edgecolor='black')
    ax.axvline(x=x_mean, color='black', linestyle='--')
    ax.annotate('Mean Correlation Score = %.3f' % x_mean, xy=(150, 200), xycoords='figure pixels', size=28)
    ax.set_xlim([0.0, 1.0])
    ax.set_xlabel('Correlation Score')
    ax.set_ylabel('Frequency')
    ax.set_title('Distribution of Correlation Scores')
    return fig


def plot_model_performance_gaussian_r2_score(x):
    bins = np.arange(0, 1, 0.02)
    x_mean = np.mean(x)
    fig, ax = plt.figure(), plt.gca()
    ax.hist(x, bins=bins, color='white', edgecolor='black')
    ax.axvline(x=x_mean, color='black', linestyle='--')
    ax.annotate('Mean R^2 Score = %.3f' % x_mean, xy=(150, 200), xycoords='figure pixels', size=28)
    ax.set_xlim([0.0, 1.0])
    ax.set_xlabel('R^2')
    ax.set_ylabel('Frequency')
    ax.set_title('Distribution of R^2 scores')
    return fig


def plot_model_performance_binomial_area_under_curve(x):
    bins = np.arange(0, 1, 0.02)
    x_mean = np.mean(x)
    fig, ax = plt.figure(), plt.gca()
    ax.hist(x, bins=bins, color='white', edgecolor='black')
    ax.axvline(x=x_mean, color='black', linestyle='--')
    ax.annotate('Mean AUC = %.3f' % x_mean, xy=(150, 200), xycoords='figure pixels', size=28)
    ax.set_xlim([0.0, 1.0])
    ax.set_xlabel('AUC')
    ax.set_ylabel('Frequency')
    ax.set_title('Distribution of AUCs')
    return fig


def plot_predictions_gaussian(y_true, y_pred):
    fig, ax = plt.figure(), plt.gca()
    ax.scatter(y_pred, y_true, color='black')
    ax.set_xlabel('Predicted y values')
    ax.set_ylabel('True y values')
    ax.set_title('')
    return fig


def plot_predictions_binomial(y_true, y_pred):
    Y_pred = np.concatenate((np.expand_dims(1 - y_pred, axis=1), np.expand_dims(y_pred, axis=1)), axis=1)
    fig = skplt.plot_roc_curve(y_true, Y_pred)
    return fig
