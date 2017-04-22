"""
Functions for plotting.
"""
import matplotlib.pyplot as plt
import numpy as np
import scikitplot.plotters as skplt



__all__ = []


def plot_coefficients_processed(coefficients):
    return coefficients


def plot_variable_importances_processed(importances):
    # Plot the feature importances of the forest
    importances_mean = np.mean(importances, axis=0)
    importances_std = np.std(importances, axis=0)
    n = importances.shape[1]
    fig = plt.figure()
    plt.title('Feature importances')
    plt.bar(range(n), importances_mean, color='grey', ecolor='black',
            yerr=importances_std, align='center')
    return fig

def plot_metrics_gaussian_mean_squared_error(x):
    bins = np.linspace(0, np.max(x), 100)
    x_mean = np.mean(x)
    fig = plt.figure()
    plt.hist(x, bins=bins, color='white', edgecolor='black')
    plt.axvline(x=x_mean, color='black', linestyle='--')
    plt.annotate('Mean MSE = %.3f' % x_mean, xy=(150, 200), xycoords='figure pixels', size=28)
    plt.xlabel('MSE')
    plt.ylabel('Frequency')
    plt.title('Distribution of MSEs')
    return fig


def plot_metrics_gaussian_cor_score(x):
    bins = np.arange(0, 1, 0.02)
    x_mean = np.mean(x)
    plt.figure()
    plt.hist(x, bins=bins, color='white', edgecolor='black')
    plt.axvline(x=x_mean, color='black', linestyle='--')
    plt.annotate('Mean Correlation Score = %.3f' % x_mean, xy=(150, 200), xycoords='figure pixels', size=28)
    plt.xlim([0.0, 1.0])
    plt.xlabel('Correlation Score')
    plt.ylabel('Frequency')
    plt.title('Distribution of Correlation Scores')


def plot_metrics_gaussian_r2_score(x):
    bins = np.arange(0, 1, 0.02)
    x_mean = np.mean(x)
    plt.figure()
    plt.hist(x, bins=bins, color='white', edgecolor='black')
    plt.axvline(x=x_mean, color='black', linestyle='--')
    plt.annotate('Mean R^2 Score = %.3f' % x_mean, xy=(150, 200), xycoords='figure pixels', size=28)
    plt.xlim([0.0, 1.0])
    plt.xlabel('R^2')
    plt.ylabel('Frequency')
    plt.title('Distribution of R^2 scores')


def plot_metrics_binomial_area_under_curve(x):
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


def plot_predictions_gaussian(y_true, y_pred):
    plt.figure()
    plt.plot(y_pred, y_true, "o")
    plt.xlabel('Predicted y values')
    plt.ylabel('True y values')
    plt.title('')


def plot_predictions_binomial(y_true, y_pred):
    Y_pred = np.concatenate((np.expand_dims(1 - y_pred, axis=1), np.expand_dims(y_pred, axis=1)), axis=1)
    fig = skplt.plot_roc_curve(y_true, Y_pred)
    return fig
