"""
Functions for plotting.
"""
import matplotlib.pyplot as plt
import numpy as np
import scikitplot.plotters as skplt
from sklearn import linear_model
from sklearn.metrics import roc_auc_score


__all__ = []

# Settings
plt.style.use('ggplot')


def plot_model_performance_gaussian_mean_squared_error(x, subtitle='Train'):
    bins = np.linspace(0, np.max(x), 30)
    x_mean = np.mean(x)
    fig, ax = plt.figure(), plt.gca()
    ax.hist(x, bins=bins, color='white', edgecolor='black')
    ax.axvline(x=x_mean, color='black', linestyle='--')
    ax.annotate('Mean MSE = %.3f' % x_mean, xy=(150, 200), xycoords='figure pixels', size=28)
    ax.set_xlabel('MSE')
    ax.set_ylabel('Frequency')
    ax.set_title('Distribution of MSEs\n{} Dataset'.format(subtitle), loc='left')
    return fig


def plot_model_performance_gaussian_cor_score(x, subtitle='Train'):
    bins = np.arange(0, 1.01, 0.01)
    x_mean = np.mean(x)
    fig, ax = plt.figure(), plt.gca()
    ax.hist(x, bins=bins, color='white', edgecolor='black')
    ax.axvline(x=x_mean, color='black', linestyle='--')
    ax.annotate('Mean Correlation Score = %.3f' % x_mean, xy=(150, 200), xycoords='figure pixels', size=28)
    ax.set_xlim([0.0, 1.0])
    ax.set_xlabel('Correlation Score')
    ax.set_ylabel('Frequency')
    ax.set_title('Distribution of Correlation Scores\n{} Dataset'.format(subtitle), loc='left')
    return fig


def plot_model_performance_gaussian_r2_score(x, subtitle='Train'):
    bins = np.arange(0, 1.01, 0.01)
    x_mean = np.mean(x)
    fig, ax = plt.figure(), plt.gca()
    ax.hist(x, bins=bins, color='white', edgecolor='black')
    ax.axvline(x=x_mean, color='black', linestyle='--')
    ax.annotate('Mean R^2 Score = %.3f' % x_mean, xy=(150, 200), xycoords='figure pixels', size=28)
    ax.set_xlim([0.0, 1.0])
    ax.set_xlabel('R^2')
    ax.set_ylabel('Frequency')
    ax.set_title('Distribution of R^2 scores\n{} Dataset'.format(subtitle), loc='left')
    return fig


def plot_model_performance_binomial_area_under_curve(x, subtitle='Train'):
    bins = np.arange(0, 1.01, 0.01)
    x_mean = np.mean(x)
    fig, ax = plt.figure(), plt.gca()
    ax.hist(x, bins=bins, color='white', edgecolor='black')
    ax.axvline(x=x_mean, color='black', linestyle='--')
    ax.annotate('Mean AUC = %.3f' % x_mean, xy=(150, 200), xycoords='figure pixels', size=28)
    ax.set_xlim([0.0, 1.0])
    ax.set_xlabel('AUC')
    ax.set_ylabel('Frequency')
    ax.set_title('Distribution of AUCs\n{} Dataset'.format(subtitle), loc='left')
    return fig


def plot_predictions_gaussian(y_true, y_pred, subtitle='Train'):
    # run the classifier
    clf = linear_model.LinearRegression()
    clf.fit(y_pred.reshape(-1, 1), y_true.reshape(-1, 1))

    fig, ax = plt.figure(), plt.gca()
    ax.scatter(y_pred, y_true, color='black')
    newx = np.linspace(0, np.max(y_pred) + np.std(y_pred), 100)

    loss = clf.predict(newx.reshape(-1, 1)).ravel()
    ax.plot(newx, loss, color='black')
    ax.set_xlabel('Predicted y values')
    ax.set_ylabel('True y values')
    correlation_score = round(np.corrcoef(y_true, y_pred)[0, 1], 2)
    title = 'Actual vs. Predicted y values (Correlation Score =  {})\n{} Dataset'.format(correlation_score, subtitle)
    ax.set_title(title, loc='left')
    return fig


def plot_predictions_binomial(y_true, y_pred, subtitle='Train'):
    # run the classifier
    clf = linear_model.LogisticRegression()
    clf.fit(y_pred.reshape(-1, 1), y_true.reshape(-1, 1))

    fig, ax = plt.figure(), plt.gca()
    ax.scatter(y_pred, y_true, color='black')
    newx = np.arange(0, 1, 0.001)

    def model(x):
        return 1 / (1 + np.exp(-x))

    loss = model(newx * clf.coef_ + clf.intercept_).ravel()
    ax.plot(newx, loss, color='black')
    ax.set_xlabel('Predicted y values')
    ax.set_ylabel('True y values')
    ax.set_xticks(np.arange(0, 1.05, 0.05))
    ax.set_yticks(np.arange(0, 1.05, 0.05))
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    correlation_score = round(np.corrcoef(y_true, y_pred)[0, 1], 2)
    title = 'Actual vs. Predicted y values (Correlation Score =  {})\n{} Dataset'.format(correlation_score, subtitle)
    ax.set_title(title, loc='left')
    return fig


def plot_roc_single_train_test_split(y_true, y_pred, subtitle='Train'):
    auc = round(roc_auc_score(y_true, y_pred), 2)
    title = 'ROC Curve (AUC = {})\n{} Dataset'.format(auc, subtitle)
    Y_pred = np.concatenate((np.expand_dims(1 - y_pred, axis=1), np.expand_dims(y_pred, axis=1)), axis=1)
    fig = skplt.plot_roc_curve(y_true, Y_pred, title=title)
    return fig
