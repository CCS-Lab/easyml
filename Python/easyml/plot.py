"""
Functions for plotting.
"""
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn import linear_model
from sklearn.metrics import auc, roc_auc_score, roc_curve


__all__ = []

# Settings
sns.set_style('whitegrid')


def plot_model_performance_gaussian_mean_squared_error(x, subtitle='Train'):
    bins = np.linspace(0, np.max(x), 30)
    x_mean = np.mean(x)
    fig, ax = plt.subplots()
    ax.hist(x, bins=bins, color='black', edgecolor='black')
    ax.axvline(x=x_mean, color='black', linestyle='--')
    ax.set_xlabel('MSE')
    ax.set_ylabel('Frequency')
    title = 'Distribution of MSEs (Mean MSE = {})\n{} Dataset'.format(x_mean, subtitle)
    ax.set_title(title, loc='left')
    return fig, ax


def plot_model_performance_gaussian_cor_score(x, subtitle='Train'):
    bins = np.arange(0, 1.01, 0.01)
    x_mean = np.mean(x)
    fig, ax = plt.subplots()
    ax.hist(x, bins=bins, color='black', edgecolor='black')
    ax.axvline(x=x_mean, color='black', linestyle='--')
    ax.set_xlim([-0.05, 1.05])
    ax.set_xticks(np.arange(0, 1.05, 0.05))
    ax.set_xlabel('Correlation Score')
    ax.set_ylabel('Frequency')
    title = 'Distribution of Correlation Scores (Mean Correlation Score = {})\n{} Dataset'.format(x_mean, subtitle)
    ax.set_title(title, loc='left')
    return fig, ax


def plot_model_performance_gaussian_r2_score(x, subtitle='Train'):
    bins = np.arange(0, 1.01, 0.01)
    x_mean = np.mean(x)
    fig, ax = plt.subplots()
    ax.hist(x, bins=bins, color='black', edgecolor='black')
    ax.axvline(x=x_mean, color='black', linestyle='--')
    ax.annotate('Mean R^2 Score = %.3f' % x_mean, xy=(150, 200), xycoords='figure pixels', size=28)
    ax.set_xlim([-0.05, 1.05])
    ax.set_xticks(np.arange(0, 1.05, 0.05))
    ax.set_xlabel('R^2')
    ax.set_ylabel('Frequency')
    title = 'Distribution of R^2 Scores (Mean R^2 Score = {})\n{} Dataset'.format(x_mean, subtitle)
    ax.set_title(title, loc='left')
    return fig, ax


def plot_model_performance_binomial_area_under_curve(x, subtitle='Train'):
    bins = np.arange(0, 1.01, 0.01)
    x_mean = round(np.mean(x), 2)
    fig, ax = plt.subplots()
    ax.hist(x, bins=bins, color='black', edgecolor='black')
    ax.axvline(x=x_mean, color='black', linestyle='--')
    ax.set_xlim([-0.05, 1.05])
    ax.set_xticks(np.arange(0, 1.05, 0.05))
    ax.set_xlabel('AUC')
    title = 'Distribution of AUC Scores (Mean AUC Score = {})\n{} Dataset'.format(x_mean, subtitle)
    ax.set_title(title, loc='left')
    return fig, ax


def plot_predictions_gaussian(y_true, y_pred, subtitle='Train'):
    # run the classifier
    clf = linear_model.LinearRegression()
    clf.fit(y_pred.reshape(-1, 1), y_true.reshape(-1, 1))

    fig, ax = plt.subplots()
    ax.scatter(y_pred, y_true, color='black')
    newx = np.linspace(0, np.max(y_pred) + np.std(y_pred), 100)

    loss = clf.predict(newx.reshape(-1, 1)).ravel()
    ax.plot(newx, loss, color='black')
    ax.set_xlabel('Predicted y values')
    ax.set_ylabel('True y values')
    correlation_score = round(np.corrcoef(y_true, y_pred)[0, 1], 2)
    title = 'Actual vs. Predicted y values (Correlation Score =  {})\n{} Dataset'.format(correlation_score, subtitle)
    ax.set_title(title, loc='left')
    return fig, ax


def plot_predictions_binomial(y_true, y_pred, subtitle='Train'):
    # run the classifier
    clf = linear_model.LogisticRegression()
    clf.fit(y_pred.reshape(-1, 1), y_true.reshape(-1, 1))

    fig, ax = plt.subplots()
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
    return fig, ax


def plot_roc_single_train_test_split(y_true, y_pred, subtitle='Train'):
    auc_label = round(roc_auc_score(y_true, y_pred), 2)
    title = 'ROC Curve (AUC Score = {})\n{} Dataset'.format(auc_label, subtitle)
    Y_true = np.concatenate((np.expand_dims(1 - y_true, axis=1), np.expand_dims(y_true, axis=1)), axis=1)
    Y_pred = np.concatenate((np.expand_dims(1 - y_pred, axis=1), np.expand_dims(y_pred, axis=1)), axis=1)
    n_classes = Y_true.shape[1]

    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(Y_true[:, i], Y_pred[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    tpr[1][0] = 0
    fig, ax = plt.subplots()
    ax.plot(fpr[1], tpr[1], color='black')
    ax.plot([0, 1], [0, 1], color='black', linestyle='--')
    ax.set_xlabel('1 - Specificity')
    ax.set_xlim([-0.05, 1.05])
    ax.set_xticks(np.arange(0, 1.05, 0.05))
    ax.set_ylabel('Sensitivity')
    ax.set_ylim([-0.05, 1.05])
    ax.set_yticks(np.arange(0, 1.05, 0.05))
    ax.set_title(title, loc='left')
    return fig, ax
