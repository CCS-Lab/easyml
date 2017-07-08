"""
The core functionality of easyml.
"""
import numpy as np
import progressbar

from . import plot
from . import setters
from . import utils


__all__ = ['easy_analysis']


class easy_analysis:
    """
    The core recipe of easyml.

    This recipe is the workhorse behind all of the easy_* functions.
    """
    def __init__(self, data, dependent_variable,
                 algorithm=None, family='gaussian',
                 resample=None, preprocess=None, measure=None,
                 exclude_variables=None, categorical_variables=None,
                 train_size=0.667, survival_rate_cutoff=0.05,
                 n_samples=1000, n_divisions=1000, n_iterations=10,
                 random_state=None, progress_bar=True, n_core=1,
                 generate_coefficients=None,
                 generate_variable_importances=None,
                 generate_predictions=None, generate_model_performance=None,
                 model_args=None):
        # set attributes
        self.data = data
        self.dependent_variable = dependent_variable
        self.algorithm = algorithm
        self.family = family
        self.exclude_variables = exclude_variables
        self.categorical_variables = categorical_variables
        self.train_size = train_size
        self.survival_rate_cutoff = survival_rate_cutoff
        self.n_samples = n_samples
        self.n_divisions = n_divisions
        self.n_iterations = n_iterations
        self.random_state = random_state
        self.progress_bar = progress_bar
        self.n_core = n_core
        self.model_args = model_args
        self.estimator = self.create_estimator()
        if self.model_args is not None:
            self.estimator.set_params(**self.model_args)

        # Set random state
        setters.set_random_state(self.random_state)

        # Set resample function
        resample = setters.set_resample(resample, family)
        self.resample = resample

        # Set preprocess function
        preprocess = setters.set_preprocess(preprocess)
        self.preprocess = preprocess

        # Set measure function
        measure = setters.set_measure(measure, self.family)
        self.measure = measure

        # Set column names
        column_names = list(data.columns.values)
        column_names = setters.set_column_names(column_names, dependent_variable,
                                                exclude_variables, preprocess,
                                                categorical_variables)
        self.column_names = column_names

        # Remove variables
        data = utils.remove_variables(data, exclude_variables)

        # Set categorical variables
        categorical_variables = setters.set_categorical_variables(self.column_names, self.categorical_variables)
        self.categorical_variables = categorical_variables

        # Set dependent variable
        y = setters.set_dependent_variable(data, self.dependent_variable)
        self.y = y

        # Set independent variables
        data = data[[self.dependent_variable] + column_names]
        X = setters.set_independent_variables(data, self.dependent_variable)
        self.X = X

        # Preprocess X
        self.X_preprocessed = self.preprocess(self.X, categorical_variables=self.categorical_variables)

        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = self.resample(self.X, self.y)

        # Preprocess X_train, X_test
        self.X_train_preprocessed, self.X_test_preprocessed = preprocess(self.X_train, self.X_test,
                                                                         categorical_variables=categorical_variables)

        # Generate coefficients
        if generate_coefficients:
            self.coefficients = self.generate_coefficients()
            self.coefficients_processed = self.process_coefficients(self.coefficients, self.column_names,
                                                                    survival_rate_cutoff=self.survival_rate_cutoff)

        if generate_variable_importances:
            self.variable_importances = self.generate_variable_importances()
            self.variable_importances_processed = self.process_variable_importances(self.variable_importances)

        if generate_predictions:
            # generate predictions
            self.predictions = self.generate_predictions()

            # upack train and test predictions
            self.predictions_train, self.predictions_test = self.predictions

            # take average of predictions
            self.predictions_train_mean = np.mean(self.predictions_train, axis=0)
            self.predictions_test_mean = np.mean(self.predictions_test, axis=0)

        if generate_model_performance:
            # generate measures of model performance
            self.model_performance = self.generate_model_performance()

            # model performance function
            self.plot_model_performance = setters.set_plot_model_performance(self.measure)

            # unpack train and test measures of model performance
            self.model_performance_train, self.model_performance_test = self.model_performance

    def create_estimator(self):
        """
        See the subclass documentation for more information on this method.
        """
        raise NotImplementedError

    def extract_coefficients(self, estimator):
        """
        See the subclass documentation for more information on this method.
        """
        raise NotImplementedError

    def process_coefficients(self):
        """
        See the subclass documentation for more information on this method.
        """
        raise NotImplementedError

    def plot_coefficients_processed(self):
        """
        See the subclass documentation for more information on this method.
        """
        raise NotImplementedError

    def extract_variable_importances(self, estimator):
        """
        See the subclass documentation for more information on this method.
        """
        raise NotImplementedError

    def process_variable_importances(self):
        """
        See the subclass documentation for more information on this method.
        """
        raise NotImplementedError

    def plot_variable_importances_processed(self):
        """
        See the subclass documentation for more information on this method.
        """
        raise NotImplementedError

    def predict_model(self):
        """
        See the subclass documentation for more information on this method.
        """
        raise NotImplementedError

    def generate_coefficients_(self):
        """
        Generate coefficients for a model (if applicable).

        :return: An ndarray.
        """
        # Create estimator
        estimator = self.create_estimator()

        # Configure the estimator with custom model arguments
        if self.model_args:
            estimator = estimator.set_params(**self.model_args)

        # Fit estimator with the training set
        model = estimator.fit(self.X_preprocessed, self.y)

        # Extract coefficient
        coefficient = self.extract_coefficients(model)

        # Save coefficient
        return coefficient

    def generate_coefficients(self):
        """
        Generate coefficients for a model (if applicable).

        :return: An ndarray.
        """
        # Initialize progress bar (optional)
        if self.progress_bar:
            bar = progressbar.ProgressBar(max_value=self.n_samples)
            i = 0

        # Initialize containers
        coefficients = []

        # Run sequentially
        print('Generating coefficients from multiple model builds:')

        # Loop over number of iterations
        for _ in range(self.n_samples):
            coefficient = self.generate_coefficients_()
            coefficients.append(coefficient)

            # Increment progress bar
            if self.progress_bar:
                bar.update(i)
                i += 1

        # cast to np.ndarray
        coefficients = np.asarray(coefficients)

        # return coefficients
        return coefficients

    def generate_variable_importances_(self):
        """
        Generate variable importances for a model (if applicable).

        :return: An ndarray.
        """
        # Create estimator
        estimator = self.create_estimator()

        # Configure the estimator with custom model arguments
        if self.model_args:
            estimator = estimator.set_params(**self.model_args)

        # Fit estimator with the training set
        model = estimator.fit(self.X_preprocessed, self.y)

        # Extract variable_importance
        variable_importance = self.extract_variable_importances(model)

        # Save variable_importance
        return variable_importance

    def generate_variable_importances(self):
        """
        Generate variable importances for a model (if applicable).

        :return: An ndarray.
        """
        # Initialize progress bar (optional)
        if self.progress_bar:
            bar = progressbar.ProgressBar(max_value=self.n_samples)
            i = 0

        # Initialize containers
        variable_importances = []

        # Run sequentially
        print('Generating variable importances from multiple model builds:')

        # Loop over number of iterations
        for _ in range(self.n_samples):
            variable_importance = self.generate_variable_importances_()
            variable_importances.append(variable_importance)

            # Increment progress bar
            if self.progress_bar:
                bar.update(i)
                i += 1

        # cast to np.ndarray
        variable_importances = np.asarray(variable_importances)

        # return variable_importances
        return variable_importances

    def generate_predictions_(self):
        """
        Generate predictions for a model.

        :return: An ndarray.
        """
        # Create estimator
        estimator = self.create_estimator()

        # Configure the estimator with custom model arguments
        if self.model_args:
            estimator = estimator.set_params(**self.model_args)

        # Fit estimator with the training set
        model = estimator.fit(self.X_train_preprocessed, self.y_train)

        # Generate predictions for training and test sets
        y_train_pred = self.predict_model(model, self.X_train_preprocessed)
        y_test_pred = self.predict_model(model, self.X_test_preprocessed)

        # Save predictions
        return y_train_pred, y_test_pred

    def generate_predictions(self):
        """
        Generate predictions for a model.

        :return: An ndarray.
        """
        # Initialize progress bar (optional)
        if self.progress_bar:
            bar = progressbar.ProgressBar(max_value=self.n_samples)
            i = 0

        # Initialize containers
        y_train_preds = []
        y_test_preds = []

        # Run sequentially
        print('Generating predictions for a single train test split:')

        # Loop over number of iterations
        for _ in range(self.n_samples):
            y_train_pred, y_test_pred = self.generate_predictions_()
            y_train_preds.append(y_train_pred)
            y_test_preds.append(y_test_pred)

            # Increment progress bar
            if self.progress_bar:
                bar.update(i)
                i += 1

        # cast to np.ndarray
        y_train_preds = np.asarray(y_train_preds)
        y_test_preds = np.asarray(y_test_preds)

        return y_train_preds, y_test_preds

    def generate_model_performance_(self):
        """
        Generate measures of model performance for a model.

        :return: An ndarray.
        """
        # Split data
        X_train, X_test, y_train, y_test = self.resample(self.X, self.y)

        # Preprocess data
        X_train_preprocessed, X_test_preprocessed = self.preprocess(X_train, X_test,
                                                                    categorical_variables=self.categorical_variables)

        # Create temporary containers
        y_train_preds = []
        y_test_preds = []

        # Loop over number of iterations
        for _ in range(self.n_iterations):
            # Create estimator
            estimator = self.create_estimator()

            # Configure the estimator with custom model arguments
            if self.model_args:
                estimator = estimator.set_params(**self.model_args)

            # Fit estimator with the training set
            model = estimator.fit(self.X_train_preprocessed, self.y_train)

            # Generate predictions for training and test sets
            y_train_pred = self.predict_model(model, X_train_preprocessed)
            y_test_pred = self.predict_model(model, X_test_preprocessed)

            # Save measures of model performance
            y_train_preds.append(y_train_pred)
            y_test_preds.append(y_test_pred)

        # Take mean of measures of model performance
        predictions_train = np.mean(np.asarray(y_train_preds), axis=0)
        predictions_test = np.mean(np.asarray(y_test_preds), axis=0)

        # Create measures of model performance
        model_performance_train = self.measure(y_train, predictions_train)
        model_performance_test = self.measure(y_test, predictions_test)

        # Save model_performances
        return model_performance_train, model_performance_test

    def generate_model_performance(self):
        """
        Generate measures of model performance for a model.

        :return: An ndarray.
        """
        # Initialize progress bar (optional)
        if self.progress_bar:
            bar = progressbar.ProgressBar(max_value=self.n_divisions)
            i = 0

        # Create temporary containers
        model_performance_train_all = []
        model_performance_test_all = []
        
        # Run sequentially
        print('Generating measures of model performance over multiple train test splits:')

        # Loop over number of divisions
        for _ in range(self.n_divisions):
            # Bootstrap metric
            model_performance_train, model_performance_test = self.generate_model_performance_()

            # Process loop and save in temporary containers
            model_performance_train_all.append(model_performance_train)
            model_performance_test_all.append(model_performance_test)

            # Increment progress bar
            if self.progress_bar:
                bar.update(i)
                i += 1

        # cast to np.ndarray
        model_performance_train = np.asarray(model_performance_train_all)
        model_performance_test = np.asarray(model_performance_test_all)

        return model_performance_train, model_performance_test

    def plot_predictions_single_train_test_split_train(self):
        """
        Plot predictions.

        :return: Figure and axe objects.
        """
        y_train = self.y_train
        y_train_pred = np.mean(self.predictions_train, axis=0)
        if self.family == 'gaussian':
            fig, ax = plot.plot_predictions_gaussian(y_train, y_train_pred)
        else:
            fig, ax = plot.plot_predictions_binomial(y_train, y_train_pred)
        return fig, ax

    def plot_predictions_single_train_test_split_test(self):
        """
        Plot predictions.

        :return: Figure and axe objects.
        """
        y_test = self.y_test
        y_test_pred = np.mean(self.predictions_test, axis=0)
        if self.family == 'gaussian':
            fig, ax = plot.plot_predictions_gaussian(y_test, y_test_pred, subtitle='Test')
        else:
            fig, ax = plot.plot_predictions_binomial(y_test, y_test_pred, subtitle='Test')
        return fig, ax

    def plot_roc_single_train_test_split_train(self):
        """
        Plot ROC Curve.

        :return: Figure and axe objects.
        """
        y_train = self.y_train
        y_train_pred = np.mean(self.predictions_train, axis=0)
        if self.family == 'gaussian':
            raise NotImplementedError
        else:
            fig, ax = plot.plot_roc_single_train_test_split(y_train, y_train_pred)
        return fig, ax

    def plot_roc_single_train_test_split_test(self):
        """
        Plot ROC Curve.

        :return: Figure and axe objects.
        """
        y_test = self.y_test
        y_test_pred = np.mean(self.predictions_test, axis=0)
        if self.family == 'gaussian':
            raise NotImplementedError
        else:
            fig, ax = plot.plot_roc_single_train_test_split(y_test, y_test_pred, subtitle='Test')
        return fig, ax

    def plot_model_performance_train(self):
        """
        Plot model performance.

        :return: Figure and axe objects.
        """
        fig, ax = self.plot_model_performance(self.model_performance_train)
        return fig, ax

    def plot_model_performance_test(self):
        """
        Plot model performance.

        :return: Figure and axe objects.
        """
        fig, ax = self.plot_model_performance(self.model_performance_test, subtitle='Test')
        return fig, ax
