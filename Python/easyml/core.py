"""
The core functionality of easyml.
"""
import numpy as np
import progressbar

from . import setters
from . import utils


__all__ = ['EasyAnalysis']


class EasyAnalysis:
    def __init__(self, data, dependent_variable,
                 algorithm=None, family='gaussian',
                 resample=None, preprocess=None, measure=None,
                 exclude_variables=None, categorical_variables=None,
                 train_size=0.667, survival_rate_cutoff=0.05,
                 n_samples=1000, n_divisions=1000, n_iterations=10,
                 random_state=None, progress_bar=True, n_core=1,
                 generate_coefficients=None,
                 generate_variable_importances=None,
                 generate_predictions=None, generate_metrics=None,
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

        # Set random state
        setters.set_random_state(self.random_state)

        # Set preprocess function
        preprocess = setters.set_preprocess(preprocess)
        self.preprocess = preprocess

        # Set resample function
        resample = setters.set_resample(resample, family)
        self.resample = resample

        # Set measure function
        measure = setters.set_measure(measure)
        self.measure = measure

        # Set column names
        column_names = list(data.columns.values)
        column_names = setters.set_column_names(column_names, dependent_variable,
                                                exclude_variables, preprocess,
                                                categorical_variables)

        # Remove variables
        data = utils.remove_variables(data, exclude_variables)

        # Set categorical variables
        categorical_variables = setters.set_categorical_variables(column_names, self.categorical_variables)
        self.categorical_variables = categorical_variables

        # Set dependent variable
        y = setters.set_dependent_variable(data, self.dependent_variable)
        self.y = y

        # Set independent variables
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

        if generate_variable_importances:
            print("Not implemented.")

        if generate_predictions:
            self.predictions = self.generate_predictions()

        if generate_metrics:
            self.metrics = self.generate_metrics()

    def create_estimator(self):
        raise NotImplementedError

    def extract_coefficients(self, estimator):
        raise NotImplementedError

    def process_coefficients(self):
        raise NotImplementedError

    def extract_variable_importances(self, estimator):
        raise NotImplementedError

    def process_variable_importances(self):
        raise NotImplementedError

    def predict_model(self):
        raise NotImplementedError

    def generate_coefficient(self):
        # Create estimator
        estimator = self.create_estimator()

        # Fit estimator with the training set
        model = estimator.fit(self.X_preprocessed, self.y)

        # Extract coefficient
        coefficient = self.extract_coefficients(model)

        # Save coefficient
        return coefficient

    def generate_coefficients(self):
        # Initialize progress bar (optional)
        if self.progress_bar:
            bar = progressbar.ProgressBar(max_value=self.n_samples)
            i = 0

        # Initialize containers
        coefficients = []

        # Run sequentially
        print("Generating coefficients:")

        # Loop over number of iterations
        for _ in range(self.n_samples):
            coefficient = self.generate_coefficient()
            coefficients.append(coefficient)

            # Increment progress bar
            if self.progress_bar:
                bar.update(i)
                i += 1

        # cast to np.ndarray
        coefficients = np.asarray(coefficients)

        # return coefficients
        return coefficients

    def generate_prediction(self):
        # Create estimator
        estimator = self.create_estimator()

        # Fit estimator with the training set
        model = estimator.fit(self.X_train_preprocessed, self.y_train)

        # Generate predictions for training and test sets
        y_train_pred = self.predict_model(model, self.X_train_preprocessed)
        y_test_pred = self.predict_model(model, self.X_test_preprocessed)

        # Save predictions
        return y_train_pred, y_test_pred

    def generate_predictions(self):
        # Initialize progress bar (optional)
        if self.progress_bar:
            bar = progressbar.ProgressBar(max_value=self.n_samples)
            i = 0

        # Initialize containers
        y_train_preds = []
        y_test_preds = []

        # Run sequentially
        print("Generating predictions:")

        # Loop over number of iterations
        for _ in range(self.n_samples):
            y_train_pred, y_test_pred = self.generate_prediction()
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

    def generate_metric(self):
        # Split data
        X_train, X_test, y_train, y_test = self.resample(self.X, self.y)

        # Preprocess data
        X_train_preprocessed, X_test_preprocessed = self.preprocess(X_train, X_test,
                                                                    categorical_variables=self.categorical_variables)

        # Create temporary containers
        train_metrics = []
        test_metrics = []

        # Loop over number of iterations
        for _ in range(self.n_iterations):
            # Create estimator
            estimator = self.create_estimator()

            # Fit estimator with the training set
            model = estimator.fit(self.X_train_preprocessed, self.y_train)


            # Generate predictions for training and test sets
            y_train_pred = self.predict_model(model, X_train_preprocessed)
            y_test_pred = self.predict_model(model, X_test_preprocessed)

            # Calculate metric on training and test sets
            train_metric = self.measure(y_train, y_train_pred)
            test_metric = self.measure(y_test, y_test_pred)

            # Save metrics
            train_metrics.append(train_metric)
            test_metrics.append(test_metric)

        # Take mean of metrics
        mean_train_metric = np.mean(np.asarray(train_metrics))
        mean_test_metric = np.mean(np.asarray(test_metrics))

        # Save metrics
        return mean_train_metric, mean_test_metric

    def generate_metrics(self):
        # Initialize progress bar (optional)
        if self.progress_bar:
            bar = progressbar.ProgressBar(max_value=self.n_divisions)
            i = 0

        # Create temporary containers
        mean_train_metrics = []
        mean_test_metrics = []
        
        # Run sequentially
        print("Generating metrics:")

        # Loop over number of divisions
        for _ in range(self.n_divisions):
            # Bootstrap metric
            mean_train_metric, mean_test_metric = self.generate_metric()

            # Process loop and save in temporary containers
            mean_train_metrics.append(mean_train_metric)
            mean_test_metrics.append(mean_test_metric)

            # Increment progress bar
            if self.progress_bar:
                bar.update(i)
                i += 1

        # cast to np.ndarray
        mean_train_metrics = np.asarray(mean_train_metrics)
        mean_test_metrics = np.asarray(mean_test_metrics)

        return mean_train_metrics, mean_test_metrics
