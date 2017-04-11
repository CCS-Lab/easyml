"""Functions for basic analysis.
"""
import concurrent.futures
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
                  generate_coefficients=True,
                  model_args=None):
        # set attributes
        self.data = data
        self.dependent_variable = dependent_variable
        self.algorithm = algorithm
        self.family = family
        self.resample = resample
        self.preprocess = preprocess
        self.measure = measure
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
        preprocess = setters.set_preprocess(self.preprocess)
        self.preprocess = preprocess

        # Set column names
        column_names = list(self.data.columns.values)
        column_names = setters.set_column_names(column_names, self.dependent_variable,
                                                self.exclude_variables, preprocess,
                                                self.categorical_variables)

        # Remove variables
        data = utils.remove_variables(self.data, self.exclude_variables)

        # Set categorical variables
        categorical_variables = setters.set_categorical_variables(column_names, self.categorical_variables)
        self.categorical_variables = categorical_variables

        # Set dependent variable
        y = setters.set_dependent_variable(data, self.dependent_variable)
        self.y = y

        # Set independent variables
        X = setters.set_independent_variables(data, self.dependent_variable)
        self.X = X

        # Preprocess data
        self.X_processed = self.preprocess(self.X, categorical_variables=self.categorical_variables)

        # Replicate coefficients
        if generate_coefficients:
            self.coefficients = self.generate_coefficients()

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
        model = estimator.fit(self.X_processed, self.y)

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

        if self.n_core > 1:
            print('Parallel is currently disabled. Running sequentially.')
            self.n_core = 1

        # Evaluate parallelism
        if self.n_core == 1:
            # Run sequentially
            print("Replicating coefficients:")

            # Loop over number of iterations
            for _ in range(self.n_samples):
                coefficient = self.generate_coefficient()
                coefficients.append(coefficient)

                # Increment progress bar
                if self.progress_bar:
                    bar.update(i)
                    i += 1
        # elif self.n_core > 1:
        #     # Run in parallel using n_core cores
        #     print("Replicating coefficients in parallel:")
        #
        #     # Handle case where n_core > os.cpu_count()
        #     n_core = utils.reduce_cores(self.n_core)
        #
        #     # Loop over number of iterations
        #     indexes = range(self.n_samples)
        #     with concurrent.futures.ProcessPoolExecutor(max_workers=n_core) as executor:
        #         futures = {executor.submit(self.generate_coefficient): ix for ix in indexes}
        #         for future in concurrent.futures.as_completed(futures):
        #             try:
        #                 coefficient = future.result()
        #                 coefficients.append(coefficient)
        #             except:
        #                 coefficients.append(None)
        #
        #             # increment progress bar
        #             if self.progress_bar:
        #                 bar.update(i)
        #                 i += 1
        # else:
        #     raise ValueError

        # cast to np.ndarray
        coefficients = np.asarray(coefficients)

        # return coefficients
        return coefficients

    def generate_prediction(self, estimator, fit_model, predict_model, X_train, y_train, X_test):
        # Fit estimator with the training set
        estimator = fit_model(estimator, X_train, y_train)

        # Generate predictions for training and test sets
        y_train_pred = predict_model(estimator, X_train)
        y_test_pred = predict_model(estimator, X_test)

        # Save predictions
        return y_train_pred, y_test_pred

    def generate_predictions(self, estimator, fit_model, predict_model, preprocessor, X_train, y_train, X_test,
                             categorical_variables=None, n_samples=1000, progress_bar=True, n_core=1):
        # Initialize progress bar (optional)
        if progress_bar:
            bar = progressbar.ProgressBar(max_value=n_samples)
            i = 0

        # Preprocess data
        X_train, X_test = preprocessor(X_train, X_test, categorical_variables=categorical_variables)

        # Initialize containers
        y_train_preds = []
        y_test_preds = []

        # Evaluate parallelism
        if n_core == 1:
            # Run sequentially
            print("Replicating predictions:")

            # Loop over number of iterations
            for _ in range(n_samples):
                y_train_pred, y_test_pred = generate_prediction(estimator, fit_model, predict_model, X_train, y_train,
                                                                X_test)
                y_train_preds.append(y_train_pred)
                y_test_preds.append(y_test_pred)

                # Increment progress bar
                if progress_bar:
                    bar.update(i)
                    i += 1

        elif n_core > 1:
            # Run in parallel using n_core cores
            print("Replicating predictions in parallel:")

            # Handle case where n_core > os.cpu_count()
            n_core = utils.reduce_cores(n_core)

            # Loop over number of iterations
            indexes = range(n_samples)
            with concurrent.futures.ProcessPoolExecutor(max_workers=n_core) as executor:
                futures = {
                executor.submit(generate_prediction, estimator, fit_model, predict_model, X_train, y_train, X_test): ix
                for ix in indexes}
                for future in concurrent.futures.as_completed(futures):
                    try:
                        y_train_pred, y_test_pred = future.result()
                        y_train_preds.append(y_train_pred)
                        y_test_preds.append(y_test_pred)
                    except:
                        y_train_preds.append(None)
                        y_test_preds.append(None)

                    # Increment progress bar
                    if progress_bar:
                        bar.update(i)
                        i += 1
        else:
            raise ValueError

        # cast to np.ndarray
        y_train_preds = np.asarray(y_train_preds)
        y_test_preds = np.asarray(y_test_preds)

        return y_train_preds, y_test_preds

    def generate_metric(self, estimator, sampler, fit_model, predict_model, preprocessor, measure, X, y,
                        categorical_variables=None, n_iterations=100):
        # Split data
        try:
            X_train, X_test, y_train, y_test = sampler(X, y)
        except Exception as e:
            print('Sample Exception: {}'.format(e))

        # Preprocess data
        X_train, X_test = preprocessor(X_train, X_test, categorical_variables=categorical_variables)

        # Create temporary containers
        train_metrics = []
        test_metrics = []

        # Loop over number of iterations
        for _ in range(n_iterations):
            # Fit estimator with the training set
            try:
                results = fit_model(estimator, X_train, y_train)
            except Exception as e:
                print('Model Exception: {}'.format(e))

            # Generate predictions for training and test sets
            try:
                y_train_pred = predict_model(results, X_train)
                y_test_pred = predict_model(results, X_test)
            except Exception as e:
                print('Predict Exception: {}'.format(e))

            # Calculate metric on training and test sets
            try:
                train_metric = measure(y_train, y_train_pred)
                test_metric = measure(y_test, y_test_pred)
            except Exception as e:
                print('Measure Exception: {}'.format(e))

            # Save metrics
            try:
                train_metrics.append(train_metric)
                test_metrics.append(test_metric)
            except Exception as e:
                print('Append Exception: {}'.format(e))

        # Take mean of metrics
        try:
            mean_train_metric = np.mean(np.asarray(train_metrics))
            mean_test_metric = np.mean(np.asarray(test_metrics))
        except Exception as e:
            print('Mean Exception: {}'.format(e))

        # Save metrics
        return mean_train_metric, mean_test_metric

    def generate_metrics(self, estimator, sampler, fit_model, predict_model, preprocessor, measure, X, y,
                         categorical_variables=None, n_divisions=1000, n_iterations=100, progress_bar=True, n_core=1):
        # Initialize progress bar (optional)
        if progress_bar:
            bar = progressbar.ProgressBar(max_value=n_divisions)
            i = 0

        # Create temporary containers
        mean_train_metrics = []
        mean_test_metrics = []

        # Evaluate parallelism
        if True:
            # Run sequentially
            print("Replicating metrics (parallelism is currently disabled for this function):")

            # Loop over number of divisions
            for _ in range(n_divisions):
                # Bootstrap metric
                mean_train_metric, mean_test_metric = generate_metric(estimator, sampler, fit_model,
                                                                      predict_model, preprocessor, measure, X, y,
                                                                      categorical_variables, n_iterations)

                # Process loop and save in temporary containers
                mean_train_metrics.append(mean_train_metric)
                mean_test_metrics.append(mean_test_metric)

                # Increment progress bar
                if progress_bar:
                    bar.update(i)
                    i += 1

        elif False:
            # Run in parallel using n_core cores
            print("Replicating metrics in parallel:")

            # Handle case where n_core > os.cpu_count()
            n_core = utils.reduce_cores(n_core)

            # Loop over number of iterations
            indexes = range(n_divisions)
            with concurrent.futures.ProcessPoolExecutor(max_workers=n_core) as executor:
                futures = {executor.submit(generate_metric, estimator, sample, fit_model, predict_model, measure, X, y,
                                           n_iterations): ix for ix in indexes}
                for future in concurrent.futures.as_completed(futures):
                    try:
                        mean_train_metric, mean_test_metric = future.result()
                    except Exception as e:
                        print('Exception: {}'.format(e))
                        mean_train_metric = None
                        mean_test_metric = None

                    # Save metrics
                    # print(mean_train_metric, mean_test_metric)
                    mean_train_metrics.append(mean_train_metric)
                    mean_test_metrics.append(mean_test_metric)

                    # Increment progress bar
                    if progress_bar:
                        bar.update(i)
                        i += 1
        else:
            raise ValueError

        # cast to np.ndarray
        mean_train_metrics = np.asarray(mean_train_metrics)
        mean_test_metrics = np.asarray(mean_test_metrics)

        return mean_train_metrics, mean_test_metrics
