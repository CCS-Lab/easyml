import pandas as pd

from easyml.factory import easy_glmnet


if __name__ == "__main__":
    # Load data
    data = pd.read_table('./cocaine.txt')

    # Analyze data
    easy_glmnet(data, dependent_variable='DIAGNOSIS', family='binomial', exclude_variables=['subject', 'AGE'])
