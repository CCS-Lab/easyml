import pandas as pd

from easyml.factory import easy_glmnet


if __name__ == "__main__":
    # Load data
    data = pd.read_table('./cocaine.txt')

    # Analyze data
    easy_glmnet(data) # need to update constants and make them arguments
