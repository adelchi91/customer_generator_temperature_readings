import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from src.local_libraries.preprocessing.quantile_binning import QuantileBinning
from src.local_libraries.preprocessing.most_frequent_binning import MostFrequentBinning
from src.local_libraries.preprocessing.column_transformer_df import ColumnTransformerDF
import numpy as np


def binning(alt_w_pred_converted, numerical_vars, categorical_vars):
    # Define transformers for numerical and string columns
    num_transformer = Pipeline(steps=[
        ('binning', QuantileBinning(nb_bins=4, output_labels=True))
    ])

    str_transformer = Pipeline(steps=[
        ('imputer', MostFrequentBinning(top_n=4, output_labels=True)),
    ])

    # # Define preprocessor to apply transformers to respective columns
    preprocessor = ColumnTransformerDF([
        ('num', num_transformer, numerical_vars),
        ('str', str_transformer, categorical_vars),
    ], remainder="passthrough")

    # Transform data using pipeline
    output = preprocessor.fit_transform(alt_w_pred_converted)
    return output
