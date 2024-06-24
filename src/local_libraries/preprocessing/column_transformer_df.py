from __future__ import annotations

from typing import Union

from numpy import array
from pandas import DataFrame
from sklearn.compose import ColumnTransformer


class ColumnTransformerDF(ColumnTransformer):
    def __init__(
        self,
        transformers,
        remainder="drop",
        sparse_threshold=0.3,
        n_jobs=None,
        transformer_weights=None,
        verbose=False,
        verbose_feature_names_out=True,
        output_dtypes: dict = None,
    ):
        super().__init__(
            transformers,
            remainder=remainder,
            sparse_threshold=sparse_threshold,
            n_jobs=n_jobs,
            transformer_weights=transformer_weights,
            verbose=verbose,
            verbose_feature_names_out=verbose_feature_names_out,
        )
        self.output_dtypes = output_dtypes

    def transform(self, X: Union[DataFrame, array]) -> DataFrame:
        """
        Transform X separately by each transformer, concatenate results into a DataFrame with the right column names
        and the right specified formats
        :param X: Data to be preprocessed
        :return: A DataFrame transformed by each transformer
        """
        # Use the original function from the inherited class
        array = super().transform(X)
        # Get the feature names from the inherited class
        columns = super().get_feature_names_out()
        # Create a DataFrame from the data and features
        df = DataFrame(array, columns=columns)
        # Apply the specified types
        if self.output_dtypes is not None:
            return df.astype(dtype=self.output_dtypes)
        return df

    def fit_transform(self, X: Union[DataFrame, array], y: Union[DataFrame, array] = None) -> DataFrame:
        """
        Fit all transformers, transform the data and concatenate results
        :param X: Data to be preprocessed
        :param y: Target data to be used for data processing
        :return: A DataFrame transformed by each transformer
        """
        # Use the original function from the inherited class
        array = super().fit_transform(X, y)
        # Get the feature names from the inherited class
        columns = super().get_feature_names_out()
        # Create a DataFrame from the data and features
        df = DataFrame(array, columns=columns)
        # Apply the specified types
        if self.output_dtypes is not None:
            return df.astype(dtype=self.output_dtypes)
        return df
