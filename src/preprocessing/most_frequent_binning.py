from __future__ import annotations

from copy import copy
from typing import Union

from numpy import array, char, number, select, where
from pandas import DataFrame, isnull
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import (
    _check_feature_names_in,
    check_array,
    check_is_fitted,
)


class MostFrequentBinning(BaseEstimator, TransformerMixin):
    def __init__(self, top_n: int = 4, output_labels: bool = False):
        """
        Bin continuous/categorical data into by keeping the top n occurrences and putting the remainder in 'Others'.
        :param top_n: Number of top features to keep (the last on will contain the remaining and wille be labelled
                "Others").
        :param output_labels: Whether to use labels instead of bins indices.
        """
        super().__init__()
        self.top_n = top_n
        self.output_labels = output_labels

    def fit(self, X: Union[DataFrame, array], y: None = None):
        """
        Fit the estimator
        :param X: Data to be discretized
        :param y: Ignored. This parameter exists only for compatibility with Pipeline.
        :return: Returns the instance itself.
        """
        if isinstance(X, DataFrame):
            # Check the data format and values
            check_array(X, force_all_finite="allow-nan", dtype=None)
            # Check the DataFrame dtypes
            try:
                assert X.shape[1] == X.select_dtypes(include=[number, "string", "boolean", "category"]).shape[1]
            except Exception:
                raise TypeError("DataFrame must have numeric, string or boolean dtypes.")
        else:
            X = copy(X)
            # Check the data format and values
            X = check_array(X, force_all_finite="allow-nan", dtype="numeric")
            # Transform the dataset to a DataFrame
            X = DataFrame(X)

        # Define the bins
        self.bins_ = []
        # Define the number of features expected by the transformer
        self.n_features_in_ = X.shape[1]

        # get the top n features for each column by iterating over them
        for col in X.columns.values:
            bins = list(X[col].dropna().astype(str).value_counts(ascending=False).head(4).index)
            self.bins_.append(bins)

        return self

    def transform(self, X: Union[DataFrame, array]) -> array:
        """
        Discretize the data.
        :param X: Data to be discretized.
        :return: Data in the binned space.
        """
        # Input validation
        if isinstance(X, DataFrame):
            X = X.copy()
            # Check the data format and values
            check_array(X, force_all_finite="allow-nan", dtype=None)
            try:
                assert X.shape[1] == X.select_dtypes(include=[number, "string", "boolean", "category"]).shape[1]
            except Exception:
                raise TypeError("DataFrame must have numeric, string or boolean dtypes.")
        else:
            X = copy(X)
            # Check the data format and values
            X = check_array(X, force_all_finite="allow-nan")
            # Transform the dataset to a DataFrame
            X = DataFrame(X)

        # Check is fit had been called
        check_is_fitted(self, ["bins_", "n_features_in_"])

        # Assert that the number of features matches the one seen during fit
        try:
            assert X.shape[1] == self.n_features_in_
        except Exception:
            raise ValueError("The number of features provided don't match the one used for training")

        # Iterate over the columns to apply the discretization
        for col_index in range(0, self.n_features_in_):

            # Discretization by bucket index
            indices = select(
                [isnull(X.iloc[:, col_index]), ~X.iloc[:, col_index].isin(self.bins_[col_index])],
                [-1, len(self.bins_[col_index])],
                where(
                    X.iloc[:, col_index].astype(str).isin(self.bins_[col_index]),
                    [
                        self.bins_[col_index].index(value)
                        for value in where(
                            X.iloc[:, col_index].astype(str).isin(self.bins_[col_index]),
                            X.iloc[:, col_index].astype(str),
                            self.bins_[col_index][0],
                        )
                    ],
                    None,
                ),
            ).astype(float)

            # If the user chose to output labels instead of bins indices
            if self.output_labels:
                # Discretization by bucket content
                bins = select(
                    [isnull(X.iloc[:, col_index]), X.iloc[:, col_index].astype(str).isin(self.bins_[col_index])],
                    ["Missing", "Others"],
                    X.iloc[:, col_index],
                ).astype(str)
                X.iloc[:, col_index] = (char.array(indices.astype(int).astype(str)) + "_" + char.array(bins)).astype(
                    str
                )
            else:
                X.iloc[:, col_index] = indices

        return X.to_numpy()

    def get_feature_names_out(self, input_features: list[str] = None) -> list[str]:
        """
        Get output feature names for transformation
        :param input_features: Input features
        :return: Transformed feature names
        """
        input_features = _check_feature_names_in(self, input_features)
        return input_features

    def _more_tags(self):
        return {"allow_nan": True}
