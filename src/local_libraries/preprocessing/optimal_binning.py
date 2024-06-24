from __future__ import annotations

from copy import copy
from typing import Union

from numpy import array, char, number, where
from optbinning import OptimalBinning as OptBin
from pandas import DataFrame, Series, isnull
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import (
    _check_feature_names_in,
    check_array,
    check_is_fitted,
    check_X_y,
)


class OptimalBinning(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        min_nb_buckets: int = None,
        max_nb_buckets: int = None,
        dtype: str = "numerical",
        output_labels: bool = False,
    ):
        """
        Bin continuous/categorical data into optimal intervals/groups
        :param min_nb_buckets: Minimum number of buckets to be created
        :param max_nb_buckets: Maximum number of buckets to be created
        :param dtype: Types of the data to be discretized ("numerical" or "categorical")
        :param output_labels: Whether to use labels instead of bins indices
        """
        super().__init__()
        self.min_nb_buckets = min_nb_buckets
        self.max_nb_buckets = max_nb_buckets
        self.dtype = dtype
        try:
            assert dtype in ["numerical", "categorical"]
        except Exception:
            raise ValueError("dtype parameter must be in (numerical, categorical)")
        self.output_labels = output_labels

    def fit(self, X: Union[DataFrame, array], y: Union[Series, array]):
        """
        Fit the estimator
        :param X: Data to be discretized
        :param y: Target to fit the transformer on
        :return: the instance itself
        """
        if isinstance(X, DataFrame):
            X = X.copy()
            # Adapt the type verification according to the option chosen by the user
            if self.dtype == "categorical":
                try:
                    assert X.shape[1] == X.select_dtypes(include=[number, "string", "boolean", "category"]).shape[1]
                except Exception:
                    raise TypeError("DataFrame must have numeric, string or boolean dtypes.")
                # Cast them to string
                X = X.apply(lambda x: where(isnull(x), "Missing", x.astype("str")), axis=0)
                # Check that X and y have correct shape
                _, y = check_X_y(X, y, force_all_finite="allow-nan", dtype=str)
            else:
                try:
                    assert X.shape[1] == X.select_dtypes(include=[number]).shape[1]
                except Exception:
                    raise TypeError("DataFrame must have numeric dtypes.")
                # Check that X and y have correct shape
                _, y = check_X_y(X, y, force_all_finite="allow-nan")
        else:
            X = copy(X)
            # Check that X and y have correct shape
            X, y = check_X_y(X, y, force_all_finite="allow-nan")
            # Transform the dataset to a DataFrame
            X = DataFrame(X)
            # cast the columns to string if the categorical option is selected
            if self.dtype == "categorical":
                X = X.apply(lambda x: where(isnull(x), "Missing", x.astype("str")), axis="columns")

        # Set class attributes
        self.feature_names_ = X.columns.values
        self.n_features_in_ = len(self.feature_names_)
        self.preprocessors_ = []

        # Fit a preprocessor on each column and store them in an attribute
        for col in self.feature_names_:
            self.preprocessors_.append(
                OptBin(
                    name=col if type(col) == str else str(col),
                    dtype=self.dtype,
                    min_n_bins=self.min_nb_buckets,
                    max_n_bins=self.max_nb_buckets,
                ).fit(x=X[col], y=y)
            )

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
            check_array(X, force_all_finite="allow-nan", dtype=None)
        else:
            X = copy(X)
            X = check_array(X, force_all_finite="allow-nan")
            X = DataFrame(X)

        if isinstance(X, DataFrame):
            X = X.copy()
            # Adapt the type verification according to the option chosen by the user
            if self.dtype == "categorical":
                try:
                    assert X.shape[1] == X.select_dtypes(include=[number, "string", "boolean", "category"]).shape[1]
                except Exception:
                    raise TypeError("DataFrame must have numeric, string or boolean dtypes.")
                # Cast them to string
                X = X.apply(lambda x: where(isnull(x), "Missing", x.astype("str")), axis=0)
                # Check that X and y have correct shape
                check_array(X, force_all_finite="allow-nan", dtype=str)
            else:
                try:
                    assert X.shape[1] == X.select_dtypes(include=[number]).shape[1]
                except Exception:
                    raise TypeError("DataFrame must have numeric dtypes.")
                # Check that X and y have correct shape
                check_array(X, force_all_finite="allow-nan")
        else:
            X = copy(X)
            # Check that X and y have correct shape
            X = check_array(X, force_all_finite="allow-nan")
            X = DataFrame(X)
            if self.dtype == "categorical":
                X = X.apply(lambda x: where(isnull(x), "Missing", x.astype("str")), axis="columns")

        # Check is fit had been called
        check_is_fitted(self, ["preprocessors_", "n_features_in_", "feature_names_"])

        # Assert that the number of features provided matches the one used during fit
        try:
            assert X.shape[1] == len(self.feature_names_)
        except Exception:
            raise ValueError("The number of features provided don't match the one used for training")

        # If the user chooses to output bins labels instead of indices
        if self.output_labels:
            for col_index in range(0, len(self.feature_names_)):
                # Discretization by bins indices
                x_indices = char.array(
                    self.preprocessors_[col_index]
                    .transform(X.iloc[:, col_index], metric_missing=-1, metric="indices")
                    .astype(str)
                )
                # Discretization by bins content labels
                x_bins = char.array(
                    self.preprocessors_[col_index]
                    .transform(X.iloc[:, col_index], metric_missing=-1, metric="bins")
                    .astype(str)
                )
                # Build ordered labels as output
                X.iloc[:, col_index] = (x_indices + "_" + x_bins).astype(str)
        # If the user chooses to output bins indices instead of labels
        else:
            for col_index in range(0, len(self.feature_names_)):
                # Discretization by bins indices
                X.iloc[:, col_index] = (
                    self.preprocessors_[col_index]
                    .transform(X.iloc[:, col_index], metric_missing=-1, metric="indices")
                    .astype(float)
                )

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
        return {"allow_nan": True, "binary_only": True}
