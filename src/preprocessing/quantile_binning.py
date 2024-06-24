from __future__ import annotations

from copy import copy
from typing import Union

from numpy import array, where
from pandas import DataFrame, isnull
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.utils.validation import (
    _check_feature_names_in,
    check_array,
    check_is_fitted,
)


class QuantileBinning(BaseEstimator, TransformerMixin):
    def __init__(self, nb_bins: int = 4, output_labels: bool = False):
        """
        Bin continuous data into quantiles
        :param nb_bins: Number of quantiles to create the bins from
        :param output_labels: Whether to use labels instead of bins indices
        """
        super().__init__()
        self.nb_bins = nb_bins
        self.output_labels = output_labels

    def fit(self, X: Union[DataFrame, array], y: None = None):
        """
        Fit the estimator
        :param X: Data to be discretized
        :param y: Ignored. This parameter exists only for compatibility with Pipeline
        :return: Returns the instance itself
        """
        X = copy(X)
        # Check the data format and values
        X = check_array(X, force_all_finite="allow-nan")
        # Transform the data to a DataFrame
        X = DataFrame(X)

        # Set the transformer attributes
        self.preprocessors_ = []
        self.bins_ = []
        self.n_features_in_ = X.shape[1]

        # Iterate over the columns to get the bins from sklearn's KBinsDiscretizer with a quantile strategy,
        # then store them into the transformer's attributes.
        for col in X.columns.values:
            preprocessor = KBinsDiscretizer(n_bins=self.nb_bins, encode="ordinal", strategy="quantile").fit(
                X[[col]].dropna()
            )
            self.preprocessors_.append(preprocessor)
            # Get the bins edges from the KBinsDiscretizer
            edges = preprocessor.bin_edges_[0]
            # Add First and last edges
            edges[0] = -float("inf")
            edges[-1] = float("inf")
            # Create the bins from the edges
            bins = []
            for i in range(0, len(edges) - 1):
                bins.append(str([edges[i], edges[i + 1]]))
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
            check_array(X, force_all_finite="allow-nan")
        else:
            X = copy(X)
            # Check the data format and values
            X = check_array(X, force_all_finite="allow-nan")
            # Transform the data to a DataFrame
            X = DataFrame(X)

        # Check is fit had been called
        check_is_fitted(self, ["preprocessors_", "n_features_in_"])

        # Assert that the number of features matches the one seen during fit
        try:
            assert X.shape[1] == self.n_features_in_
        except Exception:
            raise ValueError("The number of features provided don't match the one used for training")

        # Create a copy of the DataFrame in order to mask the missing values
        X_out = copy(X)

        # Iterate over the columns in order to apply the quantile discretization from the fitted bins
        for col_index in range(0, self.n_features_in_):

            # Discretize the copied DataFrame with imputed values (KBinsDiscretizer doesn't support missing values)
            X_out.iloc[:, col_index] = self.preprocessors_[col_index].transform(
                DataFrame(X_out.iloc[:, col_index]).fillna(0).to_numpy()
            )
            # Replace the DataFrame values by the ones from the previously transformed DataFrame
            # only when it's not missing, otherwise we indicate them with a special value
            X.iloc[:, col_index] = where(isnull(X.iloc[:, col_index]), -1, X_out.iloc[:, col_index]).astype(float)
            # If the user chooses the use bins content labels instead of bins indices
            if self.output_labels:
                # We concatenate the indices to the bins definitions to obtain the final label
                X.iloc[:, col_index] = (
                    X.iloc[:, col_index]
                    .apply(
                        lambda x: str(int(x)) + "_" + "Missing"
                        if x == -1
                        else str(int(x)) + "_" + self.bins_[col_index][int(x)]
                    )
                    .astype(str)
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
