from __future__ import annotations

from copy import copy
from typing import Union, Optional

from numpy import array, concatenate
from pandas import DataFrame
from sklearn.base import BaseEstimator, TransformerMixin
from cinnamon.drift import AdversarialDriftExplainer
from sklearn.utils.validation import (
    _check_feature_names_in,
    check_array,
    check_is_fitted,
)


class ReweightingPopulation(BaseEstimator, TransformerMixin):
    def __init__(self, seed: int = 123, max_ratio: int = 10):
        """
        DEFINITION
        :param seed
        :param max_ratio
        """
        super().__init__()
        self.seed = seed
        self.max_ratio = max_ratio


    def fit(self, X: None = None, y: None = None, reference_df: Optional[DataFrame] = None):
        """
        Fit the estimator
        :param X:
        :param y:
        :param reference_df:
        :return: Returns the instance itself
        """
        if reference_df:
            reference_df = copy(reference_df)
            # Check the data format and values
            reference_df = check_array(reference_df, force_all_finite="allow-nan")
            # Transform the data to a DataFrame
            reference_df = DataFrame(reference_df)
            self.reference_df_ = reference_df
            self.n_features_in_ = reference_df.shape[1]
        else:
            X = copy(X)
            # Check the data format and values
            X = check_array(X, force_all_finite="allow-nan")
            # Transform the data to a DataFrame
            X = DataFrame(X)
            self.reference_df_ = X.copy()
            self.n_features_in_ = X.shape[1]

        return self

    def transform(self, X: Union[DataFrame, array]) -> array:
        """
        DEFINITION
        :param X:
        :return:
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
        check_is_fitted(self, ["reference_df_", "n_features_in_"])

        # Assert that the number of features matches the one seen during fit
        try:
            assert X.shape[1] == self.n_features_in_
        except Exception:
            raise ValueError("The number of features provided don't match the one used for training")

        # Set the transformer attributes
        sample_weights_adversarial = (AdversarialDriftExplainer(seed=self.seed)
                                            .fit(X.set_axis(self.reference_df_.columns.values, axis=1, inplace=False), self.reference_df_)
                                            .get_adversarial_correction_weights(max_ratio=self.max_ratio))

        X.loc[:, 'sample_weights'] = sample_weights_adversarial
        return X.to_numpy()

    def get_feature_names_out(self, input_features: list[str] = None) -> list[str]:
        """
        Get output feature names for transformation
        :param input_features: Input features
        :return: Transformed feature names
        """
        checked_input_features = _check_feature_names_in(self, input_features)
        return concatenate([checked_input_features, ['sample_weights']])

    def _more_tags(self):
        # The algorithm is non deterministic as it is sensitive to samples' order
        return {"allow_nan": True, "non_deterministic": True}