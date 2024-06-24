from __future__ import annotations

from typing import TYPE_CHECKING

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y

if TYPE_CHECKING:
    from typing import Str


class Tokenizer(BaseEstimator, TransformerMixin):
    def __init__(self, model_type: Str):
        super().__init__()
        self.model_type = model_type
        self.model = None

    def fit(self, X, y=None):
        # Check that X and y have correct shape
        X, y = check_X_y(X, y)
        ...
        return self

    def predict(self, X):
        # Check is fit had been called
        check_is_fitted(self)

        # Input validation
        X = check_array(X)
        ...
        return ...
