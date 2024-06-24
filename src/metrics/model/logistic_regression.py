from typing import Union

from numpy import array, cosh, diagonal, dot, linalg, sqrt, tile
from pandas import DataFrame, concat
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression


def p_values_sigma_computation(model: LogisticRegression, X: Union[DataFrame, array]) -> (list[float], list[float]):
    """
    Computes p-values and standard deviation errors associated with the training of a Logistic Regression.
    The following code was adapted from the source here below
    https://gist.github.com/rspeare/77061e6e317896be29c6de9a85db301d

    :param model: trained model. In this case the model needs to be a Logistic Regression
    :param X: training dataset
    :return: p_values, sigma_estimates
    """
    denom = 2.0 * (1.0 + cosh(model.decision_function(X)))
    denom = tile(denom, (X.shape[1], 1)).T
    f_ij = dot((X / denom).T, X)  # Fisher Information Matrix
    cramer_rao = linalg.pinv(f_ij)  # Inverse Information Matrix
    # standard deviation
    sigma_estimates = sqrt(diagonal(cramer_rao))
    z_scores = model.coef_[0] / sigma_estimates  # z-score for each model coefficient
    p_values = [norm.sf(abs(x)) * 2 for x in z_scores]  # two tailed test for p-values
    return p_values, sigma_estimates


def lr_summary(model: LogisticRegression, X: Union[DataFrame, array]) -> (DataFrame, float):
    """
    Produces the the logistic regression score card, namely
    - standard deviation
    - p-value
    - coefficients

    Code was adapted from the following sources:
    - https://pythonguides.com/scikit-learn-logistic-regression/
    - https://stats.stackexchange.com/questions/89484/how-to-compute-the-standard-errors\
    -of-a-logistic-regressions-coefficients

    :param model: trained model. In this case the model needs to be a Logistic Regression
    :param X: training dataset
    :return: df_summary, intercept
    """
    # logistic regression coefficients
    coeffs = model.coef_
    # feature names
    features_names = model.feature_names_in_
    # p-value computation
    p_values, std = p_values_sigma_computation(model, X)
    # intercept
    intercept = model.intercept_
    # dataframes
    df_coeffs = DataFrame(data=coeffs, columns=features_names, index=["coefficients"])
    df_p_values = DataFrame(data=array([p_values]), columns=features_names, index=["p_values"])
    df_std = DataFrame(data=array([std]), columns=features_names, index=["std"])
    df_summary = concat([df_coeffs.T, df_p_values.T, df_std.T], axis=1)
    return df_summary, intercept
