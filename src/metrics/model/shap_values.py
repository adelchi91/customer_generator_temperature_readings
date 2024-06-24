from io import BytesIO
from typing import Union

from matplotlib.pyplot import close, savefig
from numpy import array
from pandas import DataFrame
from shap import (
    Explainer,
    KernelExplainer,
    TreeExplainer,
    dependence_plot,
    summary_plot,
)


def shap_summary_plot(model, X: Union[DataFrame, array], explainer_type: str = "generic") -> bytes:
    """
    Plots the shapley summary plot.
    See https://shap.readthedocs.io/en/latest/index.html for more information
    :param model: Model to explain
    :param X: Dataset to calculate the SHAP values on
    :param explainer_type: Type of SHAP explainer ("generic", "tree")
    :return: plot image in bytes format
    """
    shap_values = compute_shap_values(model, X, explainer_type)

    summary_plot(shap_values, X, show=False)

    buf = BytesIO()
    savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    bytes_plot = buf.read()
    buf.close()
    close("all")
    return bytes_plot


def shap_dependence_plot(model, X: Union[DataFrame, array], column_index: int) -> bytes:
    """
    Plots the shapley dependence plot for given col feature. More specifically, it shows with which variable the
    col feature interacts the most.
    See https://shap.readthedocs.io/en/latest/index.html for more information
    :param model: Model to explain
    :param X: Dataset to calculate the SHAP values on
    :param column_index: Grouping values if needed
    :return: plot image in bytes format
    """
    shap_values = compute_shap_values(model, X)

    if X.iloc[:, column_index].dtype in ["string", "boolean"]:
        X.iloc[:, column_index] = X.iloc[:, column_index].astype("category")

    dependence_plot(column_index, shap_values, X, alpha=0.5, show=False)

    buf = BytesIO()
    savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    bytes_plot = buf.read()
    buf.close()
    close("all")
    return bytes_plot


def compute_shap_values(model, X: Union[DataFrame, array], explainer_type: str = "generic") -> list:
    """
    It computes the shapley values for the corresponding model trained and fitted and returns them.
    See https://shap.readthedocs.io/en/latest/index.html for more information.
    :param model: Model to explain.
    :param X: Dataset to compute SHAP values on.
    :param explainer_type: Type of SHAP explainer ("generic", "tree" or "function").
    :return: Shap values for the dataset provided
    """
    try:
        assert explainer_type in ["generic", "tree", "function"]
    except ValueError:
        raise ValueError('explainer param must be in ["generic", "tree" or "function"]')

    if explainer_type == "tree":
        explainer = TreeExplainer(model)
    elif explainer_type == "function":
        explainer = KernelExplainer(model, X)
    else:
        explainer = Explainer(model, X)
    return explainer.shap_values(X)
