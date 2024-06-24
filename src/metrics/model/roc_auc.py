from io import BytesIO
from typing import Union

import plotnine as p9
from matplotlib.pyplot import close
from numpy import array
from pandas import DataFrame, Series
from sklearn.metrics import auc, roc_curve

from yc_younipy.plot import scales_percent, theme_yc, yc_colors


def plot_roc_curve(
    y_true: Union[Series, array], y_pred: Union[Series, array], x_group: Union[Series, array, None]
) -> bytes:
    """
    Produces the roc curve for a given set of target values and predictions
    :param y_true: Target values
    :param y_pred: Predictions to evaluate
    :param x_group: Grouping values if needed
    :return: plot image in bytes format.
    """
    df = DataFrame.from_dict({"y_true": y_true, "y_pred": y_pred, "group": x_group})

    def get_positive_rate(x):
        fpr, tpr, _ = roc_curve(x["y_true"], x["y_pred"])
        return DataFrame.from_dict({"False Positive Rate": fpr, "True Positive Rate": tpr})

    if x_group is not None:
        positive_rates = df.groupby("group").apply(get_positive_rate).reset_index()
    else:
        positive_rates = get_positive_rate(df).reset_index()

    group_color_kwarg = {"color": "group"} if x_group is not None else {}
    unique_color_kwarg = {} if x_group is not None else {"color": "#9163f2"}

    roc_curves = (
        p9.ggplot(positive_rates)
        + p9.geom_line(
            p9.aes(x="False Positive Rate", y="True Positive Rate", **group_color_kwarg), size=0.7, **unique_color_kwarg
        )
        + theme_yc()
        + p9.geom_segment(x=0, y=0, xend=1, yend=1, size=0.3, linetype="dotted")
        + p9.ggtitle("Receiver operating characteristic")
        + p9.theme(legend_position="right")
        + p9.scale_color_manual(values=yc_colors([1, 3]))
        + p9.scale_x_continuous(labels=scales_percent)
        + p9.scale_y_continuous(labels=scales_percent)
    )

    # Save the plot as a bytes image
    figfile = BytesIO()
    p9.ggsave(roc_curves, figfile, format="png", units="cm", height=14, width=24)
    bytes_plot = figfile.getvalue()
    close("all")

    return bytes_plot


def auc_computation(y_true: Union[Series, array], y_pred: Union[Series, array]) -> float:
    """
    Computes the auc value for the given dataframe as defined by the dataframe_id (e.g. validation or train_test)
    :param y_true: target dataset
    :param y_pred: predictions dataset to be evaluated
    :return: AUC metric
    """
    # Computing AUC
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    auc_metric = auc(fpr, tpr)
    return auc_metric


def gini_computation(y_true: Union[Series, array], y_pred: Union[Series, array]) -> float:
    """
    Computes the Gini metric for the given dataframe as defined by the dataframe_id (e.g. validation or train_test).
    :param y_true: target dataset
    :param y_pred: predictions dataset to be evaluated
    :return: Gini metric
    """
    # Computing AUC
    auc_metric = auc_computation(y_true, y_pred)
    # Returning Gini
    return 2 * auc_metric - 1
