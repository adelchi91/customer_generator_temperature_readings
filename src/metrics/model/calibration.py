from io import BytesIO
from typing import Union

from matplotlib.pyplot import (
    close,
    figure,
    legend,
    plot,
    savefig,
    tight_layout,
    title,
    ylabel,
    ylim,
    xlim
)
from numpy import array
from pandas import Series
from sklearn.calibration import calibration_curve


def plot_calibration_curve(
    y_true: Union[Series, array], y_pred: Union[Series, array], plot_title: str = "Calibration plot (reliability curve)"
) -> bytes:
    """
    Produces the calibration plot
    :param y_true: Target
    :param y_pred: Predictions to evaluate
    :param plot_title: Title of the plot
    :return: Bytes plot
    """
    figure(figsize=(10, 10))
    plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
    fraction_of_positives, mean_predicted_value = calibration_curve(y_true, y_pred, n_bins=5, strategy='quantile')
    plot(
        mean_predicted_value,
        fraction_of_positives,
        "s-",
        label="%s" % ("Calibration curve",),
    )
    ylabel("Fraction of positives")
    # ylim([-0.05, 0.30])
    # xlim([-0.05, 0.30])
    ylim([-0.05, 1.])
    xlim([-0.05, 1.])
    legend(loc="lower right")

    # Display
    title(plot_title)
    tight_layout()

    figfile = BytesIO()
    savefig(figfile, format="png")
    bytes_plot = figfile.getvalue()
    close("all")
    return bytes_plot
