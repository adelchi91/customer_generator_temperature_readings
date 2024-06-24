from io import BytesIO
from typing import Union

import matplotlib.pyplot as plt
import plotnine
from numpy import array
from pandas import DataFrame, Series

from yc_younipy.plot import theme_yc, yc_colors


def plot_pop(
    x: Union[Series, array], x_group: Union[Series, array], variable_name: str, group_variable_name: str
) -> bytes:
    """
    Create pop plot
    :param x: Observations to plot
    :param x_group: Values indicating the grouping to apply to the observations
    :param variable_name: Name of the plotted variable
    :param group_variable_name: Name of the grouping variable
    :return: Bytes plot
    """
    df = DataFrame({"variable": array(x), "group": array(x_group)}, copy=True)

    if (len(df["variable"].value_counts()) > 15) & (df["variable"].dtypes != "object"):
        plot_data = df[["variable", "group"]].groupby("group", as_index=False).agg(median=("variable", "median"))
        plot = (
            plotnine.ggplot(plot_data, plotnine.aes(x="group", y="median"))
            + plotnine.geom_bar(stat="identity")
            + plotnine.ggtitle(f"Median value of {variable_name} variable grouped by {group_variable_name}")
            + plotnine.scale_fill_manual(values=yc_colors())
            + theme_yc()
        )
    else:
        df["variable"] = "response_" + df["variable"].astype("str")
        plot_data = (
            df[["variable", "group"]]
            .assign(nb_pop_cluster=lambda x: x.groupby("group")["variable"].transform("count"))
            .groupby(["variable", "group"], as_index=False)
            .agg(count=("variable", "count"), nb_pop_cluster=("nb_pop_cluster", "first"))
            .assign(proportion_cluster=lambda x: x["count"] / x["nb_pop_cluster"])
        )
        plot = (
            plotnine.ggplot(
                plot_data,
                plotnine.aes(x="group", y="proportion_cluster", fill="variable"),
            )
            + plotnine.geom_bar(stat="identity", position=plotnine.position_dodge())
            + plotnine.ggtitle(f"Description of {variable_name} variable grouped by {group_variable_name}")
            + plotnine.scale_fill_manual(values=yc_colors())
            + theme_yc()
        )
    figfile = BytesIO()
    plotnine.ggsave(plot, figfile, format="png", units="cm", height=14, width=24)
    bytes_plot = figfile.getvalue()
    plt.close("all")
    return bytes_plot
