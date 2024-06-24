from io import BytesIO
from typing import Union

import matplotlib.pyplot as plt
import plotnine
from numpy import array
from pandas import DataFrame, Series, to_datetime

from yc_younipy.plot import theme_yc, yc_colors


def _create_pop_over_time_data(
    x: Union[Series, array], x_date: Union[Series, array], date_cohort_type: str = "D"
) -> DataFrame:
    """
    Create pop over time data
    :param x: Observations to plot
    :param x_date: Corresponding date values
    :param date_cohort_type: Type of period to adopt during plotting. Example "D", "5D", "W", "M"...
    :return: Infos over time regarding the studied variable (number of observations, proportion of observations)
    """

    df = DataFrame({"variable": array(x), "date": array(x_date)})

    df["date"] = to_datetime(df["date"]).dt.to_period(date_cohort_type).dt.to_timestamp()

    df_agg = (
        df.assign(n_contract_all=lambda x: x.groupby(["date"])["variable"].transform("count"))
        .groupby(["date", "variable"], as_index=False, observed=True)
        .agg(n_contract=("variable", "count"), n_contract_all=("n_contract_all", "first"))
        .assign(prop_contrat=lambda x: x["n_contract"] / x["n_contract_all"])
    )

    variable_values = DataFrame({"variable": df_agg["variable"].unique()})
    date_values = DataFrame({"date": df_agg["date"].unique()})

    return (
        date_values.join(variable_values, how="cross")
        .merge(df_agg, left_on=["date", "variable"], right_on=["date", "variable"], how="left")
        .fillna(0)
    )


def plot_pop_over_time(
    x: Union[Series, array], x_date: Union[Series, array], date_cohort_type: str, variable_name: str
) -> bytes:
    """
    Create pop over time data
    :param x: Observations to plot.
    :param x_date: Corresponding date values.
    :param date_cohort_type: Type of period to adopt during plotting. Example "D", "5D", "W", "M"...
    :param variable_name: Name of the plotted variable.
    :return: Bytes plot
    """
    df = _create_pop_over_time_data(x, x_date, date_cohort_type)

    print(df)

    n_modalities = len(df["variable"].unique())
    df["variable"] = df["variable"].astype(str)
    if n_modalities > 25:
        print("Categorical variable has more than 25 modalities ! This plot has been canceled.")
        return None
    else:
        plot = (
            plotnine.ggplot(df, plotnine.aes(x="date", y="n_contract", fill="variable"))
            + plotnine.geom_area(position="stack")
            + plotnine.scale_fill_manual(values=yc_colors(range(n_modalities)))
            + theme_yc()
            + plotnine.theme(
                legend_title=plotnine.element_blank(),
                axis_title_y=plotnine.element_blank(),
                legend_position="right",
                axis_text_x=plotnine.element_text(angle=20, hjust=1),
            )
            + plotnine.labs(title=f"{variable_name} w.r.t time")
        )

        figfile = BytesIO()
        plt.pause(1e-13)  # hack to prevent Tinker exceptions poping ...
        plotnine.ggsave(plot, figfile, format="png", units="cm", height=14, width=24)
        bytes_plot = figfile.getvalue()
        plt.close("all")
        return bytes_plot
