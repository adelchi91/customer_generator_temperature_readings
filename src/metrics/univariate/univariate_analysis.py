from io import BytesIO
from typing import Union

import matplotlib.pyplot as plt
import plotnine
from numpy import array
from pandas import DataFrame, Series, isnull, to_datetime
from statsmodels.stats.proportion import proportion_confint

from yc_younipy.plot import scales_percent, theme_yc, yc_colors


def _create_univariate_data_table(
    x: Union[Series, array], y: Union[Series, array], missing_value: str, x_group: Union[Series, array, None]
) -> DataFrame:
    """
    Create univariate analysis table
    :param x: Variable values
    :param y: Target values
    :param missing_value: Category to be considered as missing
    :param x_group: Grouping values
    :return: A DataFrame with number of observations, number of target, target rate, and confidence interval
    """
    df = DataFrame({"variable": array(x), "target": array(y), "color_group": array(x_group)}, copy=True)

    if x_group is not None:
        group_by = ["variable", "color_group"]

        def group_fn(x):
            return (x["variable"].astype(str).str.contains(missing_value)).astype(str) + x["color_group"]

    else:
        group_by = ["variable"]

        def group_fn(x):
            return (x["variable"].astype(str).str.contains(missing_value)).astype(str)

    df_agg = (
        df.groupby(group_by, as_index=False, observed=True)
        .agg(
            number_of_contract=("variable", "count"),
            number_of_target=("target", "sum"),
        )
        .assign(target_rate=lambda x: x["number_of_target"] / x["number_of_contract"], line_plot_group=group_fn)
    )

    (df_agg["target_rate_lower_ci"], df_agg["target_rate_upper_ci"],) = proportion_confint(
        df_agg["number_of_target"],
        df_agg["number_of_contract"],
        alpha=0.15,
        method="normal",
    )

    return df_agg


def plot_univariate_analysis(
    x: Union[Series, array], y: Union[Series, array], missing_value: str, x_group: Union[Series, array, None] = None
) -> bytes:
    """
    Plot univariate analysis for a given variable and target
    :param x: Variable values
    :param y: Target values
    :param missing_value: String to be considered as a missing value
    :param x_group: Grouping variable
    :return: A bytes plot
    """
    df_tmp = _create_univariate_data_table(x, y, missing_value, x_group)

    y_max = df_tmp["target_rate_upper_ci"].max()

    if df_tmp["variable"].nunique() > 25:
        print(f"Categorical variable {'variable'} has more than 25 modalities ! This plot has been canceled.")
        return None
    else:

        group_color_kwarg = {"color": "color_group"} if x_group is not None else {}
        unique_color_kwarg = {} if x_group is not None else {"color": "#9163f2"}

        plot = (
            plotnine.ggplot(df_tmp)
            + plotnine.geom_line(
                plotnine.aes(x="variable", y="target_rate", group="line_plot_group"),
                **unique_color_kwarg,
                size=0.15,
                position=plotnine.position_dodge(width=0.5),
            )
            + plotnine.geom_errorbar(
                plotnine.aes(
                    x="variable",
                    ymin="target_rate_lower_ci",
                    ymax="target_rate_upper_ci",
                    group="line_plot_group",
                    **group_color_kwarg,
                ),
                **unique_color_kwarg,
                size=0.3,
                width=0.01,
                position=plotnine.position_dodge(width=0.5),
            )
            + plotnine.geom_point(
                plotnine.aes(x="variable", y="target_rate", group="line_plot_group", **group_color_kwarg),
                **unique_color_kwarg,
                position=plotnine.position_dodge(width=0.5),
            )
            + plotnine.scale_color_manual(values=yc_colors())
            + theme_yc()
            + plotnine.coord_cartesian(ylim=(0, y_max))
            + plotnine.theme(
                legend_position="right",
                legend_title=plotnine.element_blank(),
                axis_text_x=plotnine.element_text(angle=7, hjust=1),
            )
            + plotnine.scale_y_continuous(labels=scales_percent, expand=[0, 0])
        )

        figfile = BytesIO()
        plotnine.ggsave(plot, figfile, format="png", units="cm", height=14, width=24)
        bytes_plot = figfile.getvalue()
        plt.close("all")
        return bytes_plot


def _create_target_rate_over_time(
    x: Union[Series, array], y: Union[Series, array], date_cohort_type: str, x_group: Union[Series, array, None]
) -> DataFrame:
    """
    Create a table representing the target rate over time
    :param x: Date values
    :param y: Target values
    :param date_cohort_type: Type of time period to be used
    :param x_group: Grouping variable
    :return: A DataFrame with the target rate over time
    """
    try:
        assert not any(isnull(x))
        assert not any(isnull(y))
        assert (x_group is None) or (not any(isnull(x_group)))
    except Exception:
        raise ValueError("The data provided must not contain null values")

    df = DataFrame({"raw_date": array(x), "target": array(y), "group": array(x_group)}, copy=True)

    df["date"] = to_datetime(df["raw_date"]).dt.to_period(date_cohort_type).dt.to_timestamp()

    if x_group is not None:
        group_by = ["date", "group"]
    else:
        group_by = ["date"]

    df_agg = (
        df.groupby(group_by, as_index=False, observed=True)
        .agg(
            number_of_contract=("raw_date", "count"),
            number_of_target=("target", "sum"),
        )
        .assign(target_rate=lambda x: x["number_of_target"] / x["number_of_contract"])
    )

    return df_agg


def plot_target_rate_over_time(
    x: Union[Series, array], y: Union[Series, array], date_cohort_type: str, x_group: Union[Series, array, None] = None
) -> bytes:
    """
    Create a plot representing the target rate over time
    :param x: Date values
    :param y: Target values
    :param date_cohort_type: Type of time period to be used
    :param x_group: Grouping variable
    :return: Bytes plot
    """
    df = _create_target_rate_over_time(x, y, date_cohort_type, x_group)

    group_color_kwarg = {"color": "group"} if x_group is not None else {}
    unique_color_kwarg = {} if x_group is not None else {"color": "#9163f2"}

    plot = (
        plotnine.ggplot(df)
        + plotnine.geom_line(
            plotnine.aes(x="date", y="target_rate", **group_color_kwarg),
            **unique_color_kwarg,
            size=0.15,
            position=plotnine.position_dodge(width=0.5),
        )
        + plotnine.geom_point(
            plotnine.aes(x="date", y="target_rate", **group_color_kwarg),
            **unique_color_kwarg,
            position=plotnine.position_dodge(width=0.5),
        )
        + plotnine.scale_color_manual(values=yc_colors())
        + theme_yc()
        + plotnine.theme(
            legend_position="right",
            legend_title=plotnine.element_blank(),
            axis_text_x=plotnine.element_text(angle=7, hjust=1),
        )
        + plotnine.scale_y_continuous(labels=scales_percent, expand=[0, 0])
    )

    figfile = BytesIO()
    plotnine.ggsave(plot, figfile, format="png", units="cm", height=14, width=24)
    bytes_plot = figfile.getvalue()
    plt.close("all")
    return bytes_plot
