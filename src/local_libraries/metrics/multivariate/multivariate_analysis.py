from io import BytesIO

import matplotlib.pyplot as plt
import plotnine
from numpy import array, sqrt, sum
from pandas import DataFrame, Series, concat, crosstab
from scipy.stats import chi2_contingency
from sklearn import preprocessing

from yc_younipy.plot import scales_percent, theme_yc, yc_colors


def cramers_v(X_1: Series, X_2: Series) -> float:
    """
    Compute Cramer's V between two variables
    :param X_1: first dataframe
    :param X_2: second dataframe
    :return: The result of the cramer comparison
    """
    crosstab_array = array(crosstab(X_1, X_2, rownames=None, colnames=None))  # Cross table building
    stat = chi2_contingency(crosstab_array, correction=False)[0]  # Keeping of the test statistic of the Chi2 test
    obs = sum(crosstab_array)  # Number of observations
    mini = min(crosstab_array.shape) - 1  # Take the minimum value between the columns and the rows
    return sqrt((stat / obs) / mini)


def cramers_v_matrix(X: DataFrame) -> DataFrame:
    """
    Compute Cramer's V Matrix between variables
    :param X: input dataframe
    :return: Cramer's v matrix
    """
    variable_names = X.columns
    label = preprocessing.LabelEncoder()
    df_encoded = DataFrame()

    for i in variable_names:
        df_encoded[i] = label.fit_transform(X[i].astype(str).fillna("Missing"))

    cramers_matrix = DataFrame()
    for var1 in df_encoded:
        for var2 in df_encoded:
            cramers = cramers_v(df_encoded[var1], df_encoded[var2])
            s = DataFrame([[var1, var2, cramers]], columns=["var1", "var2", "cramersV"])  # Cramer's V test
            cramers_matrix = concat([cramers_matrix, s], ignore_index=True)

    return cramers_matrix


def plot_cramers_v_matrix(X: DataFrame, title: str) -> bytes:
    """
    Create the Cramer's v matrix as a figure
    :param X: input dataframe
    :param title: name of the graph
    :return: Bytes plot
    """
    df_tmp = cramers_v_matrix(X.copy())

    correlation_plot = (
        plotnine.ggplot(
            df_tmp.assign(cramers_v_percent=lambda x: scales_percent(x["cramersV"])),
            plotnine.aes(x="var1", y="var2", fill="cramersV"),
        )
        + plotnine.geom_tile()
        + plotnine.geom_text(plotnine.aes(label="cramers_v_percent"), size=8)
        + theme_yc()
        + plotnine.theme(axis_text_x=plotnine.element_text(angle=20, hjust=1))
        + plotnine.scale_fill_gradient(low="white", high=yc_colors([1])[0])
        + plotnine.labs(x="", y="", fill="Cramer's V", subtitle=f"Population : {title}")
        + plotnine.theme(legend_position="top")
        + plotnine.guides(fill=False)
    )

    plot_size = len(df_tmp["var1"].unique()) * 0.6 + 14
    figfile = BytesIO()
    plotnine.ggsave(correlation_plot, figfile, format="png", units="cm", height=plot_size - 7, width=plot_size)
    bytes_plot = figfile.getvalue()
    plt.close("all")
    return bytes_plot
