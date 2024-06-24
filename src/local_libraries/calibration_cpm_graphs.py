import joblib
import matplotlib.pyplot as plt
import optbinning
import pathlib
import seaborn as sns
import sklearn.compose
import sys
from sklearn.preprocessing import KBinsDiscretizer
from yc_younipy.preprocessing.most_frequent_binning import MostFrequentBinning
from yc_younipy.preprocessing.quantile_binning import QuantileBinning
from local_libraries.preprocessing_methods import compute_age_wrapper
import numpy as np
import os
import pandas as pd
from scipy.stats import beta
from pandas.api.types import (
    is_categorical_dtype,
    is_datetime64_any_dtype,
    is_numeric_dtype,
    is_object_dtype,
)


#sys.path.append("./src/pipeline_files")

def plot_cpm_graphs(X, target_data, old_version, new_version):
    sklearn.set_config(transform_output="pandas")
    data = pathlib.Path("data")
    models_directory = data / "models"
    figures_directory = pathlib.Path("figures") / "score_dependencies"
    figures_directory.mkdir(exist_ok=True)

    target = "dn3_12"
    X["personal_age"] = compute_age_wrapper(X["declared_date_of_birth"],
                                                                       X["application_date"]).astype(np.float32)


    bins = 5


    discretizer = KBinsDiscretizer(bins, encode="ordinal")
    X[new_version] = X.predict_prob.copy()
    X[new_version] = discretizer.fit_transform(X[[new_version]])[new_version]


    # def binarize_age(age):
    #     if age < 25:
    #         return "[18; 25["
    #     if age < 35:
    #         return "[25; 35["
    #     if age < 45:
    #         return "[35; 45["
    #     if age < 55:
    #         return "[45; 55["
    #     if age < 65:
    #         return "[55; 65["
    #     return "[65+]"

    def binarize_age(age):
        #age_ranges = [(25, "[18; 25["), (35, "[25; 35["), (45, "[35; 45["), (55, "[45; 55["), (65, "[55; 65[")]
        age_ranges = [(30, "[18; 30["), (45, "[30; 45["), (55, "[45; 55["), (65, "[55; 65[")]
        for cutoff, label in age_ranges:
            if age < cutoff:
                return label
        return "[65+]"


    X["personal_age"] = X["personal_age"].apply(binarize_age)
    #X["bank_age"] = X["bank_age"].round(0)
    X["mortgage_amount"] = X["mortgage_amount"].round(0)

    versions = (
        new_version,
    )
    variables = [
        #("bank_age", "quantile", {"nb_bins": 4, "output_labels": True}),

        #("verified_bank_code", "most_frequent", {"top_n": 4, "output_labels": True}),
        ("verified_bank_code", "optbinning", {"dtype": "categorical"}),
        # ("verified_bank_code", None, None),
        ("project_type_code", "optbinning", {"dtype": "categorical"}),
        ("marital_status_code", "optbinning", {"dtype": "categorical"}),
        ("business_provider_code", "most_frequent", {"top_n": 4, "output_labels": True}),
        #("business_provider_code", None, None),
        ("verified_housing_code", None, None),
        ("main_net_monthly_income", "optbinning", {"monotonic_trend": "auto_asc_desc"}),

        #("marital_status_code", None, None),



        ("mortgage_amount", "optbinning", {"monotonic_trend": "auto_asc_desc"}),

        #("ongoing_credits_amount", "optbinning", {"monotonic_trend": "auto_asc_desc"}),


        ("personal_age", None, None),
        #("personal_age", "optbinning", {"monotonic_trend": "auto_asc_desc"}),

        # ("phone_prefix", "optbinning", {"dtype": "categorical"}),
        # ("postal_region", None, None),
        #("profession_code", "most_frequent", {"top_n": 10, "output_labels": True}),
        # ("professional_age", "quantile"),
        #("professional_situation_code", None, None),
        #("rent_amount", "optbinning", {}),
        # ("sector_code", None),
    ]

    X['personal_age'] = X['personal_age'].astype('string')

    # for variable, binning_type, kwargs in variables:
    #     print(f"Plotting score dependency for variable {variable}")
    #     if binning_type == "quantile":
    #         X[f"{variable}_bin"] = QuantileBinning(**kwargs).fit_transform(X[[variable]])[:, 0]
    #     elif binning_type == "most_frequent":
    #         X[f"{variable}_bin"] = MostFrequentBinning(**kwargs).fit_transform(X[[variable]])[:, 0]
    #     elif binning_type == "optbinning":
    #         X[f"{variable}_bin"] = optbinning.OptimalBinning(**kwargs).fit_transform(
    #             X[variable], target_data.astype(float), metric="bins"
    #         )
    #     else:
    #         X[f"{variable}_bin"] = X[variable]
    #     fig, axes = plt.subplots(1, len(versions), figsize=(20, 5))
    #
    #     for v, index in enumerate(versions):
    #         sns.pointplot(data=X, x=index, y=target, hue=f"{variable}_bin", n_boot=1)
    #         plt.tight_layout()
    #         os.makedirs(os.path.join("figures", "cpm_graphs"), exist_ok=True)
    #         plt.savefig(os.path.join("figures", "cpm_graphs", f"score_dependency{variable}.png"), dpi=600,
    #                     bbox_inches='tight')
    #     plt.show()

    for variable, binning_type, kwargs in variables:
        print(f"Plotting score dependency for variable {variable}")
        index = X.columns.get_loc(variable)
        if binning_type == "quantile":
            X[f"{variable}_bin"] = QuantileBinning(**kwargs).fit_transform(X.iloc[:, index])[:, 0]
        elif binning_type == "most_frequent":
            X[f"{variable}_bin"] = MostFrequentBinning(**kwargs).fit_transform(X[[variable]])
        elif binning_type == "optbinning":
            X[f"{variable}_bin"] = optbinning.OptimalBinning(**kwargs).fit_transform(
                X.iloc[:, index], target_data.astype(float), metric="bins"
            )
        else:
            X[f"{variable}_bin"] = X.iloc[:, index]
        fig, axes = plt.subplots(1, len(versions), figsize=(20, 5))

        for v, index in enumerate(versions):
            sns.pointplot(data=X, x=index, y=target, hue=f"{variable}_bin", n_boot=1000, dodge=True)
            plt.axhline(y=0.10, linestyle='--', color='k')
            plt.tight_layout()
            os.makedirs(os.path.join("figures", "cpm_graphs"), exist_ok=True)
            plt.savefig(os.path.join("figures", "cpm_graphs", f"score_dependency{variable}.png"), dpi=600,
                        bbox_inches='tight')
        plt.show()

    return


def calibration_by_feature(
        df_: pd.DataFrame,
        feature_name: str,
        prediction_name: str,
        event_name: str,
        pct: float = 0.9,
        n: int = 5,
        bonferonni_correction: bool = True
) -> pd.DataFrame:
    """
    Print the calibration test of a prediction for a feature with a confidence interval.
    :param event_name: name of the event, e.g. dn3_12
    :param prediction_name: name of the prediction
    :param df_: dataframe with the feature 'feature_name', the score 'prediction_name' and the event 'event_name'
    :param feature_name: feature used to split the population
    :param pct: optional confidence interval, default value equals 0.90
    :param bonferonni_correction: optional to correct the probability in function of the number of tests performed
    :param n: max number of splits, default value equals 5
    :return: dataframe with test results of each modality of 'feature_name'
    """
    d = df_[[feature_name, prediction_name, event_name]].copy()

    if is_numeric_dtype(d[feature_name]) and len(d[feature_name].unique()) > n:
        d[feature_name] = pd.qcut(d[feature_name], n, duplicates="drop").astype(str)

    n_test = min(n, len(d[feature_name].unique()))
    pct = 1 - (1 - pct) / n_test if bonferonni_correction else pct

    d = d.groupby(feature_name) \
        .agg(
        n_obs=(prediction_name, 'count'),
        avg_pred=(prediction_name, 'mean'),
        avg_dr=(event_name, 'mean')) \
        .sort_values(by="n_obs", ascending=False) \
        .head(n) \
        .reset_index()

    d = d[d["n_obs"] > 0]

    d["a"] = d["avg_pred"] * d["n_obs"]
    d["b"] = (1 - d["avg_pred"]) * d["n_obs"]
    d["lower_bound"] = d.apply(lambda x: beta.ppf((1 - pct) / 2, x.a, x.b + 1), axis=1)
    d["upper_bound"] = d.apply(lambda x: beta.ppf((1 + pct) / 2, x.a + 1, x.b), axis=1)
    d["success"] = (d["lower_bound"] <= d["avg_dr"]) & (d["avg_dr"] <= d["upper_bound"])

    d.drop(columns={'a', 'b'}, inplace=True)
    d.insert(0, "feature_name", feature_name)
    d.rename(columns={feature_name: "value"}, inplace=True)
    d["value"] = d["value"].astype(str)

    d["avg_pred"] = (d["avg_pred"] * 100).map('{:.2f}'.format) + "%"
    d["avg_dr"] = (d["avg_dr"] * 100).map('{:.2f}'.format) + "%"
    d["lower_bound"] = (d["lower_bound"] * 100).map('{:.2f}'.format) + "%"
    d["upper_bound"] = (d["upper_bound"] * 100).map('{:.2f}'.format) + "%"

    return d.set_index(["feature_name", "value"])