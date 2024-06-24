import sys
import os
import pickle
import cloudpickle
import yaml
from typing import TYPE_CHECKING
from pandas import options, DataFrame, concat, Series
from matplotlib.ticker import FuncFormatter

from local_libraries import calibration_cpm_graphs
from local_libraries import buckets_creation
from local_libraries.preprocessing_methods import compute_age_wrapper
import numpy as np
import pandas as pd
import plotnine as p9
import numpy as np
import matplotlib
from yc_younipy.preprocessing.column_transformer_df import ColumnTransformerDF
from yc_younipy.preprocessing.most_frequent_binning import MostFrequentBinning
from yc_younipy.query import OmniscientQuery
from sklearn.metrics import roc_auc_score
from yc_younipy.preprocessing.quantile_binning import QuantileBinning
import seaborn as sns
import matplotlib.pyplot as plt

if TYPE_CHECKING:
    from typing import List, Optional, Tuple

    from pandas.core.indexes.base import Index

options.display.max_columns = 40
options.display.width = 1000

if len(sys.argv) != 5:
    sys.stderr.write("Arguments error. Usage:\n")
    sys.stderr.write(
        "\tpython stats_model.py training_dataset_path validation_dataset_path pipeline_path variables_file")
    sys.exit(1)

# Load files
training_dataset_path = sys.argv[1]
validation_dataset_path = sys.argv[2]
pipeline_path = sys.argv[3]
variables_file = sys.argv[4]

# Load files
df_train = pickle.load(open(training_dataset_path, "rb"))
df_val = pickle.load(open(validation_dataset_path, "rb"))
pipeline = cloudpickle.load(open(pipeline_path, "rb"))

# Load params
params = yaml.safe_load(open("params.yaml"))
target = params['general']['target']
add_shap_stat = params['stats_model']['add_shap_stat']
model_type = params['general']['model_type']
variables_set_bf_pp = params['general']['variables_set_before_pp']

# Load variables set
all_variables = yaml.safe_load(open(variables_file))
form_variables = all_variables['form']
preprocess_variables = all_variables['preprocessing_use_only']

variables_bf_pp = []
for v_set in variables_set_bf_pp:
    variables_bf_pp += all_variables[v_set]

# predict proba
train_predictions = pipeline.predict_proba(df_train.drop(columns=[target]))[:, 1]
if len(df_val) > 0:
    val_predictions = pipeline.predict_proba(df_val.drop(columns=[target]))[:, 1]
else:
    val_predictions = Series()

# old_version = '1'
# new_version = '2'
df_train = df_train.assign(dataframe='train_test', predict_prob=train_predictions,
                           predict_score=(1 - train_predictions) * 10000)
df_val = df_val.assign(dataframe='validation', predict_prob=val_predictions,
                       predict_score=(1 - val_predictions) * 10000)
df_full = concat([df_train, df_val], axis=0)

#####
data_replay_piv = (df_full
                   .melt(
    id_vars=["contract_reference", "predict_score"],
    value_vars=["dn2_6", "dn3_12"],
    var_name="indicator_type", value_name="indicator_value")
                   .melt(id_vars=["contract_reference", "indicator_type", "indicator_value"],
                         value_vars=["predict_score"],
                         var_name="score_type", value_name="score_value"))


def agg_fun(x):
    return (pd.DataFrame.from_records({"count": [len(x)],
                                       "mean_indicator_rate": [np.mean(x["indicator_value"])],
                                       "n_default": [np.sum(x["indicator_value"])],
                                       "gini": [2 * roc_auc_score(-x["indicator_value"], x["score_value"]) - 1]}))


# data_replay_pref.to_csv("figures_n_tables/data_replay_pref.csv", sep=";")

data_replay_calib = (data_replay_piv
                     .assign(
    score_bands=lambda x: QuantileBinning(nb_bins=6, output_labels=True).fit_transform(x.loc[:, ["score_value"]]),
    pred_value=lambda x: 1 - x["score_value"] / 10000)
                     .groupby(["score_bands", "indicator_type", "score_type"], as_index=False)
                     .agg(count=("contract_reference", "count"),
                          mean_indicator_value=("indicator_value", "mean"),
                          mean_pred_value=("pred_value", "mean")))

data_replay_avg_score = (df_full
                         .assign(
    cohort=lambda x: pd.to_datetime(x["application_date"]).dt.to_period("Q").dt.to_timestamp(),
    target_score=lambda x: 10000 * (1 - x["dn3_12"]))
                         .melt(id_vars=["contract_reference", "cohort"],
                               value_vars=["predict_score",
                                           "target_score"],
                               var_name="score_type", value_name="score_value")
                         .groupby(["cohort", "score_type"], as_index=False)
                         .agg(avg_score_value=("score_value", "mean"))
                         .assign(avg_risk_rate=lambda x: 1 - x["avg_score_value"] / 10000)
                         )

df_predict_score = df_full.assign(
    cohort=lambda x: pd.to_datetime(x["application_date"]).dt.to_period("Q").dt.to_timestamp(),
    score_type="predict_score",
    score_value=lambda x: x["predict_score"],
    avg_score_value=lambda x: x["predict_score"],
    avg_risk_rate=lambda x: 1 - x["predict_score"] / 10000
)[["contract_reference", "cohort", "score_type", "score_value", "avg_score_value", "avg_risk_rate"]]

df_target_score = df_full.assign(
    cohort=lambda x: pd.to_datetime(x["application_date"]).dt.to_period("Q").dt.to_timestamp(),
    score_type="target_score",
    score_value=lambda x: 10000 * (1 - x["dn3_12"]),
    avg_score_value=lambda x: 10000 * (1 - x["dn3_12"]),
    avg_risk_rate=lambda x: 1 - 10000 * (1 - x["dn3_12"]) / 10000
)[["contract_reference", "cohort", "score_type", "score_value", "avg_score_value", "avg_risk_rate"]]

# df_target_score = df_full.assign(
#     cohort=lambda x: pd.to_datetime(x["application_date"]).dt.to_period("Q").dt.to_timestamp())

# plot = (
#         p9.ggplot(data_replay_avg_score, p9.aes(x="cohort", y="avg_risk_rate", color="score_type"))
#         #+ p9.geom_line(p9.aes(linetype="sample_v4_2"), size=1)
#         + p9.geom_point(size=1)
#         + p9.theme(axis_text_x=p9.element_text(angle=20, hjust=1),
#                    figure_size=(16, 8))
# )
# sns.lineplot(data=data_replay_avg_score,x="cohort", y="avg_risk_rate", hue="score_type")

# Save outputs
os.makedirs(os.path.join("figures", "temporal_calibration"), exist_ok=True)
# separate the two plots, use point plot and do a boostrap to add error bars

fig, ax = plt.subplots()
# Set seaborn style
sns.set_style("whitegrid")

# Create plot
# sns.pointplot(data=data_replay_avg_score[lambda x: x.score_type == 'predict_score'], x="cohort", y="avg_risk_rate",
#               ci=95, ax=ax, color=palette[0], label='Predict Score')
# sns.pointplot(data=data_replay_avg_score[lambda x: x.score_type == 'target_score'], x="cohort", y="avg_risk_rate",
#              ci=95, ax=ax, color=palette[1], label='Target Score')

# sns.lineplot(data=df_full, x="cohort", y="target_score",
#              ci=95, ax=ax, color='blue', label='Predict Score', err_style='bars')

sns.lineplot(data=df_target_score, x="cohort", y="avg_risk_rate",
             ci=95, ax=ax, color='blue', label='Target Score')

sns.lineplot(data=df_predict_score, x="cohort", y="avg_risk_rate",
             ci=None, ax=ax, color='red', label='Predict Score')

# Plot the second lineplot with 'target_score' score type and custom legend label
# sns.lineplot(data=data_replay_avg_score[lambda x: x.score_type == 'target_score'], x="cohort", y="avg_risk_rate",
#              ci=95, ax=ax, color='orange', label='Target Score', err_style='bars')

# ax.fill_between(data=data_replay_avg_score, x="cohort", y1="ci_low", y2="ci_high", alpha=0.2)

#
# Add grid
ax.grid(True, linestyle='--', axis='y', color='gray', alpha=0.5)



# Set plot title and axis labels
ax.set_title('Average Risk Rate by Cohort and Score Type')
ax.set_xlabel('Dates')
ax.set_ylabel('Average Risk Rate')

# Adjust legend position
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

# Rotate x-axis labels by 45 degrees
plt.xticks(rotation=45)

# Add tight layout
plt.tight_layout()
plt.savefig(os.path.join("figures", "temporal_calibration", 'temporal_calibration.png'), dpi=600, bbox_inches='tight')
# Show plot
plt.show()

# p9.ggsave(plot, f"figures_n_tables/data_replay_avg_score.png", format="png", units="cm", height=20, width=25, dpi=200)
