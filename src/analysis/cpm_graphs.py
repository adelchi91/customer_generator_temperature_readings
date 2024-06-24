import sys
import os
import pickle
import cloudpickle
import yaml
from typing import TYPE_CHECKING
from pandas import options, DataFrame, concat, Series, merge
from pathlib import Path

from local_libraries import calibration_cpm_graphs
from local_libraries import buckets_creation
from local_libraries.preprocessing_methods import compute_age_wrapper
import numpy as np
from yc_younipy.query import OmniscientQuery
from yc_younipy.metrics.univariate.univariate_analysis import (
    plot_target_rate_over_time,
    plot_univariate_analysis,
)

bq_runner = OmniscientQuery()

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

# loading is_repeat_business
print('Loading demands population from big query')
df_is_repeat_business_query = (Path(f"src/sql/load_csm_table.sql").read_text())
df_is_repeat_business = bq_runner.execute(df_is_repeat_business_query, "bigquery yc-data-science")

variables_bf_pp = []
for v_set in variables_set_bf_pp:
    variables_bf_pp += all_variables[v_set]

# predict proba
train_predictions = pipeline.predict_proba(df_train.drop(columns=[target]))[:, 1]
if len(df_val) > 0:
    val_predictions = pipeline.predict_proba(df_val.drop(columns=[target]))[:, 1]
else:
    val_predictions = Series()

old_version = '1'
new_version = '2'
df_train = df_train.assign(dataframe='train_test', predict_prob=train_predictions)
df_val = df_val.assign(dataframe='validation', predict_prob=val_predictions)
df_full = concat([df_train, df_val], axis=0)
df_full = merge(df_full, df_is_repeat_business[['contract_reference', 'is_repeat_business']], on="contract_reference", how="left").sort_values("application_date")


def convert_dataframe(alt_w_pred):
    alt_w_pred_converted = alt_w_pred.assign(verified_is_homeowner=lambda x: x.verified_is_homeowner.astype('string'),
                                             verified_bank_code=lambda x: x.verified_bank_code.astype('string'),
                                             business_provider_code=lambda x: x.business_provider_code.astype('string'),
                                             verified_housing_code=lambda x: x.verified_housing_code.astype('string'),
                                             personal_age = lambda x: compute_age_wrapper(x.declared_date_of_birth,
                                                                                     x.application_date).astype(np.float32)
    )
    return alt_w_pred_converted

numerical_vars = ['main_net_monthly_income','mortgage_amount', 'personal_age']
categorical_vars = ['verified_is_homeowner', 'verified_bank_code', 'business_provider_code', 'verified_housing_code']


# table calibration
cpm_vars = ["business_provider_code", "is_repeat_business"]
table_outputs = []
for var in cpm_vars:
    output = calibration_cpm_graphs.calibration_by_feature(df_full, feature_name=var,
                                    prediction_name='predict_prob', event_name=target)
    table_outputs.append(output)

df_calibration_table = concat(table_outputs, axis=0)
os.makedirs(os.path.join("data", "cpm_graphs"), exist_ok=True)
df_calibration_table.to_csv(
    os.path.join("data", "cpm_graphs",
                 "variables_calibration.csv"))



#
# output = buckets_creation.binning(df_full_converted[numerical_vars+categorical_vars + ['dn3_12', 'contract_reference']],
#                                   numerical_vars, categorical_vars)
# # Save outputs
# os.makedirs(os.path.join("figures", "stats_univariate_analysis"), exist_ok=True)
# vars_ = output.filter(regex='^num__|^str__', axis=1).columns.to_list()
# output['remainder__' + target] = output['remainder__' + target].astype('float64')
# #for var in vars_:

# CPM graphs
df_full_converted = convert_dataframe(df_full)
calibration_cpm_graphs.plot_cpm_graphs(df_full_converted, df_full_converted[target], old_version, new_version)
print('Hi')

df_full_converted = df_full_converted.assign(has_mortgage= lambda x: x.mortgage_amount > 0)
variables_romain = ['personal_age', 'verified_is_homeowner', 'verified_housing_code', 'marital_status_code', 'main_net_monthly_income', 'has_mortgage']
# df_full analysis per business_provider_code
