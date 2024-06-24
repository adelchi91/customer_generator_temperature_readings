import sys
import os
import pickle
import cloudpickle
import yaml
from typing import TYPE_CHECKING
from pandas import options, DataFrame, concat, melt, notna, isna
from local_libraries import score_bands
from yc_younipy.query import OmniscientQuery
from google.cloud import bigquery
from pathlib import Path
import datetime

import seaborn as sns
import matplotlib.pyplot as plt
from waterfall_chart import plot as waterfall

import numpy as np

if TYPE_CHECKING:
    from typing import List, Optional, Tuple

    from pandas.core.indexes.base import Index
bq_runner = OmniscientQuery()

options.display.max_columns = 40
options.display.width = 1000

if len(sys.argv) != 5:
    sys.stderr.write("Arguments error. Usage:\n")
    sys.stderr.write(
        "\tpython stats_model.py training_dataset_path validation_dataset_path pipeline_path variables_file")
    sys.exit(1)

# parameter
filter_only_consistent = True

# Load files
training_dataset_path = sys.argv[1]
validation_dataset_path = sys.argv[2]
pipeline_path = sys.argv[3]
variables_file = sys.argv[4]

# Load files
df_train = pickle.load(open(training_dataset_path, "rb"))
df_val = pickle.load(open(validation_dataset_path, "rb"))
df_full = concat([df_train, df_val], axis=0)
pipeline = cloudpickle.load(open(pipeline_path, "rb"))
model = pipeline  # ['clf']

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
cat_feature = all_variables['cat_feature']

df_full[target] = df_full[target].astype('int')  # df_train[target] is here of type int64, which causes problems to
# scikit learn. I have to convert it in an int

variables_bf_pp = []
for v_set in variables_set_bf_pp:
    variables_bf_pp += all_variables[v_set]

os.makedirs(os.path.join("data", "replay"), exist_ok=True)


def consistency_check(df):
    df = df[lambda x: x['equifax_rating'].notna()]
    # condition_is_consistency = ((df['aggregation_nb_months'] >= 2) &
    #                             (df['monthly_nb_transactions_income'] >= 0.7) &
    #                             (df['monthly_nb_transactions'] >= 15))
    # df['is_consistency'] = df.apply(lambda x: True if condition_is_consistency.loc[x.name] else False, axis=1)
    df.loc[df['normalized_consistency_score'] == 10000, 'is_consistent'] = 'consistent'
    df.loc[df['normalized_consistency_score'] != 10000, 'is_consistent'] = 'inconsistent'
    return df  # df[df['normalized_consistency_score'] == 10000] # df[df['is_consistency']]


def preprocessing(df):
    # apply filters of consistency
    df = consistency_check(df)
    if filter_only_consistent:
        df = df[df['normalized_consistency_score'] == 10000].copy()
    # adding verfiied is homeowner variable as in prepreocess
    df["verified_is_homeowner"] = df["declared_housing_code"].str.startswith(
        "HOME_OWNERSHIP_")
    return df


def runoff_and_uwr_rejection(training_data_raw):
    # define new column
    training_data_raw['score_category_detailed'] = training_data_raw['score_category_declared'].copy()
    # Define a mapping of preapproval_reason values to score_category_detailed values
    training_data_raw.loc[
        training_data_raw[
            'preapproval_reason'] == 'RejectionIneligibility', 'score_category_detailed'] = 'RejectionIneligibility'
    training_data_raw.loc[
        training_data_raw[
            'preapproval_reason'] == 'PricingIneligibility', 'score_category_detailed'] = 'PricingIneligibility'
    training_data_raw.loc[
        training_data_raw[
            'preapproval_reason'] == 'EquifaxIneligibility', 'score_category_detailed'] = 'EquifaxIneligibility'
    #
    training_data_raw['score_category_detailed'] = training_data_raw['score_category_detailed'].replace(['REJECTED'],
                                                                                                        'Run-off').replace(
        ['Uncategorized'], 'uwr_rejection')
    # column score_category_declared
    # Define a mapping of preapproval_reason values to score_category_declared values
    training_data_raw.loc[
        training_data_raw[
            'preapproval_reason'] == 'RejectionIneligibility', 'score_category_declared'] = 'uwr_rejection'
    training_data_raw.loc[
        training_data_raw['preapproval_reason'] == 'PricingIneligibility', 'score_category_declared'] = 'Run-off'
    training_data_raw.loc[
        training_data_raw['preapproval_reason'] == 'EquifaxIneligibility', 'score_category_declared'] = 'Run-off'
    #
    training_data_raw['score_category_declared'] = training_data_raw['score_category_declared'].replace(['REJECTED'],
                                                                                                        'Run-off').replace(
        ['Uncategorized'], 'uwr_rejection')
    return training_data_raw


def data_loader_recent_demands():
    # loading recent demands population from sql query
    print('Loading prod of interest from big query')
    training_data_load_query = (Path(f"src/sql/load_prod_of_interest.sql").read_text())
    training_data_load_query_last_months = (Path(f"src/sql/load_prod_of_interest_last_months.sql").read_text())
    training_data_raw = bq_runner.execute(training_data_load_query, "bigquery yc-data-science")
    training_data_raw_last_months = bq_runner.execute(training_data_load_query_last_months, "bigquery yc-data-science")
    training_data_raw = concat([training_data_raw[lambda x: x.application_date<='2023-04-01'], training_data_raw_last_months], axis=0)
    # preprocessing
    training_data_raw = preprocessing(training_data_raw)
    training_data_raw = runoff_and_uwr_rejection(training_data_raw)

    # Correct figures for UWR-rejection and Run-off, as seen on dashboard
    print('These are the figures on the dashboard, verified by Luca')
    print(training_data_raw.score_category_declared.value_counts(dropna=False, normalize=True))
    # filtering data on scope of interest
    # ConsumerCredit is Autogrant, we do not want to include it.
    df_preprocessed = training_data_raw[lambda x: (x.equifax_rating.notna()) &
                                                  (x.partner_code == 'YOUNITED')
                                                  # (x.product_type != 'ConsumerCredit') &
                                                  # (x.score_category_declared != 'uwr_rejection')
    ]
    return df_preprocessed


def computation_score_bands_applied_on_recent_demands(df_full, df_preprocessed):
    # Computing score bands on the totality of training dataset, i.e. df_full
    optb, table = score_bands.compute_scoreband_binning_table(model, df_full[form_variables + cat_feature],
                                                              df_full[target])
    print(table.drop("Totals").drop(["Count (%)", "WoE", "IV", "JS"], axis=1))

    # Computing score bands on recent population, i.e. past 2023-01-01, using table and optb computed on df_full
    # recommended_category is the score band computed
    # score_category_declared is the score band on the model V5.1 in production - see SQL query
    df_preprocessed['recommended_category'] = score_bands.compute_scoreband(
        model.predict_proba(df_preprocessed[form_variables + cat_feature])[:, 1], optb, table).values
    df_preprocessed['score_v6_computed'] = model.predict_proba(df_preprocessed[form_variables + cat_feature])[:, 1]
    # A7 and A8 are considered as Runoffs
    df_preprocessed['recommended_category'] = df_preprocessed['recommended_category'].replace(['A8'], 'Run-off')
    # we replace recommended_category, aka V6  values, by is_rejected_by_uwrs if value is 'uwr_rejection', which should correspond
    # to uwrs changes in 2023
    df_preprocessed['recommended_category'] = df_preprocessed.apply(
        lambda row: row['is_rejected_by_uwrs'] if row['is_rejected_by_uwrs'] == 'uwr_rejection' else row[
            'recommended_category'], axis=1)
    # we add new UWRs on PRESTALo business provider code
    condition = (lambda x: (x['business_provider_code'] == 'PRESTALO') & (x['recommended_category'].isin(['A6', 'A7'])))
    df_preprocessed.loc[condition, 'recommended_category'] = 'uwr_rejection'
    return df_preprocessed


# recent demands with appropriate filters
df_preprocessed = data_loader_recent_demands()
# score bands calculation on recent demands - addition of recommended_category and score_v6_computed columns
df_preprocessed = computation_score_bands_applied_on_recent_demands(df_full, df_preprocessed)
print(df_preprocessed.application_date.min())
print(df_preprocessed.application_date.max())
df_preprocessed.to_csv(os.path.join("data", "replay", 'df_replay.csv'))



dataset_to_push = df_preprocessed.copy()
dataset_to_push['score_'] = (1 - dataset_to_push['score_v6_computed'])*10000
# push to BigQuery
bq_client = bigquery.Client(project="yuc-pr-risk")
job = bq_client.load_table_from_dataframe(dataset_to_push, f"score_replay_sent_from_datascience.credit_score_ES_v6_full_2_{datetime.date.today()}",
                                          job_config=bigquery.LoadJobConfig(write_disposition="WRITE_TRUNCATE"))