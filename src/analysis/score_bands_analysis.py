import sys
import os
import pickle
import cloudpickle
import yaml
from typing import TYPE_CHECKING
from pandas import options, DataFrame, concat, melt, notna, isna
from local_libraries import score_bands
from yc_younipy.query import OmniscientQuery
from pathlib import Path
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
filter_only_consistent = False

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
    print('Loading demands population from big query')
    training_data_load_query = (Path(f"src/sql/load_demands.sql").read_text())
    training_data_raw = bq_runner.execute(training_data_load_query, "bigquery yc-data-science")
    # preprocessing
    training_data_raw = preprocessing(training_data_raw)
    training_data_raw = runoff_and_uwr_rejection(training_data_raw)

    # Correct figures for UWR-rejection and Run-off, as seen on dashboard
    print('These are the figures on the dashboard, verified by Luca')
    print(training_data_raw.score_category_declared.value_counts(dropna=False, normalize=True))
    # filtering data on scope of interest
    # ConsumerCredit is Autogrant, we do not want to include it.
    df_preprocessed = training_data_raw[lambda x: (x.equifax_rating.notna()) &
                                                  (x.partner_code == 'YOUNITED') &
                                                  (x.product_type != 'ConsumerCredit') &
                                                  (x.score_category_declared != 'uwr_rejection')]
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


def count_plot_migration_comparison(dfv):
    # bar plot
    # Create a new DataFrame that concatenates the two columns
    # df_new = concat([df_preprocessed['score_category_declared'], df_preprocessed['recommended_category']])
    pct = (dfv.groupby(['value', 'variab'
                                 'le']).size() / dfv.groupby(['variable']).size() * 100).reset_index().rename(
        {0: 'percent'},
        axis=1)
    ax = sns.barplot(
        data=pct,
        x="value",
        hue='variable',
        y='percent',
        order=['A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'Run-off', 'uwr_rejection'],
        palette={'V6': 'tab:blue', 'V5': 'tab:orange', 'V5_iso_risk':'tab:pink'}
    )

    # Tilt x-axis labels by 45 degrees
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    # for container in ax.containers:
    #     ax.bar_label(container, fmt='%.f%%')
    for patch in ax.patches:
        x, y = patch.get_x() + patch.get_width() / 2, patch.get_height()
        ax.text(x, y, f"{y:.0f}%", ha='center', va='bottom', fontsize=8)

    print('hey')

    plt.legend(loc='upper left')
    # add labels and legend
    plt.xlabel('Category')
    plt.ylabel('Value')
    # plt.legend()
    os.makedirs(os.path.join("figures", "impact_analysis"), exist_ok=True)
    os.makedirs(os.path.join("data", "impact_analysis"), exist_ok=True)
    plt.savefig(os.path.join("figures", "impact_analysis", 'count_plot_migration_comparison.png'), dpi=600,
                bbox_inches='tight')
    # display the plot
    plt.show()
    return


def transform_dataframe(df_original):
    # Create a new DataFrame to store the transformed data
    df_transformed = DataFrame(columns=['value', 'V5', 'V5_iso_risk', 'V6'])
    # Iterate over the rows in the original DataFrame
    for i in range(len(df_original)):
        row = df_original.iloc[i]
        value = row['value']
        for j in range(i + 1, len(df_original)):
            next_row = df_original.iloc[j]
            next_value = next_row['value']
            if next_value == 'Run-off' or next_value == 'uwr_rejection':
                break
            new_value = value + '-' + next_value
            new_v5 = df_original.loc[i:j, 'V5'].sum()
            new_v5_is_risk = df_original.loc[i:j, 'V5_iso_risk'].sum()
            new_v6 = df_original.loc[i:j, 'V6'].sum()
            new_row = DataFrame([[new_value, new_v5, new_v5_is_risk, new_v6]], columns=['value', 'V5', 'V5_iso_risk', 'V6'])
            df_transformed = concat([df_transformed, new_row])
    # Add the final rows for 'Run-off' and 'uwr_rejection'
    df_transformed = concat([df_transformed, df_original.iloc[-2:]])
    return df_transformed.set_index('value')


def table_volume_score_bands_groupes(dfv):
    # number/volume statistics
    vol = (dfv.groupby(['value', 'variable']).size()).reset_index().rename(
        {0: 'number'},
        axis=1)
    # Group the filtered DataFrame by 'variable' and 'value', and sum the 'number' column
    grouped_df = vol.groupby(['variable', 'value'])['number'].sum().reset_index()
    # Pivot the grouped DataFrame to have 'variable' as columns and 'value' as rows
    pivoted_df = grouped_df.pivot_table(index='value', columns='variable', values='number', fill_value=0)
    pivoted_df.to_csv(os.path.join("data", "impact_analysis", 'volumes_score_bands.csv'))
    df_transformed = transform_dataframe(pivoted_df.reset_index())
    df_transformed.to_csv(os.path.join("data", "impact_analysis", 'volumes_score_bands_grouped.csv'))
    return


def pre_acc_vs_inconsistent_migration_matrices(df_preprocessed):
    # Table for thimothee
    dfv = df_preprocessed[['score_category_detailed', 'recommended_category', 'is_consistent']]
    dfv = dfv.rename(columns={'score_category_detailed': 'V5', 'recommended_category': 'V6'})
    dfv['V5'] = dfv['V5'].replace({'Run-off': 'PricingIneligibility'})
    dfv['V5'] = dfv['V5'].replace(dict.fromkeys(['A2', 'A3', 'A4', 'A5', 'A6', 'A7'], 'Pre-accepted'))
    dfv['V6'] = dfv['V6'].replace({'Run-off': 'PricingIneligibility'})
    dfv['V6'] = dfv['V6'].replace(dict.fromkeys(['A2', 'A3', 'A4', 'A5', 'A6', 'A7'], 'Pre-accepted'))
    # adding consistent inconsistent category to V5 and V6 records columns
    dfv.loc[dfv['is_consistent'] == 'inconsistent', 'V6'] = 'inconsistent'
    score_bands.plot_confusion_matrix2(dfv, normalization=None, filename='numbers')  # normalization = 'true' or None
    score_bands.plot_confusion_matrix2(dfv, normalization='all', filename='percentage')
    return


def plot_waterfall(df):
    # formatting df with only columns of interest
    df = df[['score_category_detailed', 'recommended_category', 'V5_iso_risk', 'is_consistent']].copy()
    df = df.rename(columns={'score_category_detailed': 'V5', 'recommended_category': 'V6'})
    df['V5'] = df['V5'].replace({'Run-off': 'PricingIneligibility'})
    df['V5_iso_risk'] = df['V5_iso_risk'].replace({'Run-off': 'PricingIneligibility'})
    df['V6'] = df['V6'].replace({'Run-off': 'PricingIneligibility'})
    # list of conditions
    prod_condition = ['A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'uwr_rejection']
    nb_of_uwrs = df[lambda x: x.V5_iso_risk.isin(['uwr_rejection'])].V5_iso_risk.shape[0]
    v5_v5_iso_risk_prod_difference = (df[lambda x: x.V5.isin(prod_condition)].V5.shape[0] -
                                      df[lambda x: x.V5_iso_risk.isin(prod_condition)].V5_iso_risk.shape[0])
    nb_of_inconsistent = df[lambda x: (x.is_consistent == 'inconsistent') & (x.V5.isin(prod_condition))].shape[0]
    ################
    # df_waterfall #
    ################
    df_waterfall = DataFrame(columns=['V5_eligible', 'new_uwrs', 'V5_new_score_bands', 'consistent',
                                         'V6_eligible'])
    # all production is everybody but runoffs (including inconsistent)
    df_waterfall.loc[0, 'V5_eligible'] = df[lambda x: x.V5.isin(prod_condition)].V5.shape[0]
    # new UWRs : we substract the population here above the people falling in uwr_rejection
    #df_waterfall.loc[0, 'new_uwrs'] = df_waterfall.loc[0, 'V5_eligible'] - nb_of_uwrs
    df_waterfall.loc[0, 'new_uwrs'] = - nb_of_uwrs
    # V5_new_score_bands: we substract the population here above the difference between V5-prod and V5_is_risk-prod
    #df_waterfall.loc[0, 'V5_new_score_bands'] = df_waterfall.loc[0, 'new_uwrs'] - v5_v5_iso_risk_prod_difference
    df_waterfall.loc[0, 'V5_new_score_bands'] = - v5_v5_iso_risk_prod_difference
    #df_waterfall.loc[0, 'consistent'] = df_waterfall.loc[0, 'V5_new_score_bands'] - nb_of_inconsistent
    df_waterfall.loc[0, 'consistent'] = - nb_of_inconsistent
    # df_waterfall.loc[0, 'New_prod_benchmark'] = df_waterfall.loc[0, 'consistent']
    # plot
    # column_names = list(df_waterfall.columns)
    # values = df_waterfall.values.flatten()
    # sns.set_style("whitegrid")
    # plt.figure(figsize=(10, 6))
    # plt.xticks(rotation=45)
    # sns.barplot(x=column_names, y=values)
    plt.figure(figsize=(8, 6))
    # Add value label on top of the 'net' column
    net_value = df_waterfall.sum(axis=1)
    waterfall(df_waterfall.columns, df_waterfall.values.tolist()[0], formatting='{:,.0f}')
    # plt.text(len(df_waterfall) - 4, net_value + 100, f'{net_value}', ha='center')
    # Add the additional bar on the right-hand side
    # plt.twinx()
    df_waterfall.loc[0, 'V6_eligible'] = df[lambda x: x.V6.isin(['A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7'])].V6.shape[0]
    plt.bar(len(df_waterfall)+3, df_waterfall['V6_eligible'],
            color='orange')
    # Add value labels on top of the bars
    for i, value in enumerate(df_waterfall.values.tolist()[0]+[int(net_value[0])]):
        if (i == 4) or (i == 5):
            plt.text(i, value + 100, f'{value}', ha='center')
    plt.title('Waterfall Plot')
    plt.xlabel('Category')
    plt.ylabel('Value')
    plt.savefig(os.path.join("figures", "impact_analysis", 'waterfall.png'), dpi=600,
                bbox_inches='tight')
    plt.show()

def assign_risk_category(score_value):
    if isna(score_value):
        return 'Run-off'  # Handle None values: I checked and they all correspond to
                          # preapproval_reason==PricingIneligibility
    elif 8876 < score_value < 9248:
        return 'A3'
    elif 8516 < score_value < 8875:
        return 'A6'
    elif 8169 < score_value < 8515:
        return 'A7'
    elif score_value <= 8169:
        return 'Run-off'
    else:
        return 'problem'


# recent demands with appropriate filters
df_preprocessed = data_loader_recent_demands()
# score bands calculation on recent demands - addition of recommended_category and score_v6_computed columns
df_preprocessed = computation_score_bands_applied_on_recent_demands(df_full, df_preprocessed)
###########################
# migration matrix plot ###
###########################
score_bands.plot_confusion_matrix(df_preprocessed)
# dataframe with volumes
# creating V5_iso_risk column which corresponds to V5 score bands with Q1 2023 score bands cutoffs
df_preprocessed['V5_iso_risk'] = df_preprocessed['preapproval_score_value'].apply(assign_risk_category)
# adding the uwr_rejection to V5_iso_risk, which are the same as for V6, i.e. recommended_category
df_preprocessed.loc[lambda x: x['recommended_category'] == 'uwr_rejection', 'V5_iso_risk'] = 'uwr_rejection'
dfv = melt(df_preprocessed[['score_category_declared', 'recommended_category', 'V5_iso_risk']])
dfv['variable'] = dfv['variable'].replace({'score_category_declared': 'V5', 'recommended_category': 'V6'})
####################################
# count_plot_migration_comparison ##
####################################
count_plot_migration_comparison(dfv)
print(dfv[lambda x: x.variable == 'V6'][lambda x: x.value != 'Run-off'].shape[0]-
      dfv[lambda x: x.variable == 'V5_iso_risk'][lambda x: x.value != 'Run-off'].shape[0])
#################################
# volumes_score_bands_grouped ###
#################################
table_volume_score_bands_groupes(dfv)
########################################################
# migration matrices plot with inconsistent category ###
########################################################
pre_acc_vs_inconsistent_migration_matrices(df_preprocessed)
####################
# Waterfall plot ###
####################
plot_waterfall(df_preprocessed)

print('max V5', df_preprocessed.groupby('score_category_declared')['preapproval_score_value'].max())
print('min V5', df_preprocessed.groupby('score_category_declared')['preapproval_score_value'].min())
print('mean V6', df_preprocessed.groupby('score_category_declared')['score_v6_computed'].mean())
df_preprocessed.to_csv('df_for_timothee.csv')
