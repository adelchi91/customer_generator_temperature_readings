import sys
import os
import pickle
import cloudpickle
import yaml
from numpy import round
from typing import TYPE_CHECKING
from pandas import options, concat
from local_libraries import buckets_creation, pop_over_time

if TYPE_CHECKING:
    from typing import List, Optional, Tuple

    from pandas.core.indexes.base import Index

options.display.max_columns = 40
options.display.width = 1000

if len(sys.argv) != 4:
    sys.stderr.write("Arguments error. Usage:\n")
    sys.stderr.write(
        "\tpython stats_model.py training_dataset_path validation_dataset_path pipeline_path variables_file")
    sys.exit(1)

# Load files
training_dataset_path = sys.argv[1]
validation_dataset_path = sys.argv[2]
variables_file = sys.argv[3]

# Load files
df_train = pickle.load(open(training_dataset_path, "rb"))
df_val = pickle.load(open(validation_dataset_path, "rb"))

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

variables_bf_pp = []
for v_set in variables_set_bf_pp:
    variables_bf_pp += all_variables[v_set]


def convert_dataframe(alt_w_pred):
    if 'business_provider_code' in alt_w_pred.columns:
        alt_w_pred_converted = alt_w_pred.assign(verified_is_homeowner=lambda x: x.verified_is_homeowner.astype('string'),
                                                 monthly_nb_transactions_transport_other=lambda
                                                     x: x.monthly_nb_transactions_transport_other.astype('float64'),
                                                 business_provider_code=lambda x: x.business_provider_code.astype('string')
                                                 )
    else:
        alt_w_pred_converted = alt_w_pred.assign(verified_is_homeowner=lambda x: x.verified_is_homeowner.astype('string'),
                                                 monthly_nb_transactions_transport_other=lambda
                                                     x: x.monthly_nb_transactions_transport_other.astype('float64'),
                                                 )
    return alt_w_pred_converted


df_train['dataframe'] = 'train_test'
df_val['dataframe'] = 'test'
df_full = concat([df_train, df_val], axis=0)

categorical_vars = ['verified_is_homeowner', "business_provider_code"]
numerical_vars = ['equifax_rating', 'prop_time_in_overdraft',
  'mean_monthly_professional_income',
  'mean_monthly_micro_loan_instalment',
  'mean_monthly_cash_withdrawal',
  'monthly_nb_transactions_energy_utility',
  'monthly_nb_transactions_consumer_loan_instalment',
  'monthly_nb_transactions_micro_loan_instalment',
  'monthly_nb_transactions_transport_other',
  'monthly_nb_transactions_bank_expenses',
  'monthly_nb_transactions_housing_expenses',
  'mean_daily_rejection_keyword_micro_credit_amount'
]

df_full_converted = convert_dataframe(df_full)
output = buckets_creation.binning(df_full_converted[form_variables + cat_feature + ['application_date', 'contract_reference']],
                                  numerical_vars, categorical_vars)

# Save outputs
os.makedirs(os.path.join("figures", "description_over_time"), exist_ok=True)
vars_ = output.filter(regex='^num__|^str__', axis=1).columns.to_list()
for var in vars_:
    for use_prop in [True, False]:
        bytes_plot = pop_over_time.plot_pop_over_time(output[var], output['remainder__application_date'], 'M', var,
                                                      use_prop)
        with open(f"./figures/description_over_time/{var}_over_time_prop_{use_prop}.png", "wb") as f:
            f.write(bytes_plot)

# df_performance_summary.to_csv('data/stats_model/performance_summary.csv')
