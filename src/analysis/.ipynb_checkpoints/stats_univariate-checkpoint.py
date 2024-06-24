import sys
import os
import pickle
import cloudpickle
import yaml
from numpy import round
from typing import TYPE_CHECKING
from pandas import options, concat
from local_libraries import buckets_creation, pop_over_time
from yc_younipy.metrics.univariate.univariate_analysis import (
    plot_target_rate_over_time,
    plot_univariate_analysis,
)
from yc_younipy.metrics.population import pop

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

# Load variables set
all_variables = yaml.safe_load(open(variables_file))
form_variables = all_variables['form']
cat_feature = all_variables['cat_feature']
preprocess_variables = all_variables['preprocessing_use_only']


def convert_dataframe(alt_w_pred):
    if 'business_provider_code' in alt_w_pred.columns:
        alt_w_pred_converted = alt_w_pred.assign(verified_is_homeowner=lambda x: x.verified_is_homeowner.astype('string'),
                                                 business_provider_code=lambda x: x.business_provider_code.astype('string'))
    else:
        alt_w_pred_converted = alt_w_pred.assign(verified_is_homeowner=lambda x: x.verified_is_homeowner.astype('string'))
    return alt_w_pred_converted


df_train['dataframe'] = 'train_test'
df_val['dataframe'] = 'test'
df_full = concat([df_train, df_val], axis=0)

categorical_vars = cat_feature+['verified_is_homeowner']
numerical_vars =   ['equifax_rating', 'prop_time_in_overdraft',
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
output = buckets_creation.binning(df_full_converted[numerical_vars + categorical_vars + ['dn3_12', 'contract_reference', 'dataframe']],
                                  numerical_vars, categorical_vars)

# Save outputs
os.makedirs(os.path.join("figures", "stats_univariate_analysis"), exist_ok=True)
vars_ = output.filter(regex='^num__|^str__', axis=1).columns.to_list()
# for var in vars_:
#     for use_prop in [True, False]:
#         bytes_plot = pop.plot_pop(output[var], output['remainder__dn3_12'], var, 'dataframe')
#         with open(f"./figures/univariate/{var}_over_time_prop_{use_prop}.png", "wb") as f:
#             f.write(bytes_plot)
output['remainder__' + target] = output['remainder__' + target].astype('float64')
for var in vars_:
    bytes_plot = plot_univariate_analysis(output[var], output['remainder__dn3_12'], missing_value="Missing", x_group=output['remainder__dataframe'])
    with open(f"./figures/stats_univariate_analysis/{var}_univariate.png", "wb") as f:
        f.write(bytes_plot)




