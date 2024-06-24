import sys
import os
import pickle
import cloudpickle
import yaml
from yc_younipy.metrics.univariate.univariate_analysis import (
    plot_target_rate_over_time,
    plot_univariate_analysis,
)
from yc_younipy.metrics.population import pop_over_time
import PIL.Image as Image
from io import BytesIO

from yc_younipy.plot import plot_bytes_image
from yc_younipy.metrics.model.calibration import plot_calibration_curve

from typing import TYPE_CHECKING
from pandas import options, DataFrame, concat

if TYPE_CHECKING:
    from typing import List, Optional, Tuple

    from pandas.core.indexes.base import Index

options.display.max_columns = 40
options.display.width = 1000

if len(sys.argv) != 5:
    sys.stderr.write("Arguments error. Usage:\n")
    sys.stderr.write(
        "\tpython univariate.py training_dataset_path validation_dataset_path pipeline_path variables_file")
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
preprocess_variables = all_variables['preprocessing_use_only']

variables_bf_pp = []
for v_set in variables_set_bf_pp:
    variables_bf_pp += all_variables[v_set]

# df_preprocessed
df_train_test_val = concat([df_train, df_val], axis=0)
df_preprocessed = pipeline['preprocessing'].transform(df_train_test_val[preprocess_variables + variables_bf_pp])

# testing plot_univariate_analysis ------------------------------------------------------------------------------------
tmp0 = plot_univariate_analysis(df_preprocessed["verified_is_homeowner"], df_train_test_val[target], missing_value="Missing")
# img0 = plot_bytes_image(tmp0)
img0 = Image.open(BytesIO(tmp0))
# testing plot_target_rate_over_time ----------------------------------------------------------------------------------

tmp1 = plot_target_rate_over_time(df_train_test_val["application_date"], df_train_test_val[target], date_cohort_type="W")
img1 = Image.open(BytesIO(tmp1))
# plot_bytes_image(tmp1)


tmp2 = pop_over_time.plot_pop_over_time(x=df_train_test_val['mean_monthly_professional_income'], x_date = df_train_test_val["application_date"],
                                        date_cohort_type="application_date", variable_name="mean_monthly_professional_income")
img2 = Image.open(BytesIO(tmp2))

# Save outputs
os.makedirs(os.path.join("figures", "univariate"), exist_ok=True)
img0.save(os.path.join("figures", "univariate", f"verified_is_homeowner.png"))
img1.save(os.path.join("figures", "univariate", f"target_temporal.png"))

# with open(os.path.join("figures", "univariate", f"verified_is_homeowner.png"), 'wb') as f:
#     f.write(img0)