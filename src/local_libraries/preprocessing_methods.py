import pandas as pd
from datetime import timedelta
import numpy as np
from sklearn.model_selection import train_test_split
import sys

from sklearn.model_selection import TimeSeriesSplit

# Function returning output feature names when passed input feature names
def get_features_out(ft, input_features):
    return list(input_features)  # + ['feature1', 'feature2', 'feature3']


def compute_age_wrapper(date_from, date_to):
    try:
        dates = [pd.to_datetime(date, utc=True, errors='coerce')
                 if not isinstance(date, pd.Timestamp)
                 else date.dt.tz_localize('utc')
                 for date in [date_from, date_to]]
        return (dates[1] - dates[0]) / timedelta(days=365)
    except ValueError:
        res = np.nan
    return res


# Preprocessing function
def preprocess_form(df: pd.DataFrame) -> pd.DataFrame:
    df_form_preprocess = df.copy()
    df_form_preprocess["declared_applicant_age"] = compute_age_wrapper(df_form_preprocess["declared_date_of_birth"],
                                                                       df_form_preprocess["application_date"]).astype(
        np.float32)
    df_form_preprocess["verified_applicant_age"] = compute_age_wrapper(df_form_preprocess["verified_date_of_birth"],
                                                                       df_form_preprocess["application_date"]).astype(
        np.float32)

    # df_form_preprocess["declared_is_homeowner"] = df_form_preprocess["declared_housing_code"].str.startswith(
    #     "HOME_OWNERSHIP_"
    # )
    # df_form_preprocess["verified_is_homeowner"] = df_form_preprocess["verified_housing_code"].str.startswith(
    #     "HOME_OWNERSHIP_"
    # )
    # df_form_preprocess["censored_loan_age"] = (
    #     df_form_preprocess["loan_age"].clip(upper=df_form_preprocess["dn3_month"]).clip(
    #         upper=df_form_preprocess["maturity_in_months"])
    #)
    # df_form_preprocess["dn3_observed"] = df_form_preprocess["dn3_month"] <= df_form_preprocess["maturity_in_months"]
    # df_form_preprocess["signed_loan_age"] = df_form_preprocess["censored_loan_age"].where(
    #     df_form_preprocess["dn3_observed"], -df_form_preprocess["censored_loan_age"]
    # )
    # df_form_preprocess["dn_12"] = df_form_preprocess[["dn1_12", "dn2_12", "dn3_12"]].sum(axis=1)
    df_form_preprocess.sort_values("application_date", inplace=True)
    print('Features created by preprocessing:')
    print([col for col in df_form_preprocess.columns if col not in df.columns])

    output_columns = get_features_out(None, df.columns)

    return df_form_preprocess[output_columns+['verified_is_homeowner']]


# Ensembles creation
def create_ensembles(df, split_type, validation_date, validation_size):
    if split_type == 'out_of_time':
        # time_threshold = df['application_date'].quantile(1 - validation_size)
        # print('Time threshold chosen : ', time_threshold)
        # tss = TimeSeriesSplit(n_splits=2, test_size=int(len(df) * 0.3))
        # for train_index, test_index in tss.split(df):
        #     df_train, df_val = df.iloc[train_index], df.iloc[test_index]
        print('Time threshold chosen : ', validation_date)
        df = df.assign(split=lambda x: np.where(x['application_date'] < validation_date,
                                      "train_test", "validation"))
        df_train = df.loc[lambda x: x['split'] == "train_test"].sort_values("application_date")
        df_val = df.loc[lambda x: x['split'] == "validation"].sort_values("application_date")

    elif split_type == 'train_test':
        df_train, df_val, _, _ = train_test_split(df, [0] * df.shape[0], test_size=validation_size, random_state=42)

    elif split_type == 'no_split':
        df_train = df
        df_val = None

    else:
        print('ERROR: this split method isn\'t implemented yet !')
        sys.exit(1)

    return df_train, df_val


# Create new columns following the pattern <feature>_IS_NULL with binary values
# That way, we can replace the missing values within numerical features without losing this information
def handle_numerical_missing(df):
    new_df = df.copy()
    for num_col in [col for col in df.columns]:
        new_df[f'{num_col}_IS_NULL'] = pd.isnull(df[num_col]).astype(int)

    output_features = df.columns.tolist() + [col + '_IS_NULL' for col in df.columns.values]
    return new_df[output_features]


# Get the feature names after handle_numerical_missing was launched
def get_numerical_features_after_missing_treatment(tf, input_features):
    return list(input_features) + [feature + '_IS_NULL' for feature in input_features]
