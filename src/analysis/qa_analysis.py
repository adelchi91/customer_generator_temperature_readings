import json
import os
import pickle
import sys
import yaml


# elements passed in cmd of dvc.yaml
if len(sys.argv) != 4:
    sys.stderr.write("Arguments error. Usage:\n")
    sys.stderr.write("\tpython qa_analysis.py df_train df_val variables_file\n")
    sys.exit(1)

# Load args
df_train_path = sys.argv[1]
df_val_path = sys.argv[2]
variables_file = sys.argv[3]

# Load files
df_train = pickle.load(open(df_train_path, "rb"))
df_val = pickle.load(open(df_val_path, "rb"))

# Load params
params = yaml.safe_load(open("params.yaml"))
variables_set = params['general']['variables_set']

# Load variables set
variables = []
all_variables = yaml.safe_load(open(variables_file))
form_variables = all_variables['form']
for v_set in variables_set:
    variables += all_variables[v_set]

# Create directory for outputs
os.makedirs(os.path.join("data", "qa_analysis"), exist_ok=True)


# QA Analysis
def generate_qa_analysis(df):
    qa_analysis = {}
    for col in variables:
        qa_analysis[col] = {}
        # Check the type
        qa_analysis[col]['column_type'] = str(df[col].dtype)
        # Count null values
        if str(df[col].dtype) in ['category', 'string', 'boolean']:
            qa_analysis[col]['missing_values_count'] = float(len(df[df[col] == 'MISSING']))
            if (len(df[df[col] == 'MISSING']) / len(df)) > 0.5:
                print(f'WARNING: more than 50% of null values in variable {col}')
        else:
            qa_analysis[col]['null_values_count'] = float(df[col].isna().sum())
            if float(df[col].isna().mean()) > 0.5:
                print(f'WARNING: more than 50% of null values in variable {col}')
        # Check if there are mixed types
        mixed_types = (df[[col]].applymap(type) != df[[col]].iloc[0].apply(type)).any(axis=1)
        if len(df[[col]][mixed_types]) < 0:
            qa_analysis[col]['mixed_types'] = True
            print(f'WARNING: mixed types in variable {col}')
        else:
            qa_analysis[col]['mixed_types'] = False
        if str(df[col].dtype) in ['category', 'string', 'boolean']:
            qa_analysis[col]['nb_categories'] = float(df[col].nunique())
            if float(df[col].nunique()) > 15:
                print(f'WARNING: more than 15 modalities for variable {col}')

    return qa_analysis


qa_analysis_train = generate_qa_analysis(df_train[form_variables])
if len(df_val) > 0:
    qa_analysis_val = generate_qa_analysis(df_val[form_variables])
else:
    qa_analysis_val = {}


# Save outputs
with open(os.path.join("data", "qa_analysis", "qa_analysis_train.json"), 'w') as jsonfile:
    json.dump(qa_analysis_train, jsonfile)

with open(os.path.join("data", "qa_analysis", "qa_analysis_val.json"), 'w') as jsonfile:
    json.dump(qa_analysis_val, jsonfile)