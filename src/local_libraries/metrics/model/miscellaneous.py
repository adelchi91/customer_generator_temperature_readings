# from io import BytesIO
# from typing import TYPE_CHECKING
#
# import matplotlib.pyplot as plt
# import seaborn as sns
# from catboost import CatBoostClassifier, Pool
# from numpy import array, where
# from pandas import DataFrame, concat, get_dummies, melt
# from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
#
# from yc_younipy.super_scenario.utils.dataframe_preprocessing import (
#     categorical_features_indices,
# )
#
# if TYPE_CHECKING:
#     from typing import Dict
#
# target = "XXX"
# transaction_amount = "XXX"
# prediction_column = "XXX"
# split_col = "XXX"
# variables = "XXX"
# group_shuffle_split_id = "XXX"
#
#
# def compute_accuracy_by_amount(df: DataFrame) -> Dict:
#     """
#     Computes the accuracy by amount
#     :param df: input dataframe
#     :return: dictionary containing calculated metrics
#     """
#     df["correct_amount"] = where(
#         df[target] == df[prediction_column],
#         df["transaction_amount"],
#         0,
#     )
#     df = df.groupby(target, as_index=False).agg(
#         support=(target, "count"),
#         correct_amounts=("correct_amount", lambda x: sum(abs(x))),
#         total_amounts=("transaction_amount", lambda x: sum(abs(x))),
#     )
#     df["accuracy_by_amount"] = df["correct_amounts"] / df["total_amounts"]
#
#     res = dict(zip(df[target], df[["accuracy_by_amount"]].to_dict(orient="records")))
#
#     res.update(
#         {
#             "macro avg": {"accuracy_by_amount": df["accuracy_by_amount"].mean()},
#             "weighted avg": {
#                 "accuracy_by_amount": (df["accuracy_by_amount"] * df["support"]).sum() / df["support"].sum()
#             },
#         }
#     )
#
#     return res
#
#
# def compute_performances_over_population_size(df: DataFrame, model: CatBoostClassifier) -> bytes:
#     """
#
#     :param df: input dataframe
#     :param model: models to get params from
#     :return: info about the graph
#     """
#     x_validation = df[df[split_col] == "validation"][variables]
#     y_validation = df[df[split_col] == "validation"][target]
#     categorical_col_indices = categorical_features_indices(df[variables])
#
#     labels = sorted(set(df[target]))
#
#     res = []
#     for pop_pct in [x / 10 for x in range(1, 10)]:
#
#         if bool(group_shuffle_split_id):
#             ids_to_sample = df[df[split_col] == "train_test"][group_shuffle_split_id].drop_duplicates()
#             sampled_ids = ids_to_sample.sample(int(len(ids_to_sample) * pop_pct))
#             df_tmp = df.loc[df[group_shuffle_split_id].isin(sampled_ids)]
#         else:
#             df_tmp = df.sample(int(len(df) * pop_pct))
#
#         x_train_test = df_tmp[variables]
#         y_train_test = df_tmp[target]
#
#         missing_targets = list(set(y_validation).difference(y_train_test))
#
#         train_dataset = Pool(
#             data=x_train_test,
#             label=y_train_test,
#             cat_features=categorical_col_indices,
#         )
#         eval_dataset = Pool(
#             data=df[(df[split_col] == "validation") & (~df[target].isin(missing_targets))][variables],
#             label=y_validation[~y_validation.isin(missing_targets)],
#             cat_features=categorical_col_indices,
#         )
#
#         tmp_model = CatBoostClassifier(**model.get_params())
#         tmp_model.fit(train_dataset, eval_set=eval_dataset)
#
#         macro_avg = classification_report(
#             y_validation,
#             tmp_model.predict(x_validation),
#             labels=labels,
#             target_names=labels,
#             output_dict=True,
#         )["weighted avg"]
#
#         accuracy_by_amount = compute_accuracy_by_amount(
#             df=concat(
#                 [
#                     x_validation.reset_index(drop=True),
#                     DataFrame(y_validation).rename({"0": target}, axis=1).reset_index(drop=True),
#                     DataFrame(tmp_model.predict(x_validation)).rename({0: prediction_column}, axis=1),
#                 ],
#                 axis=1,
#             )
#         )["weighted avg"]
#         macro_avg.update(accuracy_by_amount)
#
#         tgt_dummies = get_dummies(y_validation)[list(tmp_model.classes_) + missing_targets]
#         preds = DataFrame(tmp_model.predict_proba(x_validation), columns=list(tmp_model.classes_))
#
#         for col in missing_targets:
#             preds[col] = 0
#
#         macro_avg.update(
#             {
#                 "population percentage": f"{pop_pct:.0%}",
#                 "AUC_macro": roc_auc_score(tgt_dummies, preds, average="macro"),
#                 "AUC_weighted": roc_auc_score(tgt_dummies, preds, average="weighted"),
#             }
#         )
#         res.append(macro_avg)
#
#     full_pop = classification_report(
#         y_validation,
#         df[df[split_col] == "validation"][prediction_column],
#         labels=labels,
#         target_names=labels,
#         output_dict=True,
#     )["weighted avg"]
#
#     accuracy_by_amount = compute_accuracy_by_amount(df=df)["weighted avg"]
#     full_pop.update(accuracy_by_amount)
#
#     tgt_dummies = get_dummies(y_validation)[list(model.classes_)]
#     preds = DataFrame(model.predict_proba(x_validation), columns=list(model.classes_))
#
#     full_pop.update(
#         {
#             "population percentage": "100%",
#             "AUC_macro": roc_auc_score(tgt_dummies, preds, average="macro"),
#             "AUC_weighted": roc_auc_score(tgt_dummies, preds, average="weighted"),
#         }
#     )
#     res.append(full_pop)
#
#     res = melt(
#         DataFrame(res),
#         id_vars="population percentage",
#         value_vars=["precision", "recall", "f1-score", "accuracy_by_amount"],
#         var_name="metric",
#     )
#
#     fig_file = BytesIO()
#
#     sns.lineplot(data=res, x="population percentage", y="value", hue="metric", palette="Set2")
#     plt.title(f"100% = {len(df[df[split_col] != 'validation'])} transactions")
#
#     plt.savefig(fig_file, format="png")
#     plt.close("all")
#     return fig_file.getvalue()
#
#
# def plot_confusion_matrix(cf_matrix, name=None, pct=False) -> bytes:
#     """
#     plot confusion matrix
#     :param cf_matrix:
#     :param name: graph name
#     :param pct: pct for the plot
#     :return: info about the graph
#     """
#     fig_file = BytesIO()
#
#     plt.figure(figsize=(14, 8))
#     if pct:
#         sns.heatmap(cf_matrix, annot=True, fmt=".1%", cmap="Blues")
#     else:
#         sns.heatmap(cf_matrix, annot=True, cmap="Blues")
#
#     plt.xlabel("y pred")
#     plt.ylabel("y target")
#     if name:
#         plt.title(name)
#     plt.tight_layout()
#
#     plt.savefig(fig_file, format="png")
#     plt.close("all")
#     return fig_file.getvalue()
#
#
# def plot_classification_report(classification_report, name=None, support_normalization=None) -> bytes:
#     """
#     Plot cassification report
#     :param classification_report: Dict of the report.
#     :param name: graph name
#     :return: info about the graph
#     """
#     fig_file = BytesIO()
#
#     plt.figure(figsize=(14, 8))
#     if bool(support_normalization):
#         sns.heatmap(
#             DataFrame(
#                 {
#                     key: {x: y / support_normalization if x == "support" else y for x, y in val.items()}
#                     if key != "accuracy"
#                     else val
#                     for key, val in classification_report.items()
#                 }
#             ).T,
#             annot=True,
#             cmap="Blues",
#         )
#     else:
#         sns.heatmap(
#             DataFrame(classification_report).iloc[:-1, :].T,
#             annot=True,
#             cmap="Blues",
#         )
#
#     if name:
#         plt.title(name.upper())
#     plt.tight_layout()
#
#     plt.savefig(fig_file, format="png")
#     plt.close("all")
#     return fig_file.getvalue()
#
#
# def plot_train_test_classification_report(df: DataFrame, prediction_column: str) -> Dict:
#     """
#     plot train test classification report
#     :param df: input dataframe
#     :return: info about the graph
#     """
#     res = {}
#     labels = sorted(set(df[target]), key=str.lower)
#
#     for split in set(df[split_col]):
#         res[split] = {}
#         df_tmp = df.loc[lambda x: x[split_col] == split]
#
#         classif_report = classification_report(
#             df_tmp[target],
#             df_tmp[prediction_column],
#             labels=labels,
#             target_names=labels,
#             output_dict=True,
#         )
#
#         accuracy_by_amount = compute_accuracy_by_amount(df=df_tmp)
#
#         [classif_report[key].update(accuracy_by_amount[key]) for key in accuracy_by_amount.keys()]
#
#         res[split]["dict_object"] = classif_report
#         res[split]["plot"] = plot_classification_report(
#             classification_report=classif_report,
#             name=f"{split.upper()}: Classification report",
#             support_normalization=len(df_tmp),
#         )
#
#     return res
#
#
# def plot_train_test_confusion_matrix(df: DataFrame, prediction_column: str) -> Dict:
#     """
#     plot train test confusion matrix
#     :param df: input dataframe
#     :return: info about the graph
#     """
#     res = {}
#     for split in set(df[split_col]):
#         res[split] = {}
#         df_tmp = df.loc[lambda x: x[split_col] == split]
#         labels = sorted(set(df_tmp[target]), key=str.lower)
#         cf_matrix = confusion_matrix(df_tmp[target], df_tmp[prediction_column])
#
#         res[split]["confusion_matrix"] = DataFrame(cf_matrix, index=labels, columns=labels)
#         res[split]["confusion_matrix_plot_volume"] = plot_confusion_matrix(
#             DataFrame(cf_matrix, index=labels, columns=labels),
#             f"{split.upper()}: Confusions matrix (Volume)",
#         )
#         res[split]["confusion_matrix_plot_pct"] = plot_confusion_matrix(
#             DataFrame(
#                 (cf_matrix.T / cf_matrix.sum(axis=1)).T,
#                 index=labels,
#                 columns=labels,
#             ),
#             f"{split.upper()}: Confusions matrix (%)",
#         )
#
#     return res
#
#
# def plot_feature_importance(importance, names, nb_features_to_plot=20) -> bytes:
#     """
#     plot feature importance
#     :param importance:
#     :param names: list of columns name
#     :param nb_features_to_plot:
#     :return: info about the graph
#     """
#     # Create arrays from feature importance and feature names
#     feature_importance = array(importance)
#     feature_names = array(names)
#     # Create a DataFrame using a Dictionary
#     data = {
#         "feature_names": feature_names,
#         "feature_importance": feature_importance,
#     }
#     fi_df = DataFrame(data)
#     # Sort the DataFrame in order decreasing feature importance
#     fi_df.sort_values(by=["feature_importance"], ascending=False, inplace=True)
#
#     # Define size of bar plot
#     fig_file = BytesIO()
#     plt.figure(figsize=(10, 8))
#     # Plot Searborn bar chart
#     sns.barplot(
#         x=fi_df.iloc[:nb_features_to_plot]["feature_importance"],
#         y=fi_df.iloc[:nb_features_to_plot]["feature_names"],
#     )
#     # Add chart labels
#     plt.title("FEATURE IMPORTANCE")
#     plt.xlabel("FEATURE IMPORTANCE")
#     plt.ylabel("FEATURE NAMES")
#
#     plt.savefig(fig_file, format="png")
#     plt.close("all")
#     return fig_file.getvalue()
