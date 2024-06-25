# Customer Lead Generator

## Code and data
code can be found in `./notebooks/Exploration_new.ipynb`

Produced figures, models and numerical results are in:
* `./notebooks/figures`
* `./notebooks/models`
* `./notebooks/results`
* `./data/`

## Solution explanation
The columns `b_in_kontakt_gewesen`, `b_gekauft_gesamt` help us to identify the
training data and the labels.

We are dealing here with a small dataset classification problem. From a business point of view, we are assessing how
calling a person results in a final purchase of a product we are trying to sell. The descriptive information of the 
variables is very limited  and one can only guess some of the variables from the feature names. 

A small analysis using a groupby on the `b_in_kontakt_gewesen` and `b_gekauft_gesamt` variables highlights how no contact
whatsover with the  clients always results in no purchase at all, wheras a contact yields to some purchases. 

| b_in_kontakt_gewesen | b_gekauft_gesamt | counts |
|----------------------|------------------|--------|
| 0                    | 0                | 3678   |
| 1                    | 0                | 38     |
| 1                    | 1                | 57     |


Overall, we have a dataset `A = df1[lambda x: x["b_in_kontakt_gewesen"]==1]` where the target `b_gekauft_gesamt` 
is defined and a dataset `B = df1[lambda x: x["b_in_kontakt_gewesen"]==0]` where the target is not really defined. 

The sizes of `A` and `B` very much differ: 

`A.shape  = (95, 26)` and `B.shape = (3678, 26)`.

Hence we decide to follow a threefold strategy:

1. Define a baseline model using only dataset `A`
   1. Do a train-test split (test_size=0.2) to have a training `X_train_cv` dataset and a validation `X_val` dataset
   2. Apply a cross-validation using three splits (due to the small number of observations) on `X_train_cv`
   3. Run a gridsearch to test 3 different models and several corresponding hyperparameters  to limit overfitting as much as possible. Models tested are Random Forest, XGBoost and Logistic Regression. 
   4. Choose best model looking at AUC and Brier scores. 
2. Transfer learning 1 from dataset B to dataset `A` and challenge baseline model 
   1. Use the covariate features of dataset `B`, which are more representative of the incoming clients, to correct dataset A covariate features distributions with a Inverse Propensity Weighting method.
   2. Compute vector weights so that, `A = weights * B`
   3. Feed `weights` to the best baseline model as a fitting parameter and see if it improved the AUC/Brier on validation dataset.
3. Transfer learning 2 from dataset B to dataset A and challenge baseline model 
   1. Use partykit library (see https://cran.r-project.org/web/packages/partykit/index.html) to study dataset `A` and predict target.
   2. Identify splits for which the library predicts class 0 or 1 with minimum error 
   3. Labelise by imputing 0 or 1-values in the targets for dataset `B` given the identifed features/splits. 
   4. Retrain the best baseline model by adding these new labelised observations and see if it improved the AUC/Brier on validation dataset.


Further code was written in order to see how the threshold value of the `predict_proba` method can be tuned in order to 
maximise recall or precision, depending on the business need. Moreover, another piece of code was also written in order to study 
the increasing the number of contacts maximises or not the gain in Euros for given condition of contact price and product cost.



The code leverages some specific libraries:
1. `Optimal Binning` in order to preprocess our data in the scikit learn pipeline of baseline models - see https://gnpalencia.org/optbinning/
2. `Cinnamon` for IPW method implementation (reweighting method), which aims at correcting covariate drifts usually, but can be used in order to match specific distributions see:
   1. https://github.com/zelros/cinnamon/tree/master - GitHub 
   2. https://cinnamon.readthedocs.io/en/latest/ - documentation
   3. https://yohannlefaou.github.io/publications/2021-cinnamon/Detect_explain_and_correct_data_drift_in_a_machine_learning_system.pdf - mathematical details 
   4. https://d2l.ai/chapter_linear-classification/environment-and-distribution-shift.html - mathematical details for a Logistic Regression approach (see Section 4.7.3.2)
3. `Partykit` library for p-values based decision trees (scikit learn implementation uses entropy criterion)
   1. see https://cran.r-project.org/web/packages/partykit/index.html for the R-code 
   2. sudo install R
   3. Open R in your terminal and type `install.packages("party")`
   4. A python wrapper was used to run the R-code in the notebook.
4. `Shapley` library to interpret model features importances - see https://shap.readthedocs.io/en/latest/index.html
5. `UMAP` library for data analysis and dimensionality reduction - see https://umap-learn.readthedocs.io/en/latest/clustering.html


The main conclusions are for the baseline model: 

| Model               | AUC mean | AUC Train mean |
|---------------------|------|----------------|
| Random Forest       | 0.47 | 0.91           |
| XGBoost             | 0.45 | 0.91           |
| Logistic Regression | 0.45 | 0.82           |

with RF being the winner. The validation AUC performance is `AUC_val = 0.58`

The Transfer learning using the reweighting approach yields an increase of 1 AUC point on the validation dataset. 

The transfer learning using the partykit approach yields an increase of 2 AUC point on the validation dataset.  

Hence the final best performance is `AUC_val = 0.60`

