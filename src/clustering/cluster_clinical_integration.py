from lifelines import CoxPHFitter
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import FunctionTransformer, OrdinalEncoder
from sklearn.model_selection import RepeatedStratifiedKFold
from sksurv.linear_model import CoxPHSurvivalAnalysis
import os
from sksurv.util import Surv
from sksurv.metrics import concordance_index_ipcw, cumulative_dynamic_auc
from itertools import combinations
import sklearn
from statsmodels.stats.multitest import multipletests
from tqdm.auto import tqdm

from src.clustering.settings import clinical_endpoints, final_cluster_name, RANDOM_STATE, best_pipeline
sklearn.set_config(transform_output="pandas")

clinical_data = pd.read_csv(os.path.join("outcomes", "all_fields_processed.csv"))
clusters = os.path.join("results_v2_permodality", "clustering", "embedding_clusters.csv")
clusters = pd.read_csv(clusters, index_col=0)[best_pipeline["name"]]
clinical_data = clinical_data.merge(clusters, left_on="patient_id", right_index=True)

results = []
for endpoint, y_cols in clinical_endpoints.items():
    t = y_cols["t"]
    e = y_cols["e"]


    clinical_data = clinical_data.dropna(subset=[t, e])
    y = Surv.from_dataframe(time=t, event=e, data=clinical_data)
    X = clinical_data.loc[clinical_data.index, ["HPV_Status", "Age", "Sex", "Clinical_T_Stage",
                                                "Clinical_N_Stage", best_pipeline["name"]]]
    X["Clinical_T_Stage"] = X["Clinical_T_Stage"].str[1].astype(float)
    X["Clinical_N_Stage"] = X["Clinical_N_Stage"].str[1].astype(float)

    transformer = ColumnTransformer(
        transformers=[
            ('cat', make_pipeline(
                SimpleImputer(strategy="most_frequent", add_indicator=True),
                OrdinalEncoder().set_output(transform="pandas"),
                ),
                ["HPV_Status", "Sex"]
                ),
            ('num', make_pipeline(
                SimpleImputer(strategy="mean", add_indicator=True),
                ), ["Clinical_T_Stage", "Clinical_N_Stage", "Age"]),
            ],
        remainder='passthrough'
    )


    rskf = RepeatedStratifiedKFold(n_splits=10, n_repeats=100, random_state=RANDOM_STATE)
    x_cols = []
    for r in range(1, len(X.columns) + 1):
        x_cols.extend(combinations(X.columns, r))
    x_cols = [list(cols) for cols in x_cols]
    for i_col, cols in enumerate(tqdm(x_cols)):
        pattern = '|'.join(cols)
        pipeline = make_pipeline(
            transformer,
            FunctionTransformer(lambda x: x[x.columns[x.columns.str.contains(pattern)]]),
            CoxPHSurvivalAnalysis()
        )

        for i, (train_index, test_index) in enumerate(rskf.split(X, clinical_data[e])):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y[train_index], y[test_index]
            pipeline.fit(X_train, y_train)
            pred = pipeline.predict(X_test)
            c_uno = concordance_index_ipcw(y_train, y_test, pred)[0]

            cox_df = pipeline[:-1].transform(X_train)
            cox_df = pd.DataFrame(cox_df, index=X_train.index)
            cox_df = pd.concat([cox_df, pd.DataFrame(y_train, index=X_train.index)], axis=1)
            cox_model = CoxPHFitter().fit(cox_df, duration_col=t, event_col=e)
            pval = cox_model.log_likelihood_ratio_test().p_value
            log_likelihood = cox_model.log_likelihood_

            train_df = pd.DataFrame(y_train)
            train_events = train_df[train_df.iloc[:, 0]]
            times = np.percentile(train_events.iloc[:, 1], np.linspace(5, 91, 15))
            test_time = pd.DataFrame(y_test)
            max_test_time = test_time[test_time.iloc[:, 0]].iloc[:, 1].max()
            min_test_time = test_time[test_time.iloc[:, 0]].iloc[:, 1].min()
            times = times[(times > min_test_time) & (times < max_test_time)]
            aucs, mean_auc = cumulative_dynamic_auc(y_train, y_test, pred, times)

            results_dict = {
                "endpoint": endpoint,
                "i_col": i_col,
                "cols": cols,
                "ncols": len(cols),
                "fold": i,
                "c_uno": c_uno,
                "aucs": aucs.tolist(),
                "mean_auc": mean_auc,
                "times": times.tolist(),
                "pval": pval,
                "log_likelihood": log_likelihood,
                "nparams": len(cox_model.params_)
            }
            results.append(results_dict)
results = pd.DataFrame(results)
results["cols"] = results["cols"].apply(lambda x: [col if col != best_pipeline["name"] else final_cluster_name for col in x ])
results["cols"] = results["cols"].apply(lambda x: [col if col != "Clinical_N_Stage" else "N-stage" for col in x ])
results["cols"] = results["cols"].apply(lambda x: [col if col != "Clinical_T_Stage" else "T-stage" for col in x ])
results["adjp"] = multipletests(results["pval"], method='fdr_bh')[1]
results.to_csv(os.path.join("results_v2_permodality", "clustering", f"cox_cluster_clinical.csv"), index=False)


