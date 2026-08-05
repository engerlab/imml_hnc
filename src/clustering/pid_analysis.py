from itertools import product
from joblib import Parallel, delayed
from joblib import Parallel
import numpy as np
import pandas as pd
import os
from imml.model_selection import MMSplitter
from imml.preprocessing import select_complete_samples
from imml.statistics import pid
from sklearn.model_selection import StratifiedShuffleSplit
from tqdm.auto import tqdm

from src.clustering.utils import get_Xs_ids, prepare_Xs
from src.clustering.settings import RANDOM_STATE, best_pipeline, clinical_endpoints, timepoints


def estimate_pid(cl, endpoint, y_cols, timepoint):
    t = y_cols["t"]
    e = y_cols["e"]
    clinical_data_endpoint = clinical_data.dropna(subset=[t, e])
    clinical_data_endpoint = clinical_data_endpoint.sort_values(by=t, ascending=False)
    clinical_data_endpoint = clinical_data_endpoint[~clinical_data_endpoint.index.duplicated(keep='first')]
    mask = clinical_data_endpoint.index.intersection(Xs[0].index)
    clinical_data_endpoint = clinical_data_endpoint.loc[mask]
    Xs_e = [X.loc[mask] for X in Xs]

    rskf = StratifiedShuffleSplit(n_splits=100, test_size=.2, random_state=RANDOM_STATE)
    mm_splitter = MMSplitter(rskf)
    result_list = []
    for i, (Xs_train, _, _, _) in enumerate(mm_splitter.split(Xs_e, clinical_data_endpoint[e])):
        clinical_data_endpoint_fold = clinical_data_endpoint.loc[Xs_train[0].index]
        conditions = [
            (clinical_data_endpoint_fold[t] <= timepoint) & (clinical_data_endpoint_fold[e] == 1),
            (clinical_data_endpoint_fold[t] > timepoint),
            (clinical_data_endpoint_fold[t] <= timepoint) & (clinical_data_endpoint_fold[e] == 0),
        ]
        choices = [1, 0, np.nan]

        binary_outcome = np.select(conditions, choices)
        binary_outcome = pd.Series(binary_outcome, index=clinical_data_endpoint_fold.index)
        mask = binary_outcome.isna()
        binary_outcome = binary_outcome.loc[~mask]
        Xs_train_pid = [X.loc[~mask] for X in Xs_train]
        try:
               pid_res = pid(Xs=Xs_train_pid, y=binary_outcome, n_clusters=cl, normalize=True, random_state=RANDOM_STATE)
        except:
             continue
        results_dict = {
            "endpoint": endpoint,
            "fold": i,
            "timepoint": timepoint,
            "n_clusters": cl,
        }
        results_dict = {**results_dict, **pid_res}
        result_list.append(results_dict)
    return result_list


clinical_data = pd.read_csv(os.path.join("outcomes", "all_fields_processed.csv"))
clinical_data = clinical_data.set_index("patient_id")
clinical_data.index = clinical_data.index.astype(str).astype(object)

Xs, ids = get_Xs_ids(llm=best_pipeline["llm"])
Xs = prepare_Xs(Xs)
Xs = select_complete_samples(Xs=Xs)

n_jobs = 20
results = Parallel(n_jobs=n_jobs, return_as="generator_unordered")(
    delayed(estimate_pid)(cl=cl, endpoint=endpoint, y_cols=y_cols, timepoint=timepoint)
    for cl,timepoint,(endpoint, y_cols)  in tqdm(product([10, 25, 50], timepoints, clinical_endpoints.items()))
)
results = [results_dict for result_list in results for results_dict in result_list]
results = pd.DataFrame(results)
results.to_csv(os.path.join("results_v2_permodality", "clustering", f"pid_results.csv"), index=False)



