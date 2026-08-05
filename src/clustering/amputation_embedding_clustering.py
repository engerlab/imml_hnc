from itertools import product
import numpy as np
import pandas as pd
import os
from sklearn.experimental import enable_iterative_imputer
from imml.ampute import Amputer
from imml.model_selection import MMSplitter
from imml.preprocessing import UMTransformer, select_complete_samples, SelectCompleteSamples, MMTransformer
from joblib import Parallel, delayed
from sklearn.impute import IterativeImputer, KNNImputer, SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import silhouette_score
from sklearn.model_selection import ShuffleSplit
from sklearn.pipeline import make_pipeline
from tqdm.auto import tqdm
from datetime import datetime

from src.clustering.utils import get_Xs_ids, prepare_Xs
from src.clustering.settings import N_REPETITIONS, RANDOM_STATE, ALGS_DICT, ps


def run_clustering(llm, alg, p, pipeline, n_cluster, strategy):
    if (p == 0) and (strategy != "All"):
        return []
    missing_p = round(float(p), 1)
    Xs = llm_Xs[llm]["Xs"]
    results_list = []
    splitter = ShuffleSplit(n_splits=N_REPETITIONS, test_size=0.2, random_state=RANDOM_STATE)
    splitter = MMSplitter(splitter=splitter)
    print(f"{llm=}, {alg=}, {n_cluster=}, {missing_p=}, {strategy=}, {datetime.now()}")
    for fold, (train_Xs, _) in enumerate(splitter.split(Xs)):
        if strategy == "All":
            model = make_pipeline(
                Amputer(p=p, random_state=RANDOM_STATE + fold),
                pipeline(n_clusters=n_cluster, random_state=RANDOM_STATE + fold)
            )
        elif strategy == "Complete":
            model = make_pipeline(
                Amputer(p=p, random_state=RANDOM_STATE + fold),
                SelectCompleteSamples(),
                pipeline(n_clusters=n_cluster, random_state=RANDOM_STATE + fold)
            )
        elif strategy == "Mean Imputation":
            model = make_pipeline(
                Amputer(p=p, random_state=RANDOM_STATE + fold),
                MMTransformer(SimpleImputer(strategy="mean").set_output(transform="pandas")),
                pipeline(n_clusters=n_cluster, random_state=RANDOM_STATE + fold)
            )
        elif strategy == "Median Imputation":
            model = make_pipeline(
                Amputer(p=p, random_state=RANDOM_STATE + fold),
                MMTransformer(SimpleImputer(strategy="median").set_output(transform="pandas")),
                pipeline(n_clusters=n_cluster, random_state=RANDOM_STATE + fold)
            )
        elif strategy == "KNN Imputation":
            model = make_pipeline(
                Amputer(p=p, random_state=RANDOM_STATE + fold),
                UMTransformer(KNNImputer().set_output(transform="pandas")),
                pipeline(n_clusters=n_cluster, random_state=RANDOM_STATE + fold)
            )
        else:
            raise ValueError(f"Strategy {strategy} not supported")
        labels = model.fit_predict(train_Xs)
        embedding = model[-1].embedding_
        if len(np.unique(labels)) > 1:
            total_sil = silhouette_score(labels=labels, X=embedding, random_state=RANDOM_STATE)
        else:
            total_sil = np.nan
        Xs_reduced = train_Xs
        for step in model.steps[:-1]:
            Xs_reduced = step[1].transform(Xs_reduced)
        ids = Xs_reduced[0].index.to_list()
        Xs_reduced = select_complete_samples(Xs_reduced)
        ids_reduced = Xs_reduced[0].index
        labels_reduced = pd.Series(labels, index=ids).loc[ids_reduced].values
        if len(np.unique(labels_reduced)) > 1:
            embedding = pd.DataFrame(embedding, index=ids).loc[ids_reduced].values
            reduced_sil = silhouette_score(labels=labels_reduced, X=embedding, random_state=RANDOM_STATE)
        else:
            reduced_sil = np.nan
        result_dict = {
            "LLM": llm,
            "Algorithm": alg,
            "Clusters": n_cluster,
            "Strategy": strategy,
            "Incomplete samples (%)": missing_p,
            "Fold": fold,
            "IDs": ids,
            "Labels": labels,
            "Total Silhouette": total_sil,
            "Reduced Silhouette": reduced_sil,
            "N": len(ids),
            "M": train_Xs[0].shape[1],
        }
        results_list.append(result_dict)
    return results_list

llms = ["smb8b_qwen3_mo"]
llm_Xs = {}
for llm in llms:
    Xs = get_Xs_ids(llm=llm)[0]
    Xs = prepare_Xs(Xs)
    Xs = select_complete_samples(Xs=Xs)
    llm_Xs[llm] = {
        "Xs": Xs,
        "ids": Xs[0].index.to_list(),
    }
n_clusters_list = [2]
ALGS_DICT = {"NEMO": ALGS_DICT["NEMO"]}
strategies = [
    "All",
    "Complete",
    "Mean Imputation",
    "Median Imputation",
    "KNN Imputation",
]
n_jobs = 20
embedding_clusters_permutation = Parallel(n_jobs=n_jobs, return_as="generator_unordered")(
    delayed(run_clustering)(llm=llm, n_cluster=n_cluster, alg=alg, pipeline=pipeline, p=p, strategy=strategy)
    for llm,n_cluster,(alg, pipeline),p,strategy  in tqdm(product(llms, n_clusters_list, ALGS_DICT.items(), ps,
                                                                  strategies))
)
embedding_clusters_permutation = [clusters_dict for clusters_l in embedding_clusters_permutation for clusters_dict in clusters_l]
embedding_clusters_permutation = pd.DataFrame(embedding_clusters_permutation)
embedding_clusters_permutation.to_csv(os.path.join("results_v2_permodality", "clustering", "embedding_clusters_amputation.csv"), index=False)
