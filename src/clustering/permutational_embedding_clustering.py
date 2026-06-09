from itertools import product
import numpy as np
import pandas as pd
import os
from imml.model_selection import MMSplitter
from joblib import Parallel, delayed
from sklearn.model_selection import ShuffleSplit
from tqdm.auto import tqdm

from src.clustering.utils import get_Xs_ids
from src.clustering.settings import llms, N_CLUSTERS_LIST, N_REPETITIONS, RANDOM_STATE, ALGS_DICT


def run_clustering(llm, alg, pipeline, n_cluster):
    Xs,ids = get_Xs_ids(llm=llm)
    half = Xs.shape[1] // 2
    Xs = [Xs.iloc[:, :half], Xs.iloc[:, -half:]]
    for X in Xs:
        X[X.eq(0).all(axis=1)] = np.nan
    results_list = []
    splitter = ShuffleSplit(n_splits=N_REPETITIONS, test_size=0.2, random_state=RANDOM_STATE)
    splitter = MMSplitter(splitter=splitter)
    for fold, (train_Xs, _) in enumerate(splitter.split(Xs)):
        print(f"{llm=}, {alg=}, {n_cluster=}, {fold=}")
        model = pipeline(n_clusters=n_cluster, random_state=RANDOM_STATE + fold)
        labels = model.fit_predict(train_Xs)
        result_dict = {
            "LLM": llm,
            "Algorithm": alg,
            "Clusters": n_cluster,
            "Fold": fold,
            "IDs": train_Xs[0].index.to_list(),
            "Labels": labels,
            "N": len(train_Xs[0]),
            "M": train_Xs[0].shape[1],
        }
        results_list.append(result_dict)
    return results_list

n_jobs = 20
embedding_clusters_permutation = Parallel(n_jobs=n_jobs)(
    delayed(run_clustering)(llm=llm, n_cluster=n_cluster, alg=alg, pipeline=pipeline)
    for llm,n_cluster,(alg, pipeline)  in tqdm(product(llms, N_CLUSTERS_LIST, ALGS_DICT.items()))
)
embedding_clusters_permutation = [clusters_dict for clusters_l in embedding_clusters_permutation for clusters_dict in clusters_l]
embedding_clusters_permutation = pd.DataFrame(embedding_clusters_permutation)
embedding_clusters_permutation.to_csv(os.path.join("results_v2_permodality", "clustering", "embedding_clusters_permutation.csv"), index=False)