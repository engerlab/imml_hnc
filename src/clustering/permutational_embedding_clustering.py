from itertools import product
import numpy as np
import pandas as pd
import os
from imml.model_selection import MMSplitter
from joblib import Parallel, delayed
from sklearn.model_selection import ShuffleSplit
from tqdm.auto import tqdm
from datetime import datetime
from sklearn.metrics import silhouette_score

from src.clustering.utils import get_Xs_ids, prepare_Xs
from src.clustering.settings import llms, N_CLUSTERS_LIST, N_REPETITIONS, RANDOM_STATE, ALGS_DICT


def run_clustering(llm, alg, pipeline, n_cluster):
    Xs = llm_Xs[llm]
    Xs, ids = Xs["Xs"], Xs["ids"]
    results_list = []
    splitter = ShuffleSplit(n_splits=N_REPETITIONS, test_size=0.2, random_state=RANDOM_STATE)
    splitter = MMSplitter(splitter=splitter)
    print(f"{llm=}, {alg=}, {n_cluster=}, {datetime.now()}")
    for fold, (train_Xs, _) in enumerate(splitter.split(Xs)):
        model = pipeline(n_clusters=n_cluster, random_state=RANDOM_STATE + fold)
        labels = model.fit_predict(train_Xs)
        embedding = model.embedding_
        if len(np.unique(labels)) == 1:
            sil = np.nan
        else:
            sil = silhouette_score(labels=labels, X=embedding, random_state=RANDOM_STATE)
        result_dict = {
            "LLM": llm,
            "Algorithm": alg,
            "Clusters": n_cluster,
            "Fold": fold,
            "IDs": train_Xs[0].index.to_list(),
            "Labels": labels.tolist(),
            "Silhouette": sil,
            "N": len(train_Xs[0]),
            "M": train_Xs[0].shape[1],
        }
        results_list.append(result_dict)
    return results_list

llm_Xs = {}
for llm in llms:
    Xs, ids = get_Xs_ids(llm=llm)
    Xs = prepare_Xs(Xs)
    llm_Xs[llm] = {
        "Xs": Xs,
        "ids": ids,
    }
n_jobs = 20
embedding_clusters_permutation = Parallel(n_jobs=n_jobs, return_as="generator_unordered")(
    delayed(run_clustering)(llm=llm, n_cluster=n_cluster, alg=alg, pipeline=pipeline)
    for llm,n_cluster,(alg, pipeline)  in tqdm(product(llms, N_CLUSTERS_LIST, ALGS_DICT.items()))
)
embedding_clusters_permutation = [clusters_dict for clusters_l in embedding_clusters_permutation for clusters_dict in clusters_l]
embedding_clusters_permutation = pd.DataFrame(embedding_clusters_permutation)
embedding_clusters_permutation.to_csv(os.path.join("results_v2_permodality", "clustering", "embedding_clusters_permutation.csv"), index=False)
