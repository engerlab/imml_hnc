import itertools
import os

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_mutual_info_score
from tqdm.auto import tqdm

from src.clustering.settings import RANDOM_STATE, llms, N_CLUSTERS_LIST, ALGS_DICT
from src.clustering.utils import get_consensus_matrix, get_Xs_ids

def run_stability(llm, alg, n_cluster):
    name = f"{llm}_{alg}_{n_cluster}"
    mask = (
            (embedding_clusters_permutation["Algorithm"] == alg) &
            (embedding_clusters_permutation["LLM"] == llm) &
            (embedding_clusters_permutation["Clusters"] == n_cluster)
            )
    cl_result = embedding_clusters_permutation.loc[mask]
    ids = ids_cache[llm]
    amis = []
    for row1, row2 in itertools.combinations(cl_result.index, 2):
        row1 = cl_result.loc[row1]
        row2 = cl_result.loc[row2]
        row1 = pd.Series(row1["Labels"], index=row1["IDs"])
        row2 = pd.Series(row2["Labels"], index=row2["IDs"])
        common_ids = row1.index.intersection(row2.index)
        row1 = row1.loc[common_ids]
        row2 = row2.loc[common_ids]
        score = adjusted_mutual_info_score(row1, row2)
        amis.append(score)
    cluster_stability = {
        "LLM": llm,
        "Algorithm": alg,
        "Clusters": n_cluster,
        "Folds": len(cl_result),
        "AMIs": amis,
        "AMI": np.mean(amis),
    }
    consensus_matrix = get_consensus_matrix(cl_result, ids=ids)
    labels = KMeans(n_clusters=n_cluster, random_state=RANDOM_STATE).fit_predict(consensus_matrix)
    labels = pd.Series(labels, index=ids, name=name)
    return cluster_stability, labels


embedding_clusters_permutation = pd.read_csv(os.path.join("results_v2_permodality",
                                                          "clustering",
                                                          "embedding_clusters_permutation.csv"
                                                          ))
embedding_clusters_permutation["IDs"] = embedding_clusters_permutation["IDs"].apply(lambda x: eval(x))
embedding_clusters_permutation["Labels"] = embedding_clusters_permutation["Labels"].apply(lambda x: [int(i) for i in x.strip("[]").split()])
ids_cache = {llm: get_Xs_ids(llm=llm)[1] for llm in llms}
n_jobs = 20
cluster_stability = Parallel(n_jobs=n_jobs)(
    delayed(run_stability)(llm=llm, n_cluster=n_cluster, alg=alg)
    for llm,n_cluster,alg  in tqdm(itertools.product(llms, N_CLUSTERS_LIST, ALGS_DICT.keys()))
)

embedding_clusters = [stability_dict[1] for stability_dict in cluster_stability]
cluster_stability = [stability_dict[0] for stability_dict in cluster_stability]
cluster_stability = pd.DataFrame(cluster_stability).sort_values("AMI", ascending=False)
embedding_clusters = pd.concat(embedding_clusters, axis=1)
cluster_stability.to_csv(os.path.join("results_v2_permodality", "clustering", "cluster_stability.csv"), index=False)
embedding_clusters.to_csv(os.path.join("results_v2_permodality", "clustering", "embedding_clusters.csv"))