import itertools
import os
from re import S
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import adjusted_mutual_info_score
from tqdm.auto import tqdm


def run_stability(llm, alg, n_cluster, p, strategy):
    mask = (
            (embedding_clusters_amputation["Algorithm"] == alg) &
            (embedding_clusters_amputation["LLM"] == llm) &
            (embedding_clusters_amputation["Clusters"] == n_cluster)
            )
    cl_result = embedding_clusters_amputation.loc[mask]
    missing_p = round(float(p), 1)
    reference_cl_result = cl_result.loc[cl_result["Incomplete samples (%)"] == 0]
    cl_result = cl_result.loc[
        (embedding_clusters_amputation["Strategy"] == strategy) &
        (cl_result["Incomplete samples (%)"] == missing_p)
    ]
    amis = []
    for row1, reference_cl in itertools.product(cl_result.index, reference_cl_result.index):
        row1 = cl_result.loc[row1]
        reference_cl = reference_cl_result.loc[reference_cl]
        row1 = pd.Series(row1["Labels"], index=row1["IDs"])
        reference_cl = pd.Series(reference_cl["Labels"], index=reference_cl["IDs"])
        common_ids = row1.index.intersection(reference_cl.index)
        row1 = row1.loc[common_ids]
        reference_cl = reference_cl.loc[common_ids]
        score = adjusted_mutual_info_score(row1, reference_cl)
        amis.append(score)
    cluster_robustness = {
        "LLM": llm,
        "Algorithm": alg,
        "Clusters": n_cluster,
        "Strategy": strategy,
        "Incomplete samples (%)": missing_p,
        "Folds": len(cl_result),
        "AMIs": amis,
        "Robustness": np.mean(amis),
    }
    return cluster_robustness


embedding_clusters_amputation = pd.read_csv(os.path.join("results_v2_permodality",
                                                          "clustering",
                                                          "embedding_clusters_amputation.csv"
                                                          ))
embedding_clusters_amputation["IDs"] = embedding_clusters_amputation["IDs"].apply(lambda x: eval(x))
embedding_clusters_amputation["Labels"] = embedding_clusters_amputation["Labels"].apply(lambda x: [int(i) for i in x.strip("[]").split()])
llms = embedding_clusters_amputation["LLM"].unique()
n_clusters_list = embedding_clusters_amputation["Clusters"].unique()
algs = embedding_clusters_amputation["Algorithm"].unique()
ps = embedding_clusters_amputation["Incomplete samples (%)"].unique()
strategies = embedding_clusters_amputation["Strategy"].unique()
n_jobs = None
cluster_robustness = Parallel(n_jobs=n_jobs, return_as="generator_unordered")(
    delayed(run_stability)(llm=llm, n_cluster=n_cluster, alg=alg, p=p, strategy=strategy)
    for llm,n_cluster,alg,p,strategy  in tqdm(itertools.product(llms, n_clusters_list, algs, ps, strategies))
)

cluster_robustness = [stability_dict for stability_dict in cluster_robustness]
cluster_robustness = pd.DataFrame(cluster_robustness).sort_values("Robustness", ascending=False)
cluster_robustness.to_csv(os.path.join("results_v2_permodality", "clustering", "cluster_robustness.csv"), index=False)
