import os
import pandas as pd

from src.clustering.utils import get_consensus_matrix, get_Xs_ids

embedding_clusters_permutation = pd.read_csv(os.path.join("results_v2_permodality",
                                                          "clustering",
                                                          "embedding_clusters_permutation.csv"
                                                          ))
embedding_clusters_permutation["IDs"] = embedding_clusters_permutation["IDs"].apply(lambda x: eval(x))
embedding_clusters_permutation["Labels"] = embedding_clusters_permutation["Labels"].apply(lambda x: [int(i) for i in x.strip("[]").split()])
cluster_stability = pd.read_csv(os.path.join("results_v2_permodality",
                                             "clustering",
                                             "cluster_stability.csv"
                                             ))

best_solution = cluster_stability.groupby(["Algorithm", "LLM", "Clusters"])["AMI"] \
    .mean().sort_values(ascending=False)
print(best_solution)
best_solution = cluster_stability.reset_index().iloc[0]
best_solution = embedding_clusters_permutation[
    (embedding_clusters_permutation["Algorithm"] == best_solution.loc["Algorithm"]) &
    (embedding_clusters_permutation["LLM"] == best_solution.loc["LLM"]) &
    (embedding_clusters_permutation["Clusters"] == best_solution.loc["Clusters"])
]
alg = best_solution["Algorithm"].iloc[0]
llm = best_solution["LLM"].iloc[0]
n_cluster = best_solution["Clusters"].iloc[0]
_, ids = get_Xs_ids(llm)
consensus_matrix = get_consensus_matrix(best_solution, ids=ids)
consensus_matrix.to_csv(os.path.join("results_v2_permodality", "clustering",
                                     f"{llm}_{alg}_{n_cluster}_consensus_matrix.csv"))

