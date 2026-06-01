import itertools
from itertools import combinations
import numpy as np
import pandas as pd
import os
from imml.cluster import NEMO, IMSR, LFIMVC, EEIMVC, SIMCADC
from imml.model_selection import MMSplitter
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_mutual_info_score
from sklearn.model_selection import ShuffleSplit


def get_consensus_matrix(df: pd.DataFrame, ids):
    connectivity_matrix = pd.DataFrame(0, index=ids, columns=ids)
    indicator_matrix = pd.DataFrame(0, index=ids, columns=ids)
    for index, row in df.iterrows():
        y_pred_idx = row['IDs']
        y_pred = row['Labels']
        for i in range(len(y_pred_idx)):
            patient_i = y_pred_idx[i]
            cluster_i = y_pred[i]
            for j in range(i + 1, len(y_pred_idx)):
                patient_j = y_pred_idx[j]
                cluster_j = y_pred[j]
                indicator_matrix.loc[patient_i, patient_j] += 1
                indicator_matrix.loc[patient_j, patient_i] += 1
                if cluster_i == cluster_j:
                    connectivity_matrix.loc[patient_i, patient_j] += 1
                    connectivity_matrix.loc[patient_j, patient_i] += 1
    consensus_matrix = connectivity_matrix.div(indicator_matrix).fillna(0)
    return consensus_matrix


MAX_N_CLUSTERS = 3
n_clusters_list = list(range(2, MAX_N_CLUSTERS+1))
RANDOM_STATE = 42
N_REPETITIONS = 3
ALGS = [
    "NEMO",
    "IMSR",
    "LFIMVC",
    "EEIMVC",
    "SIMCADC"
]
llm_folders = "embeddings"
llms = os.listdir(llm_folders)
cluster_stability = []
for llm in llms:
    llm_folder = os.path.join(llm_folders, llm)
    for file in os.listdir(llm_folder):
        if file.endswith("dual_embeddings.pkl"):
            Xs = pd.read_pickle(os.path.join(llm_folder, file))
            Xs = pd.DataFrame.from_dict(Xs, orient="index")
            half = Xs.shape[1] // 2
            Xs = [Xs.iloc[:, :half], Xs.iloc[:, -half:]]
            for X in Xs:
                X[X.eq(0).all(axis=1)] = np.nan
            for n_cluster in n_clusters_list:
                splitter = ShuffleSplit(n_splits=N_REPETITIONS, test_size=0.2, random_state=RANDOM_STATE)
                splitter = MMSplitter(splitter=splitter)
                for fold, (train_Xs, _) in enumerate(splitter.split(Xs)):
                    for alg in ALGS:
                        print(f"{llm=}, {n_cluster=}, {fold=}, {alg=}")
                        pipeline = eval(alg)
                        pipeline = pipeline(n_clusters=n_cluster, random_state=RANDOM_STATE + fold)
                        labels = pipeline.fit_predict(train_Xs)
                        result_dict = {
                            "Algorithm": alg,
                            "Clusters": n_cluster,
                            "Fold": fold,
                            "IDs": train_Xs[0].index.to_list(),
                            "LLM": llm,
                            "Labels": labels,
                            "N": len(train_Xs[0]),
                            "M": train_Xs[0].shape[1],
                        }
                        cluster_stability.append(result_dict)
    break

cluster_stability = pd.DataFrame(cluster_stability)
cluster_stability["AMI"] = 100.
clusters = pd.DataFrame([], index=Xs[0].index.to_list())
for alg,llm,n_cluster in itertools.product(ALGS, llms, n_clusters_list):
    mask = (cluster_stability["Algorithm"] == alg) & (cluster_stability["LLM"] == llm) & (cluster_stability["Clusters"] == n_cluster)
    cl_result = cluster_stability.loc[mask]
    for row1, row2 in combinations(cl_result.index, 2):
        row1 = cl_result.loc[row1]
        row2 = cl_result.loc[row2]
        row1 = pd.Series(row1["Labels"], index=row1["IDs"])
        row2 = pd.Series(row2["Labels"], index=row2["IDs"])
        common_ids = row1.index.intersection(row2.index)
        row1 = row1.loc[common_ids]
        row2 = row2.loc[common_ids]
        cluster_stability.loc[mask, "AMI"] = adjusted_mutual_info_score(row1, row2)
    consensus_matrix = get_consensus_matrix(cl_result, ids=Xs[0].index.to_list())
    labels = KMeans(n_clusters=n_cluster, random_state=RANDOM_STATE).fit_predict(consensus_matrix)
    labels = pd.Series(labels, index=Xs[0].index.to_list())
    clusters[f"{llm}_{alg}_{n_cluster}"] = labels

cluster_stability.to_csv(os.path.join("results_v2_permodality", "clustering", "cluster_stability.csv"), index=False)
clusters.to_csv(os.path.join("results_v2_permodality", "clustering", "clusters.csv"))

best_solution = cluster_stability.groupby(["Algorithm", "LLM", "Clusters"])["AMI"] \
    .mean().sort_values(ascending=False).reset_index().iloc[0]
best_solution = cluster_stability[
    (cluster_stability["Algorithm"] == best_solution.loc["Algorithm"]) &
    (cluster_stability["LLM"] == best_solution.loc["LLM"]) &
    (cluster_stability["Clusters"] == best_solution.loc["Clusters"])
]
consensus_matrix = get_consensus_matrix(best_solution, ids=Xs[0].index.to_list())
alg = best_solution["Algorithm"].iloc[0]
llm = best_solution["LLM"].iloc[0]
n_cluster = best_solution["Clusters"].iloc[0]
consensus_matrix.to_csv(os.path.join("results_v2_permodality", "clustering",
                                     f"{llm}_{alg}_{n_cluster}_consensus_matrix.csv"))