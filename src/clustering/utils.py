import os
import numpy as np
import pandas as pd

from src.clustering.settings import LLM_FOLDERS


def get_consensus_matrix(df: pd.DataFrame, ids):
    n = len(ids)
    id_to_pos = {id_: i for i, id_ in enumerate(ids)}
    connectivity = np.zeros((n, n), dtype=np.uint16)
    indicator = np.zeros((n, n), dtype=np.uint16)
    for _, row in df.iterrows():
        idx = np.fromiter(
            (id_to_pos[x] for x in row["IDs"]),
            dtype=np.int32
        )
        labels = np.asarray(row["Labels"])
        r, c = np.triu_indices(len(idx), 1)
        indicator[idx[r], idx[c]] += 1
        indicator[idx[c], idx[r]] += 1
        same = labels[r] == labels[c]
        rr = idx[r[same]]
        cc = idx[c[same]]
        connectivity[rr, cc] += 1
        connectivity[cc, rr] += 1
    consensus_matrix = np.divide(
        connectivity,
        indicator,
        out=np.zeros_like(connectivity, dtype=np.float32),
        where=indicator > 0
    )
    consensus_matrix = pd.DataFrame(consensus_matrix, index=ids, columns=ids)
    return consensus_matrix


def get_Xs_ids(llm):
    llm_folder = os.path.join(LLM_FOLDERS, llm)
    for file in os.listdir(llm_folder):
        if file.endswith("dual_embeddings.pkl"):
            Xs = pd.read_pickle(os.path.join(llm_folder, file))
            Xs = pd.DataFrame.from_dict(Xs, orient="index")
            ids = Xs[0].index.to_list()
            break
    return Xs, ids


def prepare_Xs(Xs):
    half = Xs.shape[1] // 2
    Xs = [Xs.iloc[:, :half], Xs.iloc[:, -half:]]
    for X in Xs:
        X[X.eq(0).all(axis=1)] = np.nan
    return Xs
