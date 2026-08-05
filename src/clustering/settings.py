import os

import numpy as np
from imml.cluster import NEMO, IMSR, LFIMVC, EEIMVC, SIMCADC, OPIMC, PIMVC

MAX_N_CLUSTERS = 10
N_CLUSTERS_LIST = list(range(2, MAX_N_CLUSTERS + 1))
RANDOM_STATE = 42
N_REPETITIONS = 50
ALGS_DICT = {
    "NEMO": NEMO,
    "IMSR": IMSR,
    "LFIMVC": LFIMVC,
    "EEIMVC": EEIMVC,
    "SIMCADC": SIMCADC,
    "OPIMC": OPIMC,
    "PIMVC": PIMVC,
}
ALGS = list(ALGS_DICT.keys())
LLM_FOLDERS = "embeddings"
llms = os.listdir(LLM_FOLDERS)
ps = np.arange(0., 1., 0.2)

best_pipeline = {
    "llm": "smb8b_qwen3_mo",
    "alg": "NEMO",
    "n_cluster": "2",
}
best_pipeline["name"] = f"{best_pipeline['llm']}_{best_pipeline['alg']}_{best_pipeline['n_cluster']}"

clinical_endpoints = {
    "OS": {
        "t": "OS",
        "e": "Death",
        "description": "Overall Survival"
    },
    "DistFS": {
        "t": "DistFS",
        "e": "distant failure",
        "description": "Distant Failure"
    },
    "DFS": {
        "t": "DFS",
        "e": "Failure",
        "description": "Disease-Free Survival"
    },
    "LFS": {
        "t": "LFS",
        "e": "Locoregional_Failure",
        "description": "Locoregional Failure"
    },
}
discrete_clinical_variables = [
    "HPV_Status",
    "Sex",
    "Smoking_History",
    "Alcohol_Consumption",
    "Clinical_T_Stage",
    "Clinical_N_Stage",
]

continuous_clinical_variables = [
    "Age",
]

timepoints = [24*i for i in range(1, 10)]

final_cluster_name = "Cluster (path+cons)"

rename_cols_dict = {
    "Clinical_N_Stage": "N-stage",
    "Clinical_T_Stage": "T-stage",
    "HPV_Status": "HPV Status",
    "Alcohol_Consumption": "Alcohol Consumption",
    "Smoking_History": "Smoking History",
    "Sex": "Sex",
    final_cluster_name: final_cluster_name,
    "Cluster (path+notes)": final_cluster_name,
}


