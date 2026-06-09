import os
from imml.cluster import NEMO, IMSR, LFIMVC, EEIMVC, SIMCADC

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
}
ALGS = list(ALGS_DICT.keys())
LLM_FOLDERS = "embeddings"
llms = os.listdir(LLM_FOLDERS)
