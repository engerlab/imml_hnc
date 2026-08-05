import itertools
import os
import pandas as pd
import numpy as np
from lifelines import KaplanMeierFitter
from lifelines.plotting import add_at_risk_counts
from lifelines.statistics import multivariate_logrank_test, pairwise_logrank_test
from lifelines.utils import restricted_mean_survival_time
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency, kruskal, norm
import seaborn as sns
from tqdm.auto import tqdm
from joblib import Parallel, delayed

from src.clustering.settings import ALGS_DICT, N_CLUSTERS_LIST, llms


def rmst_variance(kmf, tau):
    """
    Greenwood plug-in variance estimator used by survRM2.
    """

    et = kmf.event_table

    # event times
    times = et.index.values

    observed = et["observed"].values
    at_risk = et["at_risk"].values

    # KM step function
    sf = kmf.survival_function_

    sf_times = sf.index.values
    sf_values = sf.iloc[:, 0].values

    var = 0.0

    # event times only
    for t, d, y in zip(times, observed, at_risk):

        if d == 0:
            continue

        if t >= tau:
            break

        # Integral from t to tau of S(u) du
        grid = np.concatenate(([t], sf_times[(sf_times > t) & (sf_times < tau)], [tau]))
        surv = np.interp(grid[:-1], sf_times, sf_values)

        integral = np.sum((grid[1:] - grid[:-1]) * surv)

        var += integral**2 * d / (y * (y - d))

    return var


def get_p_rmst(diff, kmf1, kmf0, tau):
    var = rmst_variance(kmf1, tau) + rmst_variance(kmf0, tau)
    se = np.sqrt(var)
    z = diff / se
    p = 2 * norm.sf(abs(z))
    return p


def apply_evaluate_clinical_relevance(llm, n_cluster, alg):
    mask = (
            (embedding_clusters_permutation["Algorithm"] == alg) &
            (embedding_clusters_permutation["LLM"] == llm) &
            (embedding_clusters_permutation["Clusters"] == n_cluster)
            )
    cl_result = embedding_clusters_permutation.loc[mask]
    relevance_clusters_permutation_list = []
    for _, row in cl_result.iterrows():
        relevance_clusters_permutation = {
            "LLM": llm,
            "Algorithm": alg,
            "Clusters": n_cluster,
            "Fold": row["Fold"],
        }
        row = pd.Series(row["Labels"], index=row["IDs"])
        row.index = row.index.astype(int)
        relevance_clusters_permutation = evaluate_clinical_relevance(row=row, clinical_relevance=relevance_clusters_permutation)
        relevance_clusters_permutation_list.append(relevance_clusters_permutation)
    name = f"{llm}_{alg}_{n_cluster}"
    row = embedding_clusters[name]
    relevance_embedding_clusters = {
        "LLM": llm,
        "Algorithm": alg,
        "Clusters": n_cluster,
    }
    relevance_embedding_clusters = evaluate_clinical_relevance(row=row, clinical_relevance=relevance_embedding_clusters)

    return relevance_clusters_permutation_list, relevance_embedding_clusters


def evaluate_clinical_relevance(row, clinical_relevance):
    for clinical_endpoint in clinical_endpoints:
        t = clinical_endpoints[clinical_endpoint]["t"]
        e = clinical_endpoints[clinical_endpoint]["e"]
        clinical_data_endpoint = clinical_data[clinical_data[t].notna() & clinical_data[e].notna()]
        clinical_data_endpoint = clinical_data_endpoint.sort_values(by=t, ascending=False)
        clinical_data_endpoint = clinical_data_endpoint.drop_duplicates(subset='patient_id', keep='first')
        clinical_data_endpoint = clinical_data_endpoint.set_index("patient_id")
        mask = row.index.intersection(clinical_data_endpoint.index)
        clusters_for_var = row.loc[mask]
        if len(clusters_for_var.unique()) < 2:
            continue
        clinical_data_endpoint = clinical_data_endpoint.loc[mask]

        log_rank_test = multivariate_logrank_test(event_durations=clinical_data_endpoint[t], groups=clusters_for_var, events=clinical_data_endpoint[e])
        log_rank_test = log_rank_test.p_value
        pairwise_logrank_test_p_value = pairwise_logrank_test(event_durations=clinical_data_endpoint[t],
                                                            groups=clusters_for_var,
                                                            events=clinical_data_endpoint[e])
        pairwise_logrank_test_p_value = pairwise_logrank_test_p_value.p_value
        clinical_relevance[f"{clinical_endpoint}_Logrank_pval"] = log_rank_test
        clinical_relevance[f"{clinical_endpoint}_Pairwise_Logrank_pval"] = pairwise_logrank_test_p_value
        clinical_relevance[f"{clinical_endpoint}_N"] = len(clusters_for_var)
        clinical_relevance[f"{clinical_endpoint}_Logrank_pval"] = log_rank_test

        for timepoint in timepoints:
            mask = (clinical_data_endpoint[t] < timepoint) & (clinical_data_endpoint[e] == 1)
            log_rank_test = multivariate_logrank_test(event_durations=clinical_data_endpoint[t], groups=clusters_for_var, events=clinical_data_endpoint[e])
            log_rank_test = log_rank_test.p_value
            clinical_relevance[f"{clinical_endpoint}_Logrank_pval({timepoint})"] = log_rank_test
            clinical_relevance[f"{clinical_endpoint}_N({timepoint})"] = len(clusters_for_var)


        kmfs = []
        for label in sorted(clusters_for_var.unique()):
            cl = (clusters_for_var == label)
            kmf_cl = KaplanMeierFitter().fit(
                durations=clinical_data_endpoint[t][cl],
                event_observed=clinical_data_endpoint[e][cl],
                label=f"Cluster {label}"
            )
            kmf_others = KaplanMeierFitter().fit(
                durations=clinical_data_endpoint[t][~cl],
                event_observed=clinical_data_endpoint[e][~cl],
                label="Others"
            )
            kmfs.append(kmf_cl)
            for timepoint in timepoints:
                rmst = restricted_mean_survival_time(kmf_cl, t=timepoint) - restricted_mean_survival_time(kmf_others, t=timepoint)
                clinical_relevance[f"{clinical_endpoint}_{label}_RMST({timepoint})"] = rmst
                pval = get_p_rmst(diff=rmst, kmf1=kmf_cl, kmf0=kmf_others, tau=timepoint)
                clinical_relevance[f"{clinical_endpoint}_{label}_RMST_pval({timepoint})"] = pval


    for clinical_variable in discrete_clinical_variables:
        clinical_variable = clinical_data_endpoint[clinical_variable]
        if clinical_variable.name == "p16_Status":
            clinical_variable = clinical_variable[clinical_variable != "Not Inferred"]
            clinical_variable = pd.Series(pd.factorize(clinical_variable)[0], index=clinical_variable.index,
                                        name=clinical_variable.name)
        elif clinical_variable.name in ["HPV_Status", "Sex", "Alcohol_Consumption", "Smoking_History"]:
            clinical_variable = clinical_variable[clinical_variable.notna()]
            clinical_variable = pd.Series(pd.factorize(clinical_variable)[0], index=clinical_variable.index,
                                        name=clinical_variable.name)
        elif clinical_variable.name in ["Surgery", "chemotherapy"]:
            clinical_variable = clinical_variable[clinical_variable != -1]
        mask = row.index.intersection(clinical_variable.index)
        clusters_for_var = row.loc[mask]
        if len(clusters_for_var.unique()) < 2:
            continue
        contingency_table = pd.crosstab(index=clusters_for_var, columns=clinical_variable)
        p_val = chi2_contingency(contingency_table)[1]
        clinical_relevance[f"{clinical_variable.name}_pval"] = p_val
    
    for clinical_variable in continuous_clinical_variables:
        clinical_variable = clinical_data_endpoint[clinical_variable]
        if clinical_variable.name == "Age":
            clinical_variable = clinical_variable[clinical_variable.between(0, 100)]
        mask = row.index.intersection(clinical_variable.index)
        clusters_for_var = row.loc[mask]
        if len(clusters_for_var.unique()) < 2:
            continue
        p_val = kruskal(*[clinical_variable[clusters_for_var == label] for label in sorted(clusters_for_var.unique())])[1]
        clinical_relevance[f"{clinical_variable.name}_pval"] = p_val

    return clinical_relevance


if __name__ == '__main__':
    embedding_clusters_permutation = os.path.join("results_v2_permodality", "clustering", "embedding_clusters_permutation.csv")
    embedding_clusters_permutation = pd.read_csv(embedding_clusters_permutation)
    embedding_clusters = pd.read_csv(os.path.join("results_v2_permodality", "clustering", "embedding_clusters.csv"), index_col=0)
    clinical_data = pd.read_csv(os.path.join("outcomes", "all_fields_processed.csv"))
    embedding_clusters_permutation["IDs"] = embedding_clusters_permutation["IDs"].apply(lambda x: eval(x))
    embedding_clusters_permutation["Labels"] = embedding_clusters_permutation["Labels"].apply(lambda x: eval(x))

    clinical_relevance_clusters = []
    iterators = itertools.product(llms, N_CLUSTERS_LIST, ALGS_DICT.keys())
    iterators = list(iterators)

    n_jobs = 20
    clinical_relevance_clusters = Parallel(n_jobs=n_jobs, return_as="generator_unordered")(
        delayed(apply_evaluate_clinical_relevance)(llm=llm, n_cluster=n_cluster, alg=alg)
        for llm,n_cluster,alg  in tqdm(iterators, total=len(iterators))
    )
    clinical_relevance_clusters = list(clinical_relevance_clusters)

    relevance_embedding_clusters = [clinical_relevance_dict[1] for clinical_relevance_dict in clinical_relevance_clusters]
    relevance_embedding_clusters = pd.DataFrame(relevance_embedding_clusters)
    relevance_embedding_clusters.to_csv(os.path.join("results_v2_permodality", "clustering", "relevance_embedding_clusters.csv"), index=False)

    relevance_clusters_permutation = [clinical_relevance_dict for relevance_clusters_permutation_list in clinical_relevance_clusters for clinical_relevance_dict in relevance_clusters_permutation_list[0]]
    relevance_clusters_permutation = pd.DataFrame(relevance_clusters_permutation)
    relevance_clusters_permutation.to_csv(os.path.join("results_v2_permodality", "clustering", "relevance_clusters_permutation.csv"), index=False)

