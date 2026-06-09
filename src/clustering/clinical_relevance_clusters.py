import os
import pandas as pd
from lifelines import KaplanMeierFitter
from lifelines.plotting import add_at_risk_counts
from lifelines.statistics import multivariate_logrank_test, pairwise_logrank_test
from lifelines.utils import restricted_mean_survival_time
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency, kruskal
import seaborn as sns

cluster_stability = pd.read_csv(os.path.join("results_v2_permodality", "clustering", "cluster_stability.csv"), index_col=0)
best_clustering = pd.DataFrame()
for llm in cluster_stability["LLM"].unique():
    cluster_stability_llm = cluster_stability[cluster_stability["LLM"] == llm]
    cluster_stability_llm = cluster_stability_llm.loc[[cluster_stability_llm['AMI'].idxmax()]]
    best_clustering = pd.concat([best_clustering, cluster_stability_llm])

clusters = pd.read_csv(os.path.join("results_v2_permodality", "clustering", "embedding_clusters.csv"), index_col=0)
clinical_data = pd.read_csv(os.path.join("outcomes", "all_fields_processed.csv"))

clinical_endpoints = {
    "OS": {
        "t": "OS",
        "e": "Death",
    },
    "DFS": {
        "t": "DFS",
        "e": "Failure",
    },
}
discrete_clinical_variables = [
    "p16_Status",
    "HPV_Status",
    "Sex",
    "Smoking_History",
    "Alcohol_Consumption",
    "Surgery",
]
continuous_clinical_variables = [
    "Age",
]

timepoints = [24*i for i in range(1, 10)]
clinical_relevance_clusters = []
for llm in cluster_stability["LLM"].unique():
    best_row = best_clustering[best_clustering["LLM"] == llm]
    assert len(best_row) == 1
    best_row = best_row.iloc[0]
    clusters_llm = clusters[f"{llm}_{best_row.loc['Algorithm']}_{best_row.loc['Clusters']}"]
    result = {
        "LLM": llm,
        "Algorithm": best_row.loc["Algorithm"],
        "Clusters": best_row.loc["Clusters"],
        "AMI": best_row.loc["AMI"],
    }
    for clinical_endpoint in clinical_endpoints:
        t = clinical_endpoints[clinical_endpoint]["t"]
        e = clinical_endpoints[clinical_endpoint]["e"]
        clinical_data_endpoint = clinical_data[clinical_data[t].notna() & clinical_data[e].notna()]
        clinical_data_endpoint = clinical_data_endpoint.sort_values(by=t, ascending=False)
        clinical_data_endpoint = clinical_data_endpoint.drop_duplicates(subset='patient_id', keep='first')
        clinical_data_endpoint = clinical_data_endpoint.set_index("patient_id")
        clusters_llm = clusters_llm.loc[clinical_data_endpoint.index]

        log_rank_test = multivariate_logrank_test(event_durations=clinical_data_endpoint[t],
                                                  groups=clusters_llm,
                                                  events=clinical_data_endpoint[e])
        log_rank_test = log_rank_test.p_value
        pairwise_logrank_test_p_value = pairwise_logrank_test(event_durations=clinical_data_endpoint[t],
                                                              groups=clusters_llm,
                                                              events=clinical_data_endpoint[e])
        pairwise_logrank_test_p_value = pairwise_logrank_test_p_value.p_value
        result[f"{clinical_endpoint}_Logrank_test"] = log_rank_test
        result[f"{clinical_endpoint}_Pairwise_Logrank_test"] = pairwise_logrank_test_p_value
        result[f"{clinical_endpoint}_N"] = len(clusters_llm)
        result[f"{clinical_endpoint}_Logrank_test"] = log_rank_test

        ax = plt.subplot()
        kmfs = []
        for label in sorted(clusters_llm.unique()):
            cl = (clusters_llm == label)
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
            ax = kmf_cl.plot_survival_function(ax=ax)
            kmfs.append(kmf_cl)
            for timepoint in timepoints:
                result[f"{clinical_endpoint}_{label}_RMST({timepoint})"] = restricted_mean_survival_time(kmf_cl, t=timepoint) - restricted_mean_survival_time(kmf_others, t=timepoint)
        add_at_risk_counts(*kmfs, ax=ax)
        plt.tight_layout()
        plt.clf()

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
        clusters_for_var = clusters_llm.loc[clinical_variable.index]
        contingency_table = pd.crosstab(index=clusters_for_var, columns=clinical_variable)
        p_val = chi2_contingency(contingency_table)[1]
        result[clinical_variable.name] = p_val
        for clinical_variable in continuous_clinical_variables:
            clinical_variable = clinical_data_endpoint[clinical_variable]
            if clinical_variable.name == "Age":
                clinical_variable = clinical_variable[clinical_variable.between(0, 100)]
            clusters_for_var = clusters_llm.loc[clinical_variable.index]
            p_val = kruskal(*[clinical_variable[clusters_for_var == label] for label in sorted(clusters_for_var.unique())])[1]
            result[clinical_variable.name] = p_val
            if p_val < 0.05:
                clusters_for_var = pd.concat([clusters_for_var, clinical_variable], axis=1)
                sns.violinplot(x=clusters_for_var.columns[0], y=clinical_variable.name,
                               data=clusters_for_var)
                plt.clf()

    clinical_relevance_clusters.append(result)

clinical_relevance_clusters = pd.DataFrame(clinical_relevance_clusters)
clinical_relevance_clusters.to_csv(os.path.join("results_v2_permodality", "clustering", "clinical_relevance_clusters.csv"), index=None)

