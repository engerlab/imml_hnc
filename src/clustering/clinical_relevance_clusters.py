import os
import pandas as pd

labels = pd.read_csv(os.path.join("results_v2_permodality", "clustering", "labels.csv"))
clinical_data = pd.read_csv(os.path.join("results_v2_permodality", "outcomes", "all_fields_processed.csv"))

labels = labels.loc[clinical_data["patient_id"]]
