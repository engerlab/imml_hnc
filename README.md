# LLM-based multimodal data integration with missing modalities improves outcome prediction in head and neck cancer

This is the official repository accompanying the paper:

> **LLM-based multimodal data integration with missing modalities improves outcome prediction in head and neck cancer**

To reproduce the results presented in the paper, follow the steps below.

---

## Requirements

#### 1. Clone the repository

```bash
git clone https://github.com/engerlab/imml_hnc.git
cd imml_hnc
```

#### 2. Create a Python environment

###### Option 1: pyenv + virtualenv

```bash
pyenv install 3.13
pyenv virtualenv 3.13 imml_hnc
pyenv activate imml_hnc
```

###### Option 2: Conda

```bash
conda create -n imml_hnc python=3.13
conda activate imml_hnc
```

#### 3. Install the dependencies

```bash
python -m pip install -r requirements.txt
```

---

## Reproducing the results

Make sure your working directory is the root of the repository.

Run the following scripts in order:

```bash
python permutational_embedding_clustering.py
python amputation_embedding_clustering.py
python cluster_stability.py
python cluster_robustness.py
python clinical_relevance_clusters.py
python cluster_clinical_integration.py
python pid_analysis.py
```

After all scripts have finished, the generated CSV files will be available in the `results_v2_permodality/` directory.

---

## Generating the figures

Launch JupyterLab (or Jupyter notebook) from the repository root:

```bash
jupyter lab
```

Run all cells in the following notebooks (in the `notebooks/` directory) to reproduce the figures presented in the paper:

- `evaluate_cluster_permutations.ipynb`
- `robustness.ipynb`
- `relevance_clusters.ipynb`
- `cox_figures.ipynb`
- `text_analysis.ipynb`