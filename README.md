# DNA Methylation Clock: Age Prediction from CpG Methylation Data

Comparison of linear regression and machine learning models for predicting chronological age from whole-blood CpG methylation profiles, demonstrating the proof of concept behind epigenetic aging clocks.

---

## Background

DNA methylation at CpG sites changes systematically with age, enabling the construction of "epigenetic clocks" — supervised models that predict chronological (and biological) age from methylation data. This project implements and benchmarks several regression models on a curated subset of Illumina 450K methylation array data, and additionally trains classification models to predict sex from the same features.

## Dataset

- **Source:** [GEO:GSE40279](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE40279) — whole-blood DNA methylation (Illumina 450K)
- **Samples:** 556 total → 456 development / 100 evaluation
- **Features:** 1,000 CpG sites selected by absolute Spearman correlation with age
- **Metadata:** age, sex, ethnicity
- **Age range:** 19–101 years (mean 64.5 ± 14.6)

## Project Structure

```
├── data/
│   ├── development.csv          # 456-sample development set
│   └── evaluation.csv           # 100-sample holdout test set
├── src/
│   ├── preprocessing.py         # Imputation, beta→M transform, scaling, encoding
│   ├── feature_selection.py     # Stability Selection & mRMR
│   ├── models.py                # Model definitions and training
│   └── evaluation.py            # Bootstrap evaluation and metrics
├── notebooks/
│   ├── 01_eda.ipynb             # Exploratory data analysis
│   ├── 02_feature_matrix.ipynb  # Ablation over feature combinations
│   ├── 03_baselines.ipynb       # Baseline model comparison
│   ├── 04_feature_selection.ipynb
│   ├── 05_hyperparameter_tuning.ipynb
│   ├── 06_classification.ipynb  # Sex prediction
│   └── 07_final_evaluation.ipynb
├── figures/
├── report/
│   ├── main.tex
│   └── bibliography.bib
├── requirements.txt
└── README.md
```

> **Note:** Adjust the tree above to match your actual file layout.

## Methods

### Preprocessing

1. **Missing value imputation** — median imputation for numerical features, most-frequent for categorical (3% MCAR)
2. **Beta → M-value transformation** — reduces heteroscedasticity inherent in bounded beta-values
3. **Standard scaling** — zero mean, unit variance per feature
4. **One-hot encoding** — for categorical metadata (sex, ethnicity)

All preprocessing is wrapped in a scikit-learn `Pipeline` fitted exclusively on the training split to prevent data leakage.

### Regression Models (Age Prediction)

| Model | Key Property |
|---|---|
| **OLS** | No regularisation; fast baseline, overfits when *p* ≫ *n* |
| **Elastic Net** | L1 + L2 regularisation; built-in feature selection |
| **SVR** | Kernel-based; handles high-dimensional, non-linear data |
| **Bayesian Ridge** | Probabilistic; provides uncertainty estimates |

### Classification Models (Sex Prediction)

| Model | Key Property |
|---|---|
| **Logistic Regression** | L2-regularised GLM baseline |
| **Gaussian Naive Bayes** | Fast probabilistic classifier assuming feature independence |

### Feature Selection

- **Stability Selection** — retains CpG sites appearing in >50% of 50 bootstrap subsamples
- **mRMR** (minimum Redundancy Maximum Relevance) — maximises relevance to age while minimising inter-feature redundancy; *k* tuned via 5-fold CV

### Hyperparameter Tuning

- **Randomised Search CV** — random sampling from predefined distributions
- **Optuna (TPE)** — Bayesian optimisation with Tree-structured Parzen Estimator

### Evaluation

All models evaluated via **bootstrap resampling** (1,000 resamples) on the held-out set.

**Regression metrics:** RMSE, MAE, R², Pearson *r*

**Classification metrics:** Accuracy, F1, MCC, ROC-AUC, PR-AUC

## References

- Moore, L. D., Le, T., & Fan, G. (2013). DNA methylation and its basic function. *Neuropsychopharmacology*, 38(1), 23–38.
- Bell, C. G., et al. (2019). DNA methylation aging clocks: challenges and recommendations. *Genome Biology*, 20, 249.
- Horvath, S., & Raj, K. (2018). DNA methylation-based biomarkers and the epigenetic clock theory of ageing. *Nature Reviews Genetics*, 19(6), 371–384.
- Du, P., et al. (2010). Comparison of Beta-value and M-value methods for quantifying methylation levels by microarray analysis. *BMC Bioinformatics*, 11, 587.

