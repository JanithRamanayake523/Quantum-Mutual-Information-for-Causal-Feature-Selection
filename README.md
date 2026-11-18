# QMI-CFS: Quantum Mutual Information for Causal Feature Selection

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Manuscript](https://img.shields.io/badge/manuscript-forthcoming-orange.svg)](#-manuscript)

> **An empirical study of feature recovery and treatment effect estimation using quantum-inspired dependency measures.**

---

## ✨ Highlights

- **Quantum-inspired, classical implementation** — no quantum hardware needed.
- **Parameter-free QMI estimation** — no bandwidth or neighbor-count tuning unlike k-NN MI.
- **Competitive confounder recovery** — compared against MI, CMI, LASSO, Boruta, and All-Features.
- **Striking divergence finding** — confounder-recovery F1 and downstream ATE error are essentially uncorrelated across benchmarks.
- **Reproducible** — all tables, figures, and statistics can be regenerated from the saved CSVs in one command.

---

## 📄 Manuscript

This repository accompanies the manuscript:

> **Quantum Mutual Information for Causal Feature Selection: An Empirical Study of Feature Recovery and Treatment Effect Estimation**

The manuscript has been accepted for publication in a Springer CCIS proceedings volume. In line with the publisher's policy, the paper itself (PDF and LaTeX source) is **not** distributed in this repository until it is officially published. A citation entry and link will be added here after publication.

The code, experiment results (`results/`), and result tables used in the paper are available here so every reported number can be reproduced.

---

## 🚀 What is QMI-CFS?

**QMI-CFS** encodes classical covariates into quantum states via angle embedding, constructs small single-, pair-, and triple-qubit density matrices, and computes **Quantum Mutual Information (QMI)** from their von Neumann entropies. Features are scored by a composite relevance function that balances:

- propensity relevance: `QMI(X; T)`
- prognostic relevance: `QMI(X; Y)`
- conditional relevance: `QMI(X; T | Y)` (helps distinguish confounders from instruments)

Selected features are fed into a doubly-robust AIPW estimator for the Average Treatment Effect (ATE).


---

## 📁 Repository Structure

```text
QMI-CFS/
├── qmi_cfs/              # Core Python package
├── scripts/              # Experiment runners & table/figure generators
├── tests/                # Unit and integration tests
├── results/              # Saved CSV/JSON outputs, plots, tables (see results/README.md)
├── .github/workflows/    # CI (pytest)
├── CITATION.cff
├── README.md
├── LICENSE
├── requirements.txt
└── setup.py
```

---

## 🛠️ Installation

```bash
pip install -e .
```

For the optional Boruta baseline:

```bash
pip install -e ".[boruta]"
```

### Requirements

- Python ≥ 3.9
- NumPy, SciPy, scikit-learn, pandas, matplotlib, statsmodels
- tqdm
- BorutaPy (optional)

> **Note:** without BorutaPy, `BorutaFeatureSelector` silently falls back to an MI-based selector. The saved IHDP "Boruta" rows were produced by this fallback; install the `boruta` extra to use the real algorithm.

---

## 🔬 Reproducing the Paper

### 1. Run the experiments

You can either use the precomputed CSVs in [`results/`](results/) or regenerate them:

```bash
# Feature-selection accuracy
python scripts/run_experiment1.py --seeds 50 --output results/experiment1

# Treatment-effect estimation
python scripts/run_experiment2.py --seeds 30 --output results/experiment2

# Small-sample analysis
python scripts/run_experiment3.py --seeds 50 --output results/experiment3

# IHDP benchmark
python scripts/run_ihdp_experiment.py --seeds 50 --output results/ihdp

# Ablation study
python scripts/run_ablation_study.py --output results/ablation
```

### 2. Generate tables and figures

```bash
# LaTeX tables
python scripts/generate_tables.py --output results/tables

# Main figures
python scripts/generate_figures.py --output results/figures
python scripts/generate_ic_figure.py --output results/figures
```

Or regenerate all figures directly from the CSVs:

```bash
python scripts/regenerate_paper_figures.py
```

### 3. Supplementary analyses

```bash
# Recall vs. F1 as predictors of ATE error, TOST equivalence, IHDP selected sets
python scripts/recall_analysis.py            # add --skip-rerun to reuse saved selected sets

# Same-draw F1 / ATE-error correlation
python scripts/same_draw_correlation.py
```

The camera-ready tables used in the paper are stored in [`results/tables_camera_ready/`](results/tables_camera_ready/).

> The manuscript sources are intentionally excluded from this repository (publisher embargo); everything unpublished lives in the git-ignored `_local/` folder.

---

## 📊 Key Results

### Divergence between feature recovery and causal estimation

Across the six synthetic DGPs, F1 score explains only about **4%** of the variance in ATE error, and the correlation is not statistically significant.

<p align="center">
  <img src="results/experiment2/experiment2_ate_error_plot.png" alt="ATE error by method" width="90%">
</p>

### IHDP benchmark

On IHDP Setting A, QMI-CFS achieves the **lowest ATE error** despite only moderate F1, demonstrating that a feature set with lower recovery accuracy can still yield better causal estimates when it is well-calibrated for the downstream nuisance models.

### Interacting confounders

When confounders enter only through multiplicative interactions, all dependency-based selectors — including QMI-CFS — struggle, even with pairwise augmentation and nonlinear AIPW.

---

## ✅ Tests

```bash
pytest tests/ -q
```

The suite checks density-matrix validity (trace 1, symmetric, PSD), entropy and QMI properties (non-negativity, dependent > independent), AIPW recovery of a known ATE, selector determinism under a fixed seed, and an end-to-end smoke test (data generation → QMI selection → AIPW).

---

## 🤝 Contributing

This repository accompanies a research manuscript and its saved results are frozen to match the paper. Issues and pull requests (bug fixes, additional baselines) are welcome; changes that alter numerical behaviour should be clearly flagged.

---

## 📚 Citation

If you use this code or the QMI-CFS method in your research, please cite it via [`CITATION.cff`](CITATION.cff) (GitHub's "Cite this repository" button). The paper reference and DOI will be added here after publication.

---

## 📝 License

This project is licensed under the [MIT License](LICENSE).

---
