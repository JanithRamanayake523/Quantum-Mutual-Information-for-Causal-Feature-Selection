# Results

Saved outputs that back every number, table and figure in the paper. All files
can be regenerated with the scripts in [`../scripts`](../scripts).

| Directory | Produced by | Contents |
|---|---|---|
| `experiment1/` | `run_experiment1.py` | Feature-selection accuracy (F1/precision/recall) per DGP and method; t-tests |
| `experiment2/` | `run_experiment2.py` | ATE error per DGP and method; Wilcoxon tests; ATE-error plot |
| `exp2_nonlinear/` | `run_experiment2.py` (nonlinear AIPW) | Nonlinear-nuisance variant of Experiment 2 |
| `experiment3/` | `run_experiment3.py` | Small-sample sweeps |
| `ihdp/` | `run_ihdp_experiment.py` | IHDP-inspired benchmark, Setting A |
| `ablation/` | `run_ablation_study.py` | QMI-CFS component ablation |
| `recall_analysis/` | `recall_analysis.py` | R² of ATE error on F1/recall/precision, TOST, IHDP selected sets |
| `same_draw/` | `same_draw_correlation.py` | Same-draw F1 vs ATE-error correlation |
| `tables_camera_ready/` | `generate_tables.py` | LaTeX tables used in the paper |

Notes:
- Seeds are fixed in the scripts (base seed 42); rerunning with the same
  package versions reproduces the CSVs.
- The IHDP "Boruta" rows come from the MI fallback (BorutaPy was not installed).
