#!/usr/bin/env python3
"""Recall vs. F1 as predictors of ATE error, plus selected-set analysis.

F1 against the true confounders mixes two kinds of error: missing a
confounder (biases the ATE) and adding an extra covariate (nearly free for a
linear AIPW). This script separates them, using the saved results where
possible and re-running the IHDP selectors only to log *which* covariates
were picked (the saved CSV records counts, not indices).

Analyses:
  1. Six synthetic worlds (Exp 1 F1/recall x Exp 2 ATE error, 5 selectors):
     pooled Pearson R^2 for F1, recall and precision, and within-world
     Spearman rho across selectors.
  2. IHDP-inspired benchmark (saved results, 10 methods): method-level and
     run-level R^2 for F1, recall and precision.
  3. TOST equivalence of QMI-CFS against the methods it ties on ATE error.
  4. IHDP selected sets: precision variables, instruments and noise kept,
     and F1 scored against the efficient adjustment set {X0..X4}.

The re-run reproduces the saved ATE errors exactly (checked below). Without
BorutaPy installed, BorutaFeatureSelector silently falls back to MI; since the
re-run without BorutaPy matches the saved IHDP "Boruta" rows exactly, those
saved rows were produced by that fallback.

Usage:
    python scripts/recall_analysis.py [--skip-rerun]
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
from scipy import stats
from tqdm import tqdm

from qmi_cfs.data_generation import generate_ihdp
from qmi_cfs.feature_selection import (
    QMIFeatureSelector, MIFeatureSelector, CMIFeatureSelector,
    LASSOFeatureSelector, BorutaFeatureSelector,
)
from qmi_cfs.causal_estimation import AIPWEstimator
from qmi_cfs.evaluation import compute_ate_error
from qmi_cfs.utils import ensure_dir

OUT = 'results/recall_analysis'
METRICS = ['f1', 'recall', 'precision']
# Paper's five synthetic-world selectors (RF excluded, as in Section 8.3)
SYNTH_METHODS = ['QMI-CFS', 'MI', 'CMI', 'LASSO', 'Boruta']
# IHDP Setting A roles (see generate_ihdp)
CONFOUNDERS = {0, 1, 2}
PRECISION = {3, 4}
INSTRUMENTS = {5, 6}
EFFICIENT = CONFOUNDERS | PRECISION


def r2_rows(df, condition):
    rows = []
    for m in METRICS:
        r, p = stats.pearsonr(df[m], df['ate_error'])
        rows.append({'condition': condition, 'metric': m, 'n': len(df),
                     'pearson_r': r, 'R2': r ** 2, 'p': p})
    return rows


def synthetic_worlds():
    e1 = pd.read_csv('results/experiment1/experiment1_results.csv')
    e2 = pd.read_csv('results/experiment2/experiment2_results.csv')
    m = (e1.groupby(['dataset', 'method'])[METRICS].mean()
         .join(e2.groupby(['dataset', 'method'])['ate_error'].mean(), how='inner')
         .reset_index())
    m = m[m['method'].isin(SYNTH_METHODS)]

    pooled = r2_rows(m, 'six worlds') + r2_rows(m[m['dataset'] != 'IC'], 'six worlds without IC')
    within = []
    for world, g in m.groupby('dataset'):
        row = {'dataset': world, 'n_methods': len(g)}
        for k in METRICS:
            row[f'spearman_{k}'] = stats.spearmanr(g[k], g['ate_error'])[0]
        within.append(row)
    return pd.DataFrame(pooled), pd.DataFrame(within)


def ihdp_saved():
    d = pd.read_csv('results/ihdp/ihdp_results.csv').dropna(subset=['ate_error'])
    means = d.groupby('method')[METRICS + ['ate_error']].mean().reset_index()
    rows = r2_rows(means, 'IHDP method means') + r2_rows(d, 'IHDP runs')
    return pd.DataFrame(rows), means, d


def tost(d, margins=(0.01, 0.02)):
    q = d[d['method'] == 'QMI-CFS'].set_index('seed')['ate_error']
    rows = []
    for method in ['HSIC-Lasso', 'LASSO', 'All-Features', 'CMI', 'RF']:
        o = d[d['method'] == method].set_index('seed')['ate_error']
        diff = (q - o).dropna()
        for margin in margins:
            p_lo = stats.ttest_1samp(diff, -margin, alternative='greater').pvalue
            p_hi = stats.ttest_1samp(diff, margin, alternative='less').pvalue
            rows.append({'method': method, 'n_pairs': len(diff), 'mean_diff': diff.mean(),
                         'sd_diff': diff.std(), 'margin': margin, 'tost_p': max(p_lo, p_hi)})
    return pd.DataFrame(rows)


def rerun_selected_sets(n_seeds=30, base_seed=42):
    """Same seeds and settings as scripts/run_ihdp_experiment.py."""
    methods = {
        'QMI-CFS': lambda rs: QMIFeatureSelector(n_bootstrap=100, random_state=rs),
        'MI': lambda rs: MIFeatureSelector(n_features=3, random_state=rs),
        'CMI': lambda rs: CMIFeatureSelector(n_features=3, random_state=rs),
        'LASSO': lambda rs: LASSOFeatureSelector(random_state=rs),
        'Boruta': lambda rs: BorutaFeatureSelector(random_state=rs),
    }
    rows = []
    for seed_idx in tqdm(range(n_seeds), desc='IHDP selected sets'):
        seed = base_seed + seed_idx * 100
        X, T, Y, _, true_ate = generate_ihdp(setting='A', seed=seed)
        for name, factory in methods.items():
            selector = factory(seed)
            selector.fit(X, T, Y)
            selected = [int(j) for j in selector.get_selected_features()]
            if len(selected) == 0:
                selected = list(range(min(X.shape[1], 5)))
            aipw = AIPWEstimator(random_state=seed)
            aipw.fit(X[:, selected], T, Y)
            ate_err = compute_ate_error(aipw.estimate_ate(X[:, selected], T, Y), true_ate)
            rows.append({'method': name, 'seed': seed_idx,
                         'selected': ' '.join(map(str, sorted(selected))),
                         'ate_error': ate_err})
    return pd.DataFrame(rows)


def f1_against(selected, truth):
    tp = len(selected & truth)
    if tp == 0:
        return 0.0
    prec, rec = tp / len(selected), tp / len(truth)
    return 2 * prec * rec / (prec + rec)


def selected_set_summary(sets, saved):
    check = sets.merge(saved[['method', 'seed', 'ate_error']], on=['method', 'seed'],
                       suffixes=('', '_saved'))
    max_dev = (check['ate_error'] - check['ate_error_saved']).abs().max()
    print(f"Re-run vs saved ATE error, max abs difference: {max_dev:.2e}")

    s = sets.copy()
    sel = s['selected'].apply(lambda x: set(map(int, x.split())))
    s['n_selected'] = sel.apply(len)
    s['n_precision'] = sel.apply(lambda x: len(x & PRECISION))
    s['n_instruments'] = sel.apply(lambda x: len(x & INSTRUMENTS))
    s['n_noise'] = sel.apply(lambda x: len(x - EFFICIENT - INSTRUMENTS))
    s['all_precision_kept'] = sel.apply(lambda x: PRECISION <= x)
    s['f1_confounders'] = sel.apply(lambda x: f1_against(x, CONFOUNDERS))
    s['f1_efficient_set'] = sel.apply(lambda x: f1_against(x, EFFICIENT))
    cols = ['f1_confounders', 'f1_efficient_set', 'n_selected', 'n_precision',
            'n_instruments', 'n_noise', 'all_precision_kept', 'ate_error']
    return s.groupby('method')[cols].mean().reset_index(), max_dev


def main():
    parser = argparse.ArgumentParser(description='Recall vs F1 analysis')
    parser.add_argument('--skip-rerun', action='store_true',
                        help='Reuse results/recall_analysis/ihdp_selected_sets.csv')
    args = parser.parse_args()
    ensure_dir(OUT)

    pooled, within = synthetic_worlds()
    ihdp_r2, ihdp_means, ihdp_runs = ihdp_saved()
    r2 = pd.concat([pooled, ihdp_r2], ignore_index=True)
    r2.to_csv(os.path.join(OUT, 'r2_f1_recall_precision.csv'), index=False)
    within.to_csv(os.path.join(OUT, 'within_world_spearman.csv'), index=False)
    ihdp_means.to_csv(os.path.join(OUT, 'ihdp_method_means.csv'), index=False)
    print("\n=== R^2 of ATE error on F1 / recall / precision ===")
    print(r2.round(3).to_string(index=False))
    print("\n=== Within-world Spearman rho (5 selectors per world) ===")
    print(within.round(2).to_string(index=False))

    t = tost(ihdp_runs)
    t.to_csv(os.path.join(OUT, 'ihdp_tost_ate_error.csv'), index=False)
    print("\n=== TOST equivalence vs QMI-CFS (IHDP ATE error) ===")
    print(t.round(4).to_string(index=False))

    sets_path = os.path.join(OUT, 'ihdp_selected_sets.csv')
    if args.skip_rerun and os.path.exists(sets_path):
        sets = pd.read_csv(sets_path)
    else:
        sets = rerun_selected_sets()
        sets.to_csv(sets_path, index=False)
    try:
        import boruta  # noqa: F401
    except ImportError:
        print("\nNote: BorutaPy is not installed, so 'Boruta' here is the MI fallback.")
    summary, _ = selected_set_summary(sets, ihdp_runs)
    summary.to_csv(os.path.join(OUT, 'ihdp_selected_set_summary.csv'), index=False)
    print("\n=== IHDP selected sets (efficient set = confounders + precision) ===")
    print(summary.round(3).to_string(index=False))


if __name__ == '__main__':
    main()
