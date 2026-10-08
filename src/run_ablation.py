#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analiza ablacyjna: czy cechy generowane przez autoenkoder wnoszą informację
komplementarną wobec cech nadzorowanych.

Test polega na powtórzeniu walidacji kroczącej dla modeli drzewiastych w dwóch
wariantach — z cechami autoenkodera i bez nich — przy identycznych podziałach,
ziarnie losowym i hiperparametrach. Różnica w polu pod krzywą precyzja–czułość
jest miarą wartości dodanej komponentu nienadzorowanego.

Modele drzewiaste wybrano jako podstawę testu, ponieważ osiągają najwyższą jakość
w zestawieniu głównym i są szybkie w treningu, co pozwala powtórzyć procedurę
bez wielogodzinnych obliczeń.
"""
from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault('KERAS_BACKEND', 'torch')
os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')
os.environ.setdefault('MPLBACKEND', 'Agg')

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

import models_lib as ml
import pipeline_lib as pl
from run_walkforward import load_modelling_frame

BASE = Path(__file__).parent
TABLES = BASE / 'results' / 'tables'


def run_variant(f: pd.DataFrame, cols: list[str], use_ae: bool) -> pd.DataFrame:
    y_all = f['target_liquidity_stress'].astype(int)
    idx = f.index
    folds = pl.purged_walk_forward_splits(idx, n_splits=ml.N_SPLITS,
                                          min_train_frac=ml.MIN_TRAIN_FRAC)
    out = pd.DataFrame(index=idx, columns=['p_xgb', 'p_lgb'], dtype=float)

    for tr_idx, te_idx in folds:
        n_tr = len(tr_idx)
        n_inner = max(int(n_tr * ml.INNER_VAL_FRAC), 60)
        inner_tr = tr_idx[:n_tr - n_inner - pl.EMBARGO]
        inner_va = tr_idx[n_tr - n_inner:]

        Xtr_raw = f.loc[inner_tr, cols].values
        Xva_raw = f.loc[inner_va, cols].values
        Xte_raw = f.loc[te_idx, cols].values
        ytr = y_all.loc[inner_tr].values
        yva = y_all.loc[inner_va].values

        scaler = StandardScaler().fit(Xtr_raw)
        Xtr, Xva, Xte = (scaler.transform(a) for a in (Xtr_raw, Xva_raw, Xte_raw))

        if use_ae:
            ae = ml.fit_autoencoder(Xtr[ytr == 0], Xtr.shape[1])
            re_tr_normal = ml.reconstruction_error(ae, Xtr[ytr == 0])
            thr = float(np.quantile(re_tr_normal, 0.95))

            def aug(X):
                re = ml.reconstruction_error(ae, X)
                return np.column_stack([X, re, (re > thr).astype(float)])
            Xtr, Xva, Xte = aug(Xtr), aug(Xva), aug(Xte)

        for kind in ('xgb', 'lgb'):
            _, p_va, p_te = ml.fit_predict_trees(kind, Xtr, ytr, Xva, Xte)
            if len(np.unique(yva)) > 1 and len(p_va) > 30:
                cal = pl.fit_calibrator(p_va, yva, method='platt')
                p_te = pl.apply_calibrator(cal, p_te)
            out.loc[te_idx, f'p_{kind}'] = p_te
    return out


def main() -> None:
    f, cols = load_modelling_frame()
    y = f['target_liquidity_stress'].astype(int)
    print(f'Obserwacji: {len(f)} | zdarzeń: {int(y.sum())} | cech: {len(cols)}\n')

    results = {}
    for use_ae, label in ((True, 'z cechami autoenkodera'), (False, 'bez cech autoenkodera')):
        print(f'Wariant: {label}...')
        results[label] = run_variant(f, cols, use_ae)

    # ocena na wspólnym podzbiorze z zestawieniem głównym
    main_oof = pd.read_csv(BASE / 'data' / 'processed' / 'oof_walkforward.csv',
                           index_col=0, parse_dates=True)
    common = main_oof.dropna(subset=['p_ens']).index

    rows = []
    for label, pred in results.items():
        for kind, name in (('p_xgb', 'XGBoost'), ('p_lgb', 'LightGBM')):
            s = pred.loc[pred.index.intersection(common), kind].dropna()
            yy = y.reindex(s.index).values
            half = len(s) // 2
            thr = pl.select_threshold(yy[:half], s.values[:half]) if yy[:half].sum() else 0.5
            m = pl.evaluate(yy, s.values, thr)
            rows.append({'Wariant': label, 'Model': name,
                         'AUC-PR': m['auc_pr'], 'AUC-ROC': m['auc_roc'],
                         'F1': m['f1'], 'Wynik Briera': m['brier'],
                         'Iloraz do częstości bazowej': m['auc_pr_lift']})
    tab = pd.DataFrame(rows)
    tab.to_csv(TABLES / '06_ablation_autoencoder.csv', index=False)
    print('\n' + tab.to_string(index=False))

    piv = tab.pivot(index='Model', columns='Wariant', values='AUC-PR')
    piv['Różnica'] = (piv['z cechami autoenkodera'] - piv['bez cech autoenkodera']).round(4)
    print('\nWartość dodana cech autoenkodera (AUC-PR):')
    print(piv.to_string())
    piv.to_csv(TABLES / '06_ablation_summary.csv')


if __name__ == '__main__':
    main()
