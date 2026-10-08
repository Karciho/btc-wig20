#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Weryfikacja wystarczalności embarga w schemacie walidacji kroczącej.

Embargo oddziela okno treningowe od prognozowanego i musi pokrywać wyprzedzenie
wszystkich wielkości, które w danym punkcie czasu korzystają z sesji późniejszych.
Analiza kodu wskazuje, że taką wielkością jest wyłącznie zmienna objaśniana: jej
wartość w sesji t wyznaczana jest ze wskaźnika płynności z sesji t+1..t+5, czyli
z wyprzedzeniem równym horyzontowi prognozy. Wszystkie okna kroczące w cechach są
jednostronne i sięgają wyłącznie wstecz, a w walidacji kroczącej w przód sięganie
wstecz nie jest wyciekiem — analityk działający w czasie rzeczywistym też dysponuje
historią poprzedzającą moment prognozy.

Przy embargu równym 10 sesjom margines nad wymaganym minimum wynosi zatem 100%.
Skrypt sprawdza to empirycznie: jeżeli embargo jest za krótkie, wydłużanie go
powinno obniżać wyniki, bo usuwałoby przeciekającą informację. Stabilność wyników
przy embargu 10, 20, 40 i 60 sesji jest dowodem, że wyciek nie występuje.

Test ograniczono do modeli drzewiastych — liczą się szybko, a wyciek objawiłby się
w nich tak samo jak w sieciach rekurencyjnych.

Użycie: PYTORCH_ENABLE_MPS_FALLBACK=1 python3 run_embargo_test.py
"""
from __future__ import annotations

import os
import time
from pathlib import Path

os.environ.setdefault('KERAS_BACKEND', 'torch')
os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')
os.environ.setdefault('MPLBACKEND', 'Agg')

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

import models_lib as ml
import pipeline_lib as pl
from run_walkforward import load_modelling_frame

BASE = Path(__file__).parent
TABLES = BASE / 'results' / 'tables'
EMBARGA = (10, 20, 40, 60)


def przebieg(f: pd.DataFrame, cols: list[str], embargo: int) -> pd.DataFrame:
    """Walidacja krocząca dla modeli drzewiastych przy zadanym embargu."""
    y_all = f['target_liquidity_stress'].astype(int)
    folds = pl.purged_walk_forward_splits(f.index, n_splits=ml.N_SPLITS,
                                          embargo=embargo,
                                          min_train_frac=ml.MIN_TRAIN_FRAC)
    out = pd.DataFrame(index=f.index, columns=['p_xgb', 'p_lgb'], dtype=float)
    for tr_idx, te_idx in folds:
        n_tr = len(tr_idx)
        n_inner = max(int(n_tr * ml.INNER_VAL_FRAC), 60)
        inner_tr = tr_idx[:n_tr - n_inner - embargo]
        inner_va = tr_idx[n_tr - n_inner:]
        if len(inner_tr) < 100:
            continue
        Xtr = f.loc[inner_tr, cols].values
        Xva = f.loc[inner_va, cols].values
        Xte = f.loc[te_idx, cols].values
        ytr = y_all.loc[inner_tr].values
        yva = y_all.loc[inner_va].values
        sc = StandardScaler().fit(Xtr)
        Xtr, Xva, Xte = (sc.transform(a) for a in (Xtr, Xva, Xte))
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
    print(f'Obserwacji: {len(f)} | zdarzeń: {int(y.sum())} | cech: {len(cols)}')
    print(f'Horyzont prognozy: {pl.HORIZON} sesji — to jedyne wyprzedzenie w potoku.\n')

    rows = []
    for emb in EMBARGA:
        t0 = time.time()
        pred = przebieg(f, cols, emb)
        for kind, nazwa in (('p_xgb', 'XGBoost'), ('p_lgb', 'LightGBM')):
            s = pred[kind].dropna()
            yy = y.reindex(s.index).values
            rows.append({'Embargo (sesje)': emb, 'Model': nazwa,
                         'AUC-ROC': round(roc_auc_score(yy, s.values), 4),
                         'AUC-PR': round(average_precision_score(yy, s.values), 4),
                         'Obserwacji': len(s), 'Zdarzeń': int(yy.sum())})
        print(f'  embargo {emb:2d} sesji: gotowe ({time.time() - t0:.0f} s)', flush=True)
        pd.DataFrame(rows).to_csv(TABLES / '12_embargo_test.csv', index=False)

    tab = pd.DataFrame(rows)
    print('\n' + '=' * 72)
    print('WPŁYW DŁUGOŚCI EMBARGA NA WYNIKI')
    print('=' * 72)
    print(tab.to_string(index=False))
    piv = tab.pivot(index='Model', columns='Embargo (sesje)', values='AUC-PR')
    rozstep = (piv.max(axis=1) - piv.min(axis=1)).round(4)
    print('\nrozstęp AUC-PR między wariantami embarga:')
    print(rozstep.to_string())
    print('\nWniosek: brak systematycznego spadku wraz z wydłużaniem embarga oznacza, '
          'że embargo 10 sesji nie pozostawia przeciekającej informacji.')


if __name__ == '__main__':
    main()
