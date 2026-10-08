#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analiza ablacyjna autoenkodera powtórzona dla wielu ziaren losowych.

Pojedynczy przebieg nie rozstrzyga, czy wartość dodana cech autoenkodera przewyższa
szum inicjalizacji sieci. Autoenkoder ma losowe wagi początkowe, a modele drzewiaste
losowo próbkują obserwacje i cechy, więc różnica między wariantem z cechami
autoenkodera i bez nich sama jest zmienną losową. Skrypt powtarza całą procedurę
dla kolejnych ziaren i raportuje rozkład tej różnicy: średnią, odchylenie
standardowe, przedział oraz udział przebiegów, w których różnica jest dodatnia.

Wynik trafia do results/tables/06_ablation_seeds.csv i rozstrzyga hipotezę trzecią
w sposób odporny na pojedynczy przebieg.

Użycie: PYTORCH_ENABLE_MPS_FALLBACK=1 python3 run_ablation_seeds.py [liczba_ziaren]
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

os.environ.setdefault('KERAS_BACKEND', 'torch')
os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')
os.environ.setdefault('MPLBACKEND', 'Agg')

import numpy as np
import pandas as pd

import models_lib as ml
import pipeline_lib as pl
from run_ablation import run_variant
from run_walkforward import load_modelling_frame

BASE = Path(__file__).parent
TABLES = BASE / 'results' / 'tables'
N_SEEDS = int(sys.argv[1]) if len(sys.argv) > 1 else 10


def main() -> None:
    f, cols = load_modelling_frame()
    y = f['target_liquidity_stress'].astype(int)
    main_oof = pd.read_csv(BASE / 'data' / 'processed' / 'oof_walkforward.csv',
                           index_col=0, parse_dates=True)
    common = main_oof.dropna(subset=['p_ens']).index
    print(f'Obserwacji: {len(f)} | zdarzeń: {int(y.sum())} | cech: {len(cols)}')
    print(f'Powtórzeń: {N_SEEDS}\n')

    from sklearn.metrics import average_precision_score
    rows = []
    seed0 = pl.SEED                     # baza zapamiętana przed pętlą: przypisanie
    for s in range(N_SEEDS):            # pl.SEED wewnątrz pętli zmieniałoby bazę
        t0 = time.time()
        seed = seed0 + s
        np.random.seed(seed)
        ml.SEED = seed
        pl.SEED = seed
        try:
            import torch
            torch.manual_seed(seed)
        except Exception:                                   # noqa: BLE001
            pass
        rec = {'Ziarno': seed}
        for use_ae, key in ((True, 'z_ae'), (False, 'bez_ae')):
            pred = run_variant(f, cols, use_ae)
            for kind, name in (('p_xgb', 'XGBoost'), ('p_lgb', 'LightGBM')):
                ser = pred.loc[pred.index.intersection(common), kind].dropna()
                yy = y.reindex(ser.index).values
                rec[f'{name}_{key}'] = average_precision_score(yy, ser.values)
        for name in ('XGBoost', 'LightGBM'):
            rec[f'{name}_roznica'] = rec[f'{name}_z_ae'] - rec[f'{name}_bez_ae']
        rec['srednia_roznica'] = np.mean([rec['XGBoost_roznica'], rec['LightGBM_roznica']])
        rows.append(rec)
        print(f'  ziarno {seed}: XGBoost {rec["XGBoost_roznica"]:+.4f}, '
              f'LightGBM {rec["LightGBM_roznica"]:+.4f}, '
              f'średnia {rec["srednia_roznica"]:+.4f}   ({time.time() - t0:.0f} s)', flush=True)
        pd.DataFrame(rows).to_csv(TABLES / '06_ablation_seeds.csv', index=False)

    df = pd.DataFrame(rows)
    print('\n' + '=' * 70)
    print('ROZKŁAD WARTOŚCI DODANEJ CECH AUTOENKODERA (AUC-PR)')
    print('=' * 70)
    pods = []
    for col, etyk in (('XGBoost_roznica', 'XGBoost'), ('LightGBM_roznica', 'LightGBM'),
                      ('srednia_roznica', 'średnia z obu modeli')):
        v = df[col].values
        pods.append({'Model': etyk, 'Średnia różnica': round(float(v.mean()), 4),
                     'Odchylenie std': round(float(v.std(ddof=1)), 4),
                     'Minimum': round(float(v.min()), 4), 'Maksimum': round(float(v.max()), 4),
                     'Udział przebiegów dodatnich (%)': round(100 * float((v > 0).mean()), 1),
                     'Liczba przebiegów': len(v)})
    tab = pd.DataFrame(pods)
    tab.to_csv(TABLES / '06_ablation_seeds_summary.csv', index=False)
    print(tab.to_string(index=False))


if __name__ == '__main__':
    main()
