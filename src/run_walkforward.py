#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Uruchomienie pełnego przebiegu purged walk-forward dla modeli bazowych i zespołu.

Wywoływane z notebooka 05 (albo bezpośrednio) — obliczenia trwają kilkanaście minut,
więc wyniki zapisywane są na dysk i wczytywane przez kolejne notebooki.

Produkty:
    data/processed/oof_walkforward.csv   — prognozy pozapróbkowe wszystkich modeli
    results/tables/05_walkforward_folds.csv
    results/tables/08_model_comparison_oos.csv
"""
from __future__ import annotations

import os
import time
from pathlib import Path

os.environ.setdefault('KERAS_BACKEND', 'torch')
os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')
os.environ.setdefault('MPLBACKEND', 'Agg')

import pandas as pd

import models_lib as ml
import pipeline_lib as pl

BASE = Path(__file__).parent
PROC = BASE / 'data' / 'processed'
TABLES = BASE / 'results' / 'tables'

MODEL_FEATURE_PREFIXES_DROP = pl.MODEL_FEATURE_PREFIXES_DROP
BURN_IN = 200          # sesje odrzucone na początku (niepełne okna kroczące)


def load_modelling_frame() -> tuple[pd.DataFrame, list[str]]:
    f = pd.read_csv(PROC / 'features_dataset.csv', index_col=0, parse_dates=True)
    drop = [c for c in f.columns
            if c.startswith(MODEL_FEATURE_PREFIXES_DROP) or c == 'target_liquidity_stress']
    feature_cols = [c for c in f.columns if c not in drop]
    f = f.iloc[BURN_IN:].copy()
    f[feature_cols] = f[feature_cols].ffill().bfill()
    f = f.dropna(subset=['target_liquidity_stress'])
    return f, feature_cols


def main() -> None:
    f, feature_cols = load_modelling_frame()
    print(f'Cech do modelowania: {len(feature_cols)}')
    print(f'Obserwacji: {len(f)} | zdarzeń: {int(f["target_liquidity_stress"].sum())} '
          f'({100 * f["target_liquidity_stress"].mean():.2f}%)')
    print(f'Zakres: {f.index.min().date()} → {f.index.max().date()}\n')

    t0 = time.time()
    res = ml.run_walk_forward(f, feature_cols, verbose=True)
    print(f'\nCzas obliczeń: {time.time() - t0:.0f} s')

    oof = res['oof']
    oof['y_true'] = res['y'].reindex(oof.index)
    oof.to_csv(PROC / 'oof_walkforward.csv')
    res['folds'].to_csv(TABLES / '05_walkforward_folds.csv', index=False)

    metrics = ml.pooled_metrics(res)
    metrics.to_csv(TABLES / '08_model_comparison_oos.csv')
    print('\n=== Metryki pozapróbkowe (połączone foldy, wspólny podzbiór obserwacji):')
    print(metrics.to_string())

    cov = oof.dropna(subset=['p_ens'])
    print(f'\nPokrycie indeksu HRI: {cov.index.min().date()} → {cov.index.max().date()} '
          f'({len(cov)} sesji, {int(res["y"].reindex(cov.index).sum())} zdarzeń)')
    print('Zapisano: data/processed/oof_walkforward.csv')


if __name__ == '__main__':
    main()
