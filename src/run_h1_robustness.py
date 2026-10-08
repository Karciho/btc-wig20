#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test odporności hipotezy H1: czy sygnał wyprzedzający pochodzi z rynku bitcoina.

Hipoteza pierwsza głosi, że sygnały z rynku bitcoina zawierają informację
wyprzedzającą o warunkach na GPW. Wskaźnik budowany w rozdziale trzecim korzysta
jednak także z cech opisujących sam indeks WIG20 — jego zmienność zrealizowaną,
wskaźnik niepłynności i opóźnione stopy zwrotu. Analiza wartości Shapleya pokazuje,
że to właśnie one wnoszą największy udział ważności. Wyprzedzanie zwrotów WIG20
przez taki wskaźnik może więc odzwierciedlać jego własną, autoregresyjną treść,
a nie transmisję z rynku kryptowalut.

Skrypt buduje wskaźnik w wariancie zawężonym — wyłącznie na cechach bitcoina
i otoczenia makroekonomicznego, bez żadnej zmiennej opisującej WIG20 — i sprawdza,
czy wyprzedza on zwroty indeksu. Test przeprowadzany jest na szeregu zróżnicowanym,
ponieważ szereg poziomu jest silnie autoskorelowany, a test przyczynowości Grangera
na szeregach niestacjonarnych daje wyniki pozorne.

Wynik zapisywany jest do results/tables/11_h1_robustness_*.csv.

Użycie: PYTORCH_ENABLE_MPS_FALLBACK=1 python3 run_h1_robustness.py
"""
from __future__ import annotations

import os
import time
import warnings
from pathlib import Path

os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')
os.environ.setdefault('KERAS_BACKEND', 'torch')

import numpy as np
import pandas as pd

import models_lib as ml
import pipeline_lib as pl
from run_walkforward import BURN_IN, load_modelling_frame

warnings.filterwarnings('ignore')

BASE = Path(__file__).parent
PROC = BASE / 'data' / 'processed'
TABLES = BASE / 'results' / 'tables'

# Warianty zbioru cech. „pełny" odpowiada rozdziałowi trzeciemu.
WARIANTY = {
    'pełny': None,
    'bez cech WIG20': ('wig_', 'cross_'),
    'tylko bitcoin i makro': ('wig_', 'cross_'),
}


def granger_both(hri: pd.Series, wig_ret: pd.Series, maxlag: int = 10) -> pd.DataFrame:
    """Test Grangera w obu kierunkach na szeregach zróżnicowanych."""
    from statsmodels.tsa.stattools import adfuller, grangercausalitytests, kpss

    d = pd.DataFrame({'wig_ret': wig_ret, 'hri': hri.diff()}).dropna()
    adf = adfuller(d['hri'], autolag='AIC')[1]
    kp = kpss(d['hri'], regression='c', nlags='auto')[1]
    rows = []
    for lag in range(1, maxlag + 1):
        p1 = grangercausalitytests(d[['wig_ret', 'hri']], maxlag=[lag],
                                   verbose=False)[lag][0]['ssr_ftest'][1]
        p2 = grangercausalitytests(d[['hri', 'wig_ret']], maxlag=[lag],
                                   verbose=False)[lag][0]['ssr_ftest'][1]
        rows.append({'Lag': lag, 'HRI → WIG20': round(p1, 5), 'WIG20 → HRI': round(p2, 5)})
    out = pd.DataFrame(rows)
    out.attrs['adf_p'] = adf
    out.attrs['kpss_p'] = kp
    return out


def bh(p: np.ndarray) -> np.ndarray:
    """Wartości p po poprawce Benjaminiego-Hochberga."""
    order = np.argsort(p)
    m = len(p)
    out = np.empty(m)
    prev = 1.0
    for rank, i in enumerate(reversed(order), 1):
        prev = min(prev, p[i] * m / (m - rank + 1))
        out[i] = prev
    return out


def main() -> None:
    f, all_cols = load_modelling_frame()
    master = pd.read_csv(PROC / 'master_dataset.csv', index_col=0, parse_dates=True)
    wig_ret = np.log(master['wig_close']).diff()

    podsumowanie = []
    for nazwa, drop in WARIANTY.items():
        cols = all_cols if drop is None else [c for c in all_cols if not c.startswith(drop)]
        if nazwa == 'tylko bitcoin i makro':
            cols = [c for c in cols if c.startswith(('btc_', 'macro_'))]
        print(f'\n{"=" * 70}\nWARIANT: {nazwa} — {len(cols)} cech')
        print(f'{"=" * 70}')
        if nazwa == 'pełny' and (PROC / 'oof_walkforward.csv').exists():
            oof = pd.read_csv(PROC / 'oof_walkforward.csv', index_col=0, parse_dates=True)
            print('  (wykorzystano istniejące prognozy z rozdziału trzeciego)')
        else:
            t0 = time.time()
            res = ml.run_walk_forward(f, cols, verbose=False)
            oof = res['oof']
            oof['y_true'] = res['y'].reindex(oof.index)
            oof.to_csv(PROC / f'oof_{nazwa.replace(" ", "_")}.csv')
            print(f'  czas: {time.time() - t0:.0f} s')

        hri = oof['p_ens'].dropna()
        g = granger_both(hri, wig_ret)
        p_fwd = g['HRI → WIG20'].values
        p_rev = g['WIG20 → HRI'].values
        bh_fwd, bh_rev = bh(p_fwd), bh(p_rev)
        g['HRI → WIG20 (BH)'] = np.round(bh_fwd, 5)
        g['WIG20 → HRI (BH)'] = np.round(bh_rev, 5)
        g.to_csv(TABLES / f'11_h1_granger_{nazwa.replace(" ", "_")}.csv', index=False)

        print(g.to_string(index=False))
        print(f'  ΔHRI stacjonarny: ADF p={g.attrs["adf_p"]:.3g}, KPSS p={g.attrs["kpss_p"]:.3g}')
        n_fwd = int((bh_fwd < pl.ALPHA).sum())
        n_rev = int((bh_rev < pl.ALPHA).sum())
        print(f'  istotnych po BH — HRI→WIG20: {n_fwd}/10, WIG20→HRI: {n_rev}/10')

        podsumowanie.append({
            'Wariant': nazwa, 'Liczba cech': len(cols),
            'Sesji': len(hri),
            'HRI → WIG20: istotnych opóźnień (BH)': n_fwd,
            'WIG20 → HRI: istotnych opóźnień (BH)': n_rev,
            'Najmniejsze p (HRI → WIG20)': round(float(p_fwd.min()), 5),
            'Najmniejsze p (WIG20 → HRI)': round(float(p_rev.min()), 5),
        })

    df = pd.DataFrame(podsumowanie)
    df.to_csv(TABLES / '11_h1_robustness_summary.csv', index=False)
    print(f'\n{"=" * 70}\nPODSUMOWANIE\n{"=" * 70}')
    print(df.to_string(index=False))


if __name__ == '__main__':
    main()
