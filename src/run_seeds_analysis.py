#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analiza wyników walidacji kroczącej powtórzonej dla wielu ziaren losowych.

Skrypt czyta prognozy pozapróbkowe zapisane osobno dla każdego ziarna przez
run_seeds.py i wyprowadza z nich trzy rzeczy:

1. Rozkład jakości każdego modelu — średnią, medianę, odchylenie, zakres i przedział
   z percentyli 5–95. Pojedynczy przebieg jest jedną realizacją zmiennej losowej,
   a nie pomiarem; przy różnicach rzędu 0,03–0,05 pola pod krzywą trzeba wiedzieć,
   czy przekraczają one rozrzut wynikający z samego losowania.

2. Sparowane porównanie zespołu z najlepszym modelem bazowym — w ilu przebiegach
   zespół wypadł lepiej i czy różnica median jest istotna. Porównanie sparowane po
   ziarnach jest tu właściwym testem, bo eliminuje wspólną dla obu modeli zmienność
   między przebiegami.

3. Porównanie wariantów warstwy agregującej. Warianty liczone są z zapisanych
   prognoz bazowych, więc nie wymagają ponownego trenowania.

Uwaga o skali wejścia meta-modelu: prognozy bazowe rozciągają się na kilkadziesiąt
rzędów wielkości, więc liniowa kombinacja surowych prawdopodobieństw jest zdominowana
przez model o największych wartościach bezwzględnych niezależnie od jego jakości
rankingowej. Warianty logistyczne liczone są zatem na skali logitowej.

Użycie: python3 run_seeds_analysis.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize, nnls
from scipy.special import expit, logit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

BASE = Path(__file__).parent
SEEDS = BASE / 'data' / 'processed' / 'seeds'
TABLES = BASE / 'results' / 'tables'
COLS = ['p_lstm', 'p_gru', 'p_xgb', 'p_lgb']
ETYKIETY = {'p_lstm': 'LSTM', 'p_gru': 'GRU', 'p_xgb': 'XGBoost', 'p_lgb': 'LightGBM'}
EPS = 1e-6
META_MIN_EVENTS = 15


def _logit(X):
    return logit(np.clip(X, EPS, 1 - EPS))


def _nnls(Xtr, ytr, Xte):
    w, _ = nnls(Xtr, ytr.astype(float))
    if w.sum() <= 0:
        w = np.ones(Xtr.shape[1])
    return Xte @ (w / w.sum())


def _logit_nonneg(Xtr, ytr, Xte):
    Xtr, Xte = _logit(Xtr), _logit(Xte)
    k = Xtr.shape[1]
    w = np.where(ytr == 1, (len(ytr) - ytr.sum()) / max(ytr.sum(), 1), 1.0)

    def strata(th):
        p = np.clip(expit(Xtr @ th[1:] + th[0]), 1e-9, 1 - 1e-9)
        return -np.average(ytr * np.log(p) + (1 - ytr) * np.log(1 - p), weights=w)

    r = minimize(strata, np.concatenate([[0.0], np.full(k, 1.0 / k)]),
                 method='L-BFGS-B', bounds=[(None, None)] + [(0.0, None)] * k)
    return expit(Xte @ r.x[1:] + r.x[0])


def _logit_free(Xtr, ytr, Xte, skala_logitowa=True):
    if skala_logitowa:
        Xtr, Xte = _logit(Xtr), _logit(Xte)
    sc = StandardScaler().fit(Xtr)
    m = LogisticRegression(C=1.0, class_weight='balanced', max_iter=2000)
    m.fit(sc.transform(Xtr), ytr)
    return m.predict_proba(sc.transform(Xte))[:, 1]


WARIANTY = {
    'Zespół: NNLS na prawdopodobieństwach': _nnls,
    'Zespół: logit z wagami nieujemnymi': _logit_nonneg,
    'Zespół: logit bez ograniczeń': lambda a, b, c: _logit_free(a, b, c, True),
    'Zespół: logit bez ograniczeń na skali surowej': lambda a, b, c: _logit_free(a, b, c, False),
}


def warianty_agregacji(oof: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    """Liczy warianty warstwy agregującej z zapisanych prognoz bazowych."""
    out = pd.DataFrame(index=oof.index, columns=list(WARIANTY) + ['Zespół: uśrednianie'],
                       dtype=float)
    out['Zespół: uśrednianie'] = oof[COLS].mean(axis=1)
    for f_i in sorted(oof['fold'].dropna().unique()):
        prior = oof[oof['fold'] < f_i].dropna(subset=COLS)
        cur = oof[oof['fold'] == f_i].dropna(subset=COLS)
        if not len(cur):
            continue
        yp = y.reindex(prior.index)
        if len(prior) < 100 or int(yp.sum()) < META_MIN_EVENTS:
            continue
        for nazwa, fn in WARIANTY.items():
            out.loc[cur.index, nazwa] = fn(prior[COLS].values, yp.astype(int).values,
                                           cur[COLS].values)
    return out


def main() -> None:
    pliki = sorted(SEEDS.glob('oof_seed_*.csv'))
    if not pliki:
        raise SystemExit('Brak plików z ziarnami — uruchom najpierw run_seeds.py')
    print(f'Ziaren do analizy: {len(pliki)}\n')

    wiersze = []
    for f in pliki:
        seed = int(f.stem.split('_')[-1])
        oof = pd.read_csv(f, index_col=0, parse_dates=True)
        y = oof['y_true'].astype(int)
        agg = warianty_agregacji(oof, y)
        wspolny = agg['Zespół: NNLS na prawdopodobieństwach'].dropna().index
        yy = y.reindex(wspolny)
        rec = {'Ziarno': seed, 'Obserwacji': len(wspolny), 'Zdarzeń': int(yy.sum())}
        for c in COLS:
            s = oof.loc[wspolny, c]
            rec[f'{ETYKIETY[c]}|AUC-ROC'] = roc_auc_score(yy, s)
            rec[f'{ETYKIETY[c]}|AUC-PR'] = average_precision_score(yy, s)
        for nazwa in list(WARIANTY) + ['Zespół: uśrednianie']:
            s = agg.loc[wspolny, nazwa]
            if s.notna().all():
                rec[f'{nazwa}|AUC-ROC'] = roc_auc_score(yy, s)
                rec[f'{nazwa}|AUC-PR'] = average_precision_score(yy, s)
        wiersze.append(rec)
        print(f'  ziarno {seed}: {len(wspolny)} sesji, {int(yy.sum())} zdarzeń', flush=True)

    df = pd.DataFrame(wiersze).sort_values('Ziarno')
    df.to_csv(TABLES / '13_seeds_raw.csv', index=False)

    # ── rozkłady ────────────────────────────────────────────────
    modele = [ETYKIETY[c] for c in COLS] + list(WARIANTY) + ['Zespół: uśrednianie']
    rozklady = []
    for m in modele:
        for miara in ('AUC-ROC', 'AUC-PR'):
            k = f'{m}|{miara}'
            if k not in df.columns:
                continue
            v = df[k].dropna().values
            if not len(v):
                continue
            rozklady.append({'Model': m, 'Miara': miara, 'Ziaren': len(v),
                             'Średnia': round(float(v.mean()), 4),
                             'Mediana': round(float(np.median(v)), 4),
                             'Odch. std': round(float(v.std(ddof=1)), 4),
                             'Minimum': round(float(v.min()), 4),
                             'Maksimum': round(float(v.max()), 4),
                             'P5': round(float(np.percentile(v, 5)), 4),
                             'P95': round(float(np.percentile(v, 95)), 4)})
    roz = pd.DataFrame(rozklady)
    roz.to_csv(TABLES / '13_seeds_distribution.csv', index=False)
    print('\n' + '=' * 96)
    print('ROZKŁADY JAKOŚCI WEDŁUG ZIAREN')
    print('=' * 96)
    print(roz.to_string(index=False))

    # ── sparowane porównanie zespołu z modelami bazowymi ────────
    ZESPOL = 'Zespół: NNLS na prawdopodobieństwach'
    pary = []
    for m in [ETYKIETY[c] for c in COLS] + ['Zespół: uśrednianie']:
        for miara in ('AUC-ROC', 'AUC-PR'):
            a, b = f'{ZESPOL}|{miara}', f'{m}|{miara}'
            if a not in df.columns or b not in df.columns:
                continue
            d = (df[a] - df[b]).dropna().values
            if len(d) < 3:
                continue
            t, p_t = stats.ttest_rel(df[a].dropna(), df[b].dropna())
            try:
                _, p_w = stats.wilcoxon(d)
            except ValueError:
                p_w = float('nan')
            pary.append({'Porównanie': f'zespół − {m}', 'Miara': miara,
                         'Średnia różnica': round(float(d.mean()), 4),
                         'Zespół lepszy w': f'{int((d > 0).sum())}/{len(d)}',
                         'p (test par)': round(float(p_t), 4),
                         'p (Wilcoxon)': round(float(p_w), 4),
                         'Istotna': 'tak' if p_t < 0.05 else 'nie'})
    par = pd.DataFrame(pary)
    par.to_csv(TABLES / '13_seeds_paired.csv', index=False)
    print('\n' + '=' * 96)
    print('SPAROWANE PORÓWNANIE ZESPOŁU Z MODELAMI BAZOWYMI (po ziarnach)')
    print('=' * 96)
    print(par.to_string(index=False))

    # ── wrażliwość rodzin modeli na ziarno ──────────────────────
    print('\n' + '=' * 96)
    print('WRAŻLIWOŚĆ NA ZIARNO: odchylenie standardowe AUC-ROC między przebiegami')
    print('=' * 96)
    for m in [ETYKIETY[c] for c in COLS]:
        k = f'{m}|AUC-ROC'
        if k in df.columns:
            v = df[k].dropna()
            print(f'  {m:10s} {v.std(ddof=1):.4f}   (zakres {v.min():.3f}–{v.max():.3f})')


if __name__ == '__main__':
    main()
