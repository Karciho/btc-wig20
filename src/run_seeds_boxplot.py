#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Wykres pudełkowy rozkładu jakości modeli w powtórzeniach dla wielu ziaren losowych.

Tabela z pojedynczymi wartościami sugeruje precyzję, której pomiar nie ma: przy
osiemnastu zdarzeniach w okresie pozapróbkowym pojedyncza obserwacja waży ponad pięć
procent wyniku, a zmiana ziarna losowego przesuwa pole pod krzywą o wielkość
porównywalną z różnicami między architekturami. Rysunek pokazuje rozkłady zamiast
punktów, dzięki czemu widać, które różnice wykraczają poza rozrzut losowania,
a które mieszczą się w nim w całości.

Użycie: python3 run_seeds_boxplot.py
Wynik:  results/figures/fig_30_seeds_boxplot.png (oraz .pdf)
"""
from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault('MPLBACKEND', 'Agg')

import matplotlib.pyplot as plt
import pandas as pd

BASE = Path(__file__).parent
TABLES = BASE / 'results' / 'tables'
FIGS = BASE / 'results' / 'figures'

MODELE = [('LSTM', 'LSTM'),
          ('GRU', 'GRU'),
          ('XGBoost', 'XGBoost'),
          ('LightGBM', 'LightGBM'),
          ('Zespół: NNLS na prawdopodobieństwach', 'Zespół\n(NNLS)'),
          ('Zespół: uśrednianie', 'Zespół\n(uśrednianie)')]
# drzewa wyróżnione, bo to one wyznaczają poziom odniesienia dla hipotezy o zespole
BARWY = ['#9ecae1', '#9ecae1', '#fdae6b', '#fdae6b', '#a1d99b', '#a1d99b']


def styl() -> None:
    import locale
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({'font.size': 10, 'figure.dpi': 150,
                         'axes.titlesize': 12, 'axes.labelsize': 10})
    for loc in ('pl_PL.UTF-8', 'pl_PL.utf8', 'pl_PL', 'Polish_Poland.1250'):
        try:
            locale.setlocale(locale.LC_NUMERIC, loc)
            plt.rcParams['axes.formatter.use_locale'] = True
            return
        except locale.Error:
            continue
    plt.rcParams['axes.formatter.use_locale'] = False


def main() -> None:
    styl()
    df = pd.read_csv(TABLES / '13_seeds_raw.csv')
    n = len(df)

    fig, ax = plt.subplots(1, 2, figsize=(12, 5.2))
    for a, miara in zip(ax, ('AUC-ROC', 'AUC-PR')):
        dane, etykiety = [], []
        for kol, et in MODELE:
            k = f'{kol}|{miara}'
            if k in df.columns and df[k].notna().any():
                dane.append(df[k].dropna().values)
                etykiety.append(et)
        bp = a.boxplot(dane, tick_labels=etykiety, patch_artist=True, widths=0.6,
                       medianprops={'color': '#252525', 'linewidth': 1.6},
                       flierprops={'marker': 'o', 'markersize': 3.5,
                                   'markerfacecolor': '#636363',
                                   'markeredgecolor': 'none'})
        for pudlo, barwa in zip(bp['boxes'], BARWY):
            pudlo.set_facecolor(barwa)
            pudlo.set_edgecolor('#252525')
            pudlo.set_linewidth(0.9)
        # punkty pojedynczych przebiegów: pokazują, że pudełko opiera się na 15 obserwacjach
        for i, v in enumerate(dane, start=1):
            a.scatter([i] * len(v), v, s=9, color='#252525', alpha=0.45, zorder=3,
                      linewidths=0)
        a.set_ylabel(miara)
        a.set_title(f'{miara} w {n} powtórzeniach')
        a.tick_params(axis='x', labelsize=8.5)

    fig.tight_layout()
    for ext in ('png', 'pdf'):
        plt.savefig(FIGS / f'fig_30_seeds_boxplot.{ext}', bbox_inches='tight')
    plt.close(fig)
    print(f'Zapisano fig_30_seeds_boxplot (png, pdf) — {n} przebiegów.')


if __name__ == '__main__':
    main()
