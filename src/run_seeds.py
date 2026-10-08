#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pełna walidacja krocząca powtórzona dla wielu ziaren losowych.

Pojedynczy przebieg nie jest pomiarem jakości modelu, lecz jedną realizacją zmiennej
losowej. Sieci rekurencyjne startują z losowych wag i losowo mieszają mini-partie,
modele drzewiaste losowo próbkują obserwacje i cechy, a autoenkoder ma losową
inicjalizację. Przy osiemnastu zdarzeniach w okresie pozapróbkowym pojedyncza
obserwacja waży ponad pięć procent wyniku, więc różnice rzędu 0,03–0,05 pola pod
krzywą mogą być w całości artefaktem losowania.

Skrypt zapisuje pełne prognozy pozapróbkowe dla każdego ziarna osobno, dzięki czemu
warianty warstwy agregującej można policzyć później bez ponownego trenowania.

Każde ziarno zapisywane jest natychmiast po policzeniu, więc przerwanie procesu po
dziesiątym przebiegu nie unieważnia dziesięciu policzonych. Ziarno zajmowane jest
atomowo plikiem-znacznikiem tworzonym z flagą wyłączności, dzięki czemu dowolna
liczba procesów może pracować równolegle na tym samym zakresie bez ryzyka, że dwa
policzą to samo ziarno. Znacznik po awarii procesu zostaje osierocony i jest zwalniany
po przekroczeniu czasu przeterminowania.

Użycie: PYTORCH_ENABLE_MPS_FALLBACK=1 python3 run_seeds.py [liczba_ziaren] [przesunięcie]
Wynik:  data/processed/seeds/oof_seed_NN.csv
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

os.environ.setdefault('KERAS_BACKEND', 'torch')
os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')
os.environ.setdefault('MPLBACKEND', 'Agg')

import pandas as pd
from sklearn.metrics import roc_auc_score

import models_lib as ml
import pipeline_lib as pl
from run_walkforward import load_modelling_frame

BASE = Path(__file__).parent
SEEDS_DIR = BASE / 'data' / 'processed' / 'seeds'
N_SEEDS = int(sys.argv[1]) if len(sys.argv) > 1 else 15
OFFSET = int(sys.argv[2]) if len(sys.argv) > 2 else 0   # przesunięcie dla pracy równoległej
KOLUMNY = ['p_lstm', 'p_gru', 'p_xgb', 'p_lgb']
PRZETERMINOWANIE = 4 * 3600      # osierocony znacznik zwalniany po czterech godzinach


def zajmij(seed: int) -> bool:
    """Zajmuje ziarno atomowo. Zwraca False, jeżeli zajął je już inny proces.

    Tworzenie z flagą O_EXCL jest w systemie plików operacją niepodzielną, więc dwa
    procesy startujące jednocześnie nie mogą obydwa uznać ziarna za wolne. Znacznik
    starszy niż czas przeterminowania traktowany jest jako pozostałość po procesie,
    który padł, i zostaje przejęty.
    """
    znacznik = SEEDS_DIR / f'.zajete_{seed}'
    if znacznik.exists() and time.time() - znacznik.stat().st_mtime > PRZETERMINOWANIE:
        znacznik.unlink(missing_ok=True)
    try:
        fd = os.open(znacznik, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        return False
    os.write(fd, f'{os.getpid()}\n'.encode())
    os.close(fd)
    return True


def main() -> None:
    SEEDS_DIR.mkdir(parents=True, exist_ok=True)
    f, cols = load_modelling_frame()
    y = f['target_liquidity_stress'].astype(int)
    print(f'[pid {os.getpid()}] Obserwacji: {len(f)} | zdarzeń: {int(y.sum())} '
          f'| cech: {len(cols)}')
    print(f'[pid {os.getpid()}] Ziaren do policzenia: {N_SEEDS} '
          f'(przesunięcie {OFFSET})\n', flush=True)

    seed0 = pl.SEED                      # baza zapamiętana przed pętlą
    policzone = 0
    for i in range(OFFSET, OFFSET + N_SEEDS):
        seed = seed0 + i
        out = SEEDS_DIR / f'oof_seed_{seed}.csv'
        if out.exists() or not zajmij(seed):
            print(f'  ziarno {seed}: pominięte (policzone lub w toku gdzie indziej)',
                  flush=True)
            continue
        t0 = time.time()
        pl.SEED = seed
        ml.SEED = seed
        res = ml.run_walk_forward(f, cols, verbose=False)
        oof = res['oof']
        oof['y_true'] = res['y'].reindex(oof.index)
        oof.to_csv(out)                  # zapis natychmiast po policzeniu
        policzone += 1

        cov = oof.dropna(subset=KOLUMNY)
        yy = cov['y_true'].astype(int)
        auc = {k: roc_auc_score(yy, cov[k]) for k in KOLUMNY} if yy.nunique() > 1 else {}
        opis = '  '.join(f'{k[2:].upper()} {v:.3f}' for k, v in auc.items())
        print(f'  ziarno {seed}: {len(cov)} sesji, {int(yy.sum())} zdarzeń, '
              f'{(time.time() - t0) / 60:.0f} min | {opis}', flush=True)

    gotowe = len(list(SEEDS_DIR.glob('oof_seed_*.csv')))
    print(f'\n[pid {os.getpid()}] Koniec. Ten proces policzył {policzone}, '
          f'plików łącznie: {gotowe}. Katalog: {SEEDS_DIR}')


if __name__ == '__main__':
    main()
