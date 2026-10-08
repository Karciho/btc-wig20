#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Warstwa modelowa pipeline'u: trening modeli bazowych, autoenkodera i zespołu
w schemacie purged walk-forward.

Architektury i hiperparametry pozostają takie, jak w pierwotnym projekcie badania.
Zmieniony został wyłącznie schemat treningu i oceny, zgodnie z diagnozą opisaną
w METODYKA_ZMIANY.md:

  - każda prognoza wykorzystywana do oceny powstaje poza próbą treningową modelu,
    a między oknem treningowym a prognozowanym zachowywane jest embargo pokrywające
    okno zmiennej objaśnianej,
  - autoenkoder jest dopasowywany osobno w każdym foldzie, wyłącznie na spokojnych
    sesjach okna treningowego, więc jego cechy również są pozapróbkowe,
  - prognozy modeli bazowych są kalibrowane przed przekazaniem do meta-modelu,
  - meta-model w foldzie f uczy się wyłącznie na prognozach z foldów wcześniejszych.
"""
from __future__ import annotations

import json
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

import pipeline_lib as pl

warnings.filterwarnings('ignore')
os.environ.setdefault('KERAS_BACKEND', 'torch')
os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')

BASE = Path(__file__).parent
N_SPLITS = 10
MIN_TRAIN_FRAC = 0.28
INNER_VAL_FRAC = 0.15      # ogon okna treningowego na wczesne zatrzymanie i kalibrację
META_MIN_EVENTS = 15       # minimalna liczba zdarzeń, by uczyć meta-model


# ────────────────────────────────────────────────────────────────────
# Modele bazowe
# ────────────────────────────────────────────────────────────────────

def _seed_everything(seed: int | None = None) -> None:
    """Ustawia ziarno wszystkich generatorów losowych używanych w potoku.

    Wartość odczytywana jest z `pl.SEED` w chwili wywołania, a nie w chwili definicji
    funkcji. Zapis `seed: int = pl.SEED` wiązałby wartość domyślną przy imporcie
    modułu, przez co późniejsza zmiana `pl.SEED` nie miałaby wpływu na sieci
    neuronowe — a modele drzewiaste, czytające `pl.SEED` w treści funkcji, zmieniałyby
    się normalnie. Powtórzenia dla wielu ziaren dawałyby wtedy identyczne wyniki sieci
    i zmienne wyniki drzew, co zafałszowałoby ocenę wrażliwości na losowanie.
    """
    import random
    seed = pl.SEED if seed is None else seed
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass


def make_sequences(X: np.ndarray, y: np.ndarray, lookback: int = pl.LOOKBACK):
    """Zamienia tablicę cech na sekwencje o długości `lookback`."""
    xs, ys, pos = [], [], []
    for i in range(lookback - 1, len(X)):
        xs.append(X[i - lookback + 1:i + 1])
        ys.append(y[i])
        pos.append(i)
    if not xs:
        return np.empty((0, lookback, X.shape[1])), np.empty(0), np.empty(0, dtype=int)
    return np.asarray(xs, dtype='float32'), np.asarray(ys), np.asarray(pos, dtype=int)


def build_rnn(kind: str, n_features: int, lookback: int = pl.LOOKBACK):
    import keras
    from keras import layers, regularizers
    Rec = layers.LSTM if kind == 'lstm' else layers.GRU
    inp = keras.Input(shape=(lookback, n_features))
    x = Rec(128, return_sequences=True, kernel_regularizer=regularizers.l2(1e-4))(inp)
    x = layers.Dropout(0.3)(x)
    x = Rec(64, kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.Dropout(0.3)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(32, activation='relu')(x)
    x = layers.Dropout(0.2)(x)
    out = layers.Dense(1, activation='sigmoid')(x)
    model = keras.Model(inp, out)
    model.compile(optimizer=keras.optimizers.Adam(1e-3, clipnorm=1.0),
                  loss='binary_crossentropy',
                  metrics=[keras.metrics.AUC(name='auc')])
    return model


def fit_predict_rnn(kind, Xtr, ytr, Xva, yva, Xte, n_features):
    import keras
    _seed_everything()
    model = build_rnn(kind, n_features)
    cw = pl.class_weight_for(ytr)
    cbs = [keras.callbacks.EarlyStopping(monitor='val_auc', mode='max', patience=15,
                                         restore_best_weights=True, verbose=0),
           keras.callbacks.ReduceLROnPlateau(monitor='val_auc', mode='max', factor=0.5,
                                             patience=7, verbose=0)]
    validation = (Xva, yva) if len(Xva) and len(np.unique(yva)) > 1 else None
    model.fit(Xtr, ytr, validation_data=validation, epochs=100, batch_size=32,
              class_weight={0: 1.0, 1: cw}, callbacks=cbs if validation else None, verbose=0)
    p_va = model.predict(Xva, verbose=0).ravel() if len(Xva) else np.empty(0)
    p_te = model.predict(Xte, verbose=0).ravel() if len(Xte) else np.empty(0)
    return model, p_va, p_te


def fit_predict_trees(kind, Xtr, ytr, Xva, Xte):
    _seed_everything()
    spw = pl.class_weight_for(ytr)
    if kind == 'xgb':
        from xgboost import XGBClassifier
        model = XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1,
                              subsample=0.8, colsample_bytree=0.8,
                              scale_pos_weight=spw, reg_alpha=0.1, reg_lambda=1.0,
                              eval_metric='aucpr', random_state=pl.SEED, n_jobs=4)
    else:
        from lightgbm import LGBMClassifier
        model = LGBMClassifier(n_estimators=100, num_leaves=31, learning_rate=0.1,
                               subsample=0.8, colsample_bytree=0.8, max_depth=-1,
                               class_weight='balanced', min_child_samples=20,
                               random_state=pl.SEED, n_jobs=4, verbose=-1)
    model.fit(Xtr, ytr)
    p_va = model.predict_proba(Xva)[:, 1] if len(Xva) else np.empty(0)
    p_te = model.predict_proba(Xte)[:, 1] if len(Xte) else np.empty(0)
    return model, p_va, p_te


def fit_autoencoder(X_normal, n_features, encoding_dim=16):
    import keras
    from keras import layers
    _seed_everything()
    inp = keras.Input(shape=(n_features,))
    x = layers.Dense(64, activation='relu')(inp)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.1)(x)
    x = layers.Dense(32, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    z = layers.Dense(encoding_dim, activation='relu')(x)
    x = layers.Dense(32, activation='relu')(z)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.1)(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    out = layers.Dense(n_features, activation='linear')(x)
    ae = keras.Model(inp, out)
    ae.compile(optimizer=keras.optimizers.Adam(1e-3), loss='mse')
    ae.fit(X_normal, X_normal, epochs=100, batch_size=32, validation_split=0.1,
           callbacks=[keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True,
                                                    verbose=0)], verbose=0)
    return ae


def reconstruction_error(ae, X):
    rec = ae.predict(X, verbose=0)
    return np.mean((X - rec) ** 2, axis=1)


# ────────────────────────────────────────────────────────────────────
# Walk-forward
# ────────────────────────────────────────────────────────────────────

def run_walk_forward(feat: pd.DataFrame, feature_cols: list[str],
                     target_col: str = 'target_liquidity_stress',
                     n_splits: int = N_SPLITS,
                     min_train_frac: float = MIN_TRAIN_FRAC,
                     verbose: bool = True) -> dict:
    """Pełny przebieg walk-forward dla wszystkich modeli bazowych i zespołu.

    Zwraca słownik z ramkami prognoz pozapróbkowych, opisem foldów i modelami
    dopasowanymi na całości danych do końca okresu walidacyjnego.
    """
    from sklearn.preprocessing import StandardScaler

    df = feat.dropna(subset=[target_col]).copy()
    y_all = df[target_col].astype(int)
    idx_all = df.index
    folds = pl.purged_walk_forward_splits(idx_all, n_splits=n_splits,
                                          min_train_frac=min_train_frac)

    base_names = ['lstm', 'gru', 'xgb', 'lgb']
    oof = pd.DataFrame(index=idx_all, columns=[f'p_{b}' for b in base_names] +
                       ['ae_re', 'p_ens', 'fold'], dtype=float)
    oof['fold'] = np.nan
    fold_info = []

    for f_i, (tr_idx, te_idx) in enumerate(folds, start=1):
        # wewnętrzny podział okna treningowego na część uczącą i walidacyjną
        n_tr = len(tr_idx)
        n_inner = max(int(n_tr * INNER_VAL_FRAC), 60)
        inner_tr = tr_idx[:n_tr - n_inner - pl.EMBARGO]
        inner_va = tr_idx[n_tr - n_inner:]

        Xtr_raw = df.loc[inner_tr, feature_cols].values
        Xva_raw = df.loc[inner_va, feature_cols].values
        Xte_raw = df.loc[te_idx, feature_cols].values
        ytr = y_all.loc[inner_tr].values
        yva = y_all.loc[inner_va].values
        yte = y_all.loc[te_idx].values

        scaler = StandardScaler().fit(Xtr_raw)
        Xtr, Xva, Xte = (scaler.transform(a) for a in (Xtr_raw, Xva_raw, Xte_raw))

        # ── autoenkoder: uczony na spokojnych sesjach okna uczącego ──
        X_normal = Xtr[ytr == 0]
        ae = fit_autoencoder(X_normal, Xtr.shape[1])
        re_tr_normal = reconstruction_error(ae, X_normal)
        ae_thr = float(np.quantile(re_tr_normal, 0.95))
        re_va, re_te = reconstruction_error(ae, Xva), reconstruction_error(ae, Xte)
        oof.loc[te_idx, 'ae_re'] = re_te

        # cechy rozszerzone o wyjście autoenkodera (bez wycieku: AE z okna uczącego)
        def augment(Xz, re):
            return np.column_stack([Xz, re, (re > ae_thr).astype(float)])
        Xtr_a, Xva_a, Xte_a = (augment(Xtr, reconstruction_error(ae, Xtr)),
                               augment(Xva, re_va), augment(Xte, re_te))

        preds_va, preds_te = {}, {}

        # ── sieci rekurencyjne ──
        seq_source_tr = np.vstack([Xtr_a])
        for kind in ('lstm', 'gru'):
            Xs_tr, ys_tr, _ = make_sequences(Xtr_a, ytr)
            Xs_va, ys_va, _ = make_sequences(Xva_a, yva)
            Xs_te, _, pos_te = make_sequences(Xte_a, yte)
            _, p_va, p_te = fit_predict_rnn(kind, Xs_tr, ys_tr, Xs_va, ys_va,
                                            Xs_te, Xtr_a.shape[1])
            # pierwsze (lookback-1) sesji bloku prognozowanego nie ma pełnej sekwencji
            full_te = np.full(len(te_idx), np.nan)
            full_te[pos_te] = p_te
            full_va = np.full(len(inner_va), np.nan)
            _, _, pos_va = make_sequences(Xva_a, yva)
            full_va[pos_va] = p_va
            preds_va[kind], preds_te[kind] = full_va, full_te

        # ── modele drzewiaste ──
        for kind in ('xgb', 'lgb'):
            _, p_va, p_te = fit_predict_trees(kind, Xtr_a, ytr, Xva_a, Xte_a)
            preds_va[kind], preds_te[kind] = p_va, p_te

        # ── kalibracja na wewnętrznym zbiorze walidacyjnym ──
        calibrated_te = {}
        for kind in base_names:
            pv, pt = preds_va[kind], preds_te[kind]
            mv = ~np.isnan(pv)
            if mv.sum() > 30 and len(np.unique(yva[mv])) > 1:
                cal = pl.fit_calibrator(pv[mv], yva[mv], method='platt')
                pt_cal = np.full_like(pt, np.nan, dtype=float)
                mt = ~np.isnan(pt)
                pt_cal[mt] = pl.apply_calibrator(cal, pt[mt])
            else:
                pt_cal = pt
            calibrated_te[kind] = pt_cal
            oof.loc[te_idx, f'p_{kind}'] = pt_cal

        oof.loc[te_idx, 'fold'] = f_i
        fold_info.append(dict(
            fold=f_i,
            train_start=str(tr_idx.min().date()), train_end=str(tr_idx.max().date()),
            train_n=int(len(tr_idx)), train_events=int(y_all.loc[tr_idx].sum()),
            test_start=str(te_idx.min().date()), test_end=str(te_idx.max().date()),
            test_n=int(len(te_idx)), test_events=int(yte.sum()),
            ae_threshold=round(ae_thr, 5)))
        if verbose:
            print(f'  fold {f_i}/{len(folds)}: trening do {tr_idx.max().date()} '
                  f'({len(tr_idx)} sesji, {int(y_all.loc[tr_idx].sum())} zdarzeń) → '
                  f'prognoza {te_idx.min().date()}–{te_idx.max().date()} '
                  f'({len(te_idx)} sesji, {int(yte.sum())} zdarzeń)')

    oof = add_meta_variants(oof, y_all, base_names)
    return dict(oof=oof, folds=pd.DataFrame(fold_info), y=y_all,
                feature_cols=feature_cols, base_names=base_names)


# ────────────────────────────────────────────────────────────────────
# Warstwa meta: warianty agregacji prognoz bazowych
# ────────────────────────────────────────────────────────────────────

LOGIT_EPS = 1e-6


def _to_logit(X: np.ndarray) -> np.ndarray:
    """Przenosi prognozy składowe na skalę logitową.

    Prognozy modeli bazowych rozciągają się na kilkadziesiąt rzędów wielkości
    (od 10⁻⁷⁵ do jedności), więc liniowa kombinacja surowych prawdopodobieństw jest
    zdominowana przez model o największych wartościach bezwzględnych, niezależnie od
    jego jakości rankingowej. Skala logitowa usuwa ten efekt i jest naturalną
    przestrzenią dla meta-modelu logistycznego.
    """
    from scipy.special import logit as _logit
    return _logit(np.clip(X, LOGIT_EPS, 1 - LOGIT_EPS))


def _fit_nonneg_logit(X_tr: np.ndarray, y_tr: np.ndarray, X_te: np.ndarray) -> np.ndarray:
    """Regresja logistyczna z ograniczeniem nieujemności współczynników.

    Ograniczenie Breimana nałożone na model o funkcji straty właściwej dla klasyfikacji
    probabilistycznej. Wejściem są logity prognoz składowych, nie surowe
    prawdopodobieństwa. Wyraz wolny pozostaje swobodny, ponieważ ogranicza się kierunek
    wpływu składowych, a nie poziom bazowy.
    """
    from scipy.optimize import minimize
    from scipy.special import expit

    X_tr, X_te = _to_logit(X_tr), _to_logit(X_te)
    n, k = X_tr.shape
    wagi = np.where(y_tr == 1, (len(y_tr) - y_tr.sum()) / max(y_tr.sum(), 1), 1.0)

    def strata(theta):
        z = X_tr @ theta[1:] + theta[0]
        p = np.clip(expit(z), 1e-9, 1 - 1e-9)
        return -np.average(y_tr * np.log(p) + (1 - y_tr) * np.log(1 - p), weights=wagi)

    start = np.concatenate([[0.0], np.full(k, 1.0 / k)])
    res = minimize(strata, start, method='L-BFGS-B',
                   bounds=[(None, None)] + [(0.0, None)] * k)
    if not res.success and not np.all(np.isfinite(res.x)):
        return X_te.mean(axis=1)
    return expit(X_te @ res.x[1:] + res.x[0])


def add_meta_variants(oof: pd.DataFrame, y_all: pd.Series,
                      base_names: list[str] | None = None) -> pd.DataFrame:
    """Dolicza trzy warianty agregacji prognoz modeli bazowych.

    `p_ens`      — stacking z ograniczeniem nieujemności wag (Breiman, 1996) rozwiązywany
                   metodą najmniejszych kwadratów; wariant historyczny, z którego
                   wyprowadzany był Hybrydowy Indeks Ryzyka,
    `p_ens_logit` — stacking z regresją logistyczną o współczynnikach nieujemnych;
                   to samo ograniczenie Breimana, lecz przy funkcji straty właściwej
                   dla klasyfikacji probabilistycznej (log-loss zamiast błędu
                   kwadratowego względem etykiet 0/1),
    `p_ens_lr`   — stacking z nieograniczoną regresją logistyczną; wariant zachowany
                   dla udokumentowania trybu awarii przy zmianie reżimu rynkowego,
    `p_ens_avg`  — proste uśrednienie prognoz (soft voting) jako punkt odniesienia.

    Meta-model w foldzie f uczy się wyłącznie na prognozach z foldów wcześniejszych,
    więc żadna prognoza zespołu nie korzysta z obserwacji, na których jest oceniana.
    """
    from scipy.optimize import nnls
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler as SS

    base_names = base_names or ['lstm', 'gru', 'xgb', 'lgb']
    cols = [f'p_{b}' for b in base_names]
    oof = oof.copy()
    for c in ('p_ens', 'p_ens_logit', 'p_ens_lr', 'p_ens_avg'):
        oof[c] = np.nan
    weights_log = []

    oof['p_ens_avg'] = oof[cols].mean(axis=1)

    for f_i in sorted(oof['fold'].dropna().unique()):
        prior = oof[oof['fold'] < f_i].dropna(subset=cols)
        cur = oof[oof['fold'] == f_i].dropna(subset=cols)
        if not len(cur):
            continue
        y_prior = y_all.reindex(prior.index)
        if len(prior) < 100 or int(y_prior.sum()) < META_MIN_EVENTS:
            continue

        # wariant podstawowy: wagi nieujemne, znormalizowane do sumy jednostkowej
        w, _ = nnls(prior[cols].values, y_prior.astype(float).values)
        if w.sum() <= 0:
            w = np.ones(len(cols))
        w = w / w.sum()
        oof.loc[cur.index, 'p_ens'] = cur[cols].values @ w
        weights_log.append(dict(fold=int(f_i),
                                **{c.replace('p_', ''): round(float(v), 4)
                                   for c, v in zip(cols, w)}))

        # wariant poprawny metodycznie: to samo ograniczenie nieujemności, lecz przy
        # log-loss. NNLS minimalizuje błąd kwadratowy wobec etykiet 0/1, co dla zadania
        # klasyfikacji probabilistycznej nie jest właściwą funkcją straty.
        oof.loc[cur.index, 'p_ens_logit'] = _fit_nonneg_logit(
            prior[cols].values, y_prior.astype(int).values, cur[cols].values)

        # wariant nieograniczony (dokumentacja trybu awarii)
        sc = SS().fit(prior[cols].values)
        lr = LogisticRegression(C=1.0, class_weight='balanced', max_iter=2000)
        lr.fit(sc.transform(prior[cols].values), y_prior.astype(int).values)
        oof.loc[cur.index, 'p_ens_lr'] = lr.predict_proba(sc.transform(cur[cols].values))[:, 1]

    oof.attrs['meta_weights'] = weights_log
    return oof


def pooled_metrics(res: dict, threshold_source: str = 'oof') -> pd.DataFrame:
    """Metryki pozapróbkowe na połączonych prognozach ze wszystkich foldów.

    Miary rankingowe (AUC-ROC, AUC-PR, wynik Briera) nie zależą od progu i liczone
    są na całym okresie pozapróbkowym.

    Miary progowe (precyzja, czułość, F1, trafność, macierz pomyłek) raportowane są
    w dwóch wariantach, ponieważ próg dobierany jest na pierwszej połowie prognoz:
      * kolumny bez przyrostka — cały okres pozapróbkowy; wariant obciążony, bo
        obejmuje obserwacje, na których próg wybrano,
      * kolumny z przyrostkiem `_clean` — wyłącznie druga połowa, nieużyta przy
        wyborze progu; wariant nieobciążony, lecz oparty na mniejszej liczbie zdarzeń.
    Wcześniej funkcja deklarowała w opisie ocenę na drugiej połowie, a liczyła ją
    na całości — raportowane F1 było przez to zaniżone i metodycznie niespójne.
    """
    oof, y = res['oof'], res['y']
    rows = []
    common = oof.dropna(subset=['p_ens']).index         # wspólny podzbiór dla porównywalności
    names = res['base_names'] + ['ens', 'ens_avg', 'ens_lr']
    labels = {'lstm': 'LSTM', 'gru': 'GRU', 'xgb': 'XGBoost', 'lgb': 'LightGBM',
              'ens': 'Zespół (stacking z wagami nieujemnymi)',
              'ens_avg': 'Zespół (uśrednianie prognoz)',
              'ens_lr': 'Zespół (stacking nieograniczony)'}
    for name in names:
        col = f'p_{name}'
        if col not in oof.columns:
            continue
        sub = oof.loc[common, col].dropna()
        if not len(sub):
            continue
        yy = y.reindex(sub.index).astype(int).values
        pp = sub.values
        half = len(sub) // 2
        thr = pl.select_threshold(yy[:half], pp[:half]) if yy[:half].sum() > 0 else 0.5
        m = pl.evaluate(yy, pp, thr)
        clean = pl.evaluate(yy[half:], pp[half:], thr)
        for k in ('accuracy', 'precision', 'recall', 'f1', 'tp', 'fp', 'fn', 'tn'):
            m[f'{k}_clean'] = clean[k]
        m['n_clean'] = len(yy[half:])
        m['n_positive_clean'] = int(yy[half:].sum())
        m['model'] = labels[name]
        rows.append(m)
    cols = ['model', 'accuracy', 'precision', 'recall', 'f1', 'auc_roc', 'auc_pr',
            'auc_pr_lift', 'brier', 'threshold', 'n', 'n_positive', 'tp', 'fp', 'fn', 'tn',
            'accuracy_clean', 'precision_clean', 'recall_clean', 'f1_clean',
            'n_clean', 'n_positive_clean', 'tp_clean', 'fp_clean', 'fn_clean', 'tn_clean']
    return pd.DataFrame(rows)[cols].set_index('model')
