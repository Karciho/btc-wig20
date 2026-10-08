#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Wspólna warstwa metodyczna pipeline'u (notebooki 04–10).

Moduł powstał w odpowiedzi na diagnozę zdegenerowanych wyników wcześniejszej wersji
badania (opis w METODYKA_ZMIANY.md). Skupia w jednym, testowalnym miejscu te elementy
metodyki, w których popełniono błędy konstrukcyjne:

  1. definicję zmiennej objaśnianej („susza płynnościowa") jako zdarzenia rzadkiego,
     opartej na złożonym wskaźniku warunków płynnościowych, z progiem wyznaczonym
     wyłącznie na okresie treningowym,
  2. podział chronologiczny próby oraz walidację typu purged walk-forward z embargiem,
     eliminującą wyciek informacji przez nakładające się okna zmiennej objaśnianej,
  3. kalibrację prognoz probabilistycznych,
  4. wybór progu decyzyjnego na zbiorze walidacyjnym,
  5. wyznaczanie progu Hybrydowego Indeksu Ryzyka z rozkładu, a nie arbitralnie,
  6. analizę zdarzeń liczoną na kalendarzu sesyjnym GPW.

Wszystkie funkcje są czyste (bez efektów ubocznych poza zapisem plików tam, gdzie
zaznaczono) i mogą być wywoływane zarówno z notebooków, jak i z testów.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# ────────────────────────────────────────────────────────────────────
# Konfiguracja globalna badania
# ────────────────────────────────────────────────────────────────────

SEED = 42

TRAIN_END = '2023-12-31'      # koniec okresu treningowego
VAL_END = '2024-12-31'        # koniec okresu walidacyjnego (test = od 2025-01-01)

HORIZON = 5                   # horyzont prognozy w sesjach
EMBARGO = HORIZON + 5         # embargo między foldami (okno celu + bufor)
TARGET_QUANTILE = 0.90        # próg suszy płynnościowej: najgorszy decyl okresu treningowego
HRI_QUANTILE = 0.90           # próg ostrzegawczy HRI: kwantyl rozkładu z okresu referencyjnego
ALPHA = 0.05                  # poziom istotności w testach raportowanych w pracy
LOOKBACK = 20                 # długość sekwencji dla sieci rekurencyjnych
# Kolumny wyłączone ze zbioru cech podawanych modelom. Prefiks liq_ to składowe
# wskaźnika, z którego wyprowadzono zmienną objaśnianą — ich podanie modelowi
# byłoby udostępnieniem konstrukcji celu. Prefiks btc_1h_ to agregaty śróddzienne,
# odrzucone, bo dla GPW brak odpowiednika w horyzoncie całej próby (zob. pkt 3.1.1).
# Definicja wspólna dla run_walkforward.py, run_ablation.py i empirical_values.py,
# żeby liczba cech podawana w tekście nie mogła rozejść się z liczbą faktycznie użytą.
MODEL_FEATURE_PREFIXES_DROP = ('btc_1h_', 'liq_')

# progi alternatywne do analizy wrażliwości
SENSITIVITY_QUANTILES = (0.85, 0.90, 0.95)
SENSITIVITY_HORIZONS = (3, 5, 10)


# ────────────────────────────────────────────────────────────────────
# 1. Wskaźnik warunków płynnościowych i zmienna objaśniana
# ────────────────────────────────────────────────────────────────────

def corwin_schultz_spread(high: pd.Series, low: pd.Series) -> pd.Series:
    """Estymator efektywnego spreadu z dziennych maksimów i minimów.

    Implementacja wzoru Corwina i Schultza (2012). Estymator wykorzystuje fakt, że
    rozpiętość notowań w ciągu dnia zawiera zarówno komponent zmienności (rosnący
    proporcjonalnie do czasu), jak i komponent spreadu (stały), co pozwala je rozdzielić
    przez porównanie rozpiętości jednodniowych z dwudniową.

    Wartości ujemne, wynikające z szumu estymacji, zerowane zgodnie z zaleceniem autorów.
    """
    h, l = high.astype(float), low.astype(float)
    hl = np.log(h / l) ** 2
    beta = hl + hl.shift(1)                                   # suma dwóch dni
    h2 = pd.concat([h, h.shift(1)], axis=1).max(axis=1)
    l2 = pd.concat([l, l.shift(1)], axis=1).min(axis=1)
    gamma = np.log(h2 / l2) ** 2
    k = 3 - 2 * np.sqrt(2)
    alpha = (np.sqrt(2 * beta) - np.sqrt(beta)) / k - np.sqrt(gamma / k)
    spread = 2 * (np.exp(alpha) - 1) / (1 + np.exp(alpha))
    return spread.clip(lower=0)


def build_liquidity_stress_index(master: pd.DataFrame,
                                 train_end: str = TRAIN_END) -> pd.DataFrame:
    """Złożony wskaźnik warunków płynnościowych indeksu WIG20.

    Cztery komponenty, każdy rosnący wraz z pogorszeniem warunków handlu:
      - niepłynność Amihuda: wpływ cenowy jednostki obrotu,
      - estymator spreadu Corwina-Schultza: bezpośredni koszt zawarcia transakcji,
      - niedobór obrotu: ubytek wartości obrotu względem średniej z 20 sesji,
      - zmienność zrealizowana z 5 sesji: nerwowość wyceny.

    Komponenty standaryzowane parametrami okresu treningowego (średnia i odchylenie
    standardowe liczone wyłącznie do `train_end`), żeby wskaźnik nie korzystał
    z informacji z przyszłości. Wskaźnik złożony to średnia standaryzowanych komponentów.
    """
    df = master.copy()
    ret = np.log(df['wig_close'] / df['wig_close'].shift(1))
    turnover = df['wig_volume'].astype(float)

    amihud = (ret.abs() / turnover.replace(0, np.nan)) * 1e9
    spread = corwin_schultz_spread(df['wig_high'], df['wig_low'])
    spread = spread.rolling(5, min_periods=3).mean()          # wygładzenie szumu estymatora
    turnover_gap = -np.log(turnover / turnover.rolling(20, min_periods=10).mean())
    realized_vol = ret.rolling(5, min_periods=3).std() * np.sqrt(252)

    comp = pd.DataFrame({
        'liq_amihud': amihud,
        'liq_spread': spread,
        'liq_turnover_gap': turnover_gap,
        'liq_realized_vol': realized_vol,
    })

    train_mask = comp.index <= pd.Timestamp(train_end)
    z = pd.DataFrame(index=comp.index)
    for col in comp.columns:
        mu = comp.loc[train_mask, col].mean()
        sd = comp.loc[train_mask, col].std()
        z['z_' + col] = (comp[col] - mu) / (sd if sd and np.isfinite(sd) else 1.0)

    comp['liq_stress_index'] = z.mean(axis=1)
    return pd.concat([comp, z], axis=1)


def build_target(liq_index: pd.Series, horizon: int = HORIZON,
                 train_end: str = TRAIN_END,
                 quantile: float = TARGET_QUANTILE) -> tuple[pd.Series, float]:
    """Binarna zmienna objaśniana: susza płynnościowa w horyzoncie `horizon` sesji.

    Wartość 1 oznacza, że przeciętne warunki płynnościowe w kolejnych `horizon` sesjach
    znajdą się powyżej progu wyznaczonego jako `quantile` rozkładu tej samej wielkości
    w okresie treningowym.

    Uśrednienie po oknie (zamiast maksimum) jest tu istotne: maksimum sygnalizowałoby
    stres przy pojedynczej gorszej sesji, przez co zdarzenie przestawałoby być rzadkie
    i opisywałoby szum, a nie utrzymujące się pogorszenie warunków handlu.

    Zwraca zmienną objaśnianą oraz wartość progu.
    """
    fwd = (liq_index.shift(-1).rolling(horizon, min_periods=horizon).mean()
           .shift(-(horizon - 1)))
    tau = fwd.loc[:train_end].quantile(quantile)
    target = (fwd > tau).astype('float')
    target[fwd.isna()] = np.nan
    target.name = 'target_liquidity_stress'
    return target, float(tau)


# ────────────────────────────────────────────────────────────────────
# 2. Podział próby i walidacja bez wycieku informacji
# ────────────────────────────────────────────────────────────────────

def split_masks(index: pd.DatetimeIndex, train_end: str = TRAIN_END,
                val_end: str = VAL_END) -> dict[str, pd.Series]:
    idx = pd.DatetimeIndex(index)
    return {
        'train': pd.Series(idx <= pd.Timestamp(train_end), index=idx),
        'val': pd.Series((idx > pd.Timestamp(train_end)) & (idx <= pd.Timestamp(val_end)), index=idx),
        'test': pd.Series(idx > pd.Timestamp(val_end), index=idx),
    }


def purged_walk_forward_splits(index: pd.DatetimeIndex, n_splits: int = 5,
                               embargo: int = EMBARGO,
                               min_train_frac: float = 0.35):
    """Podziały typu purged walk-forward z embargiem.

    Okno treningowe rozszerza się w czasie, a między jego końcem a początkiem okna
    prognozowanego pozostawiana jest przerwa (embargo) o długości pokrywającej okno
    zmiennej objaśnianej. Bez tej przerwy obserwacje treningowe z końca okna zawierałyby
    informację o okresie prognozowanym, ponieważ ich zmienna objaśniana wyznaczana jest
    z kolejnych sesji (López de Prado, 2018).

    Zwraca listę par (indeks treningowy, indeks prognozowany).
    """
    idx = pd.DatetimeIndex(index)
    n = len(idx)
    start = int(n * min_train_frac)
    bounds = np.linspace(start, n, n_splits + 1).astype(int)
    out = []
    for i in range(n_splits):
        tr_end, te_end = bounds[i], bounds[i + 1]
        te_start = tr_end + embargo
        if te_start >= te_end:
            continue
        out.append((idx[:tr_end], idx[te_start:te_end]))
    return out


# ────────────────────────────────────────────────────────────────────
# 3. Kalibracja, próg decyzyjny, metryki
# ────────────────────────────────────────────────────────────────────

def fit_calibrator(p: np.ndarray, y: np.ndarray, method: str = 'isotonic'):
    """Dopasowuje przekształcenie kalibrujące prognozy probabilistyczne.

    Kalibracja izotoniczna jest nieparametryczna i wymaga większej próby; skalowanie
    Platta (regresja logistyczna na logicie prognozy) jest stabilniejsze przy małej
    liczbie obserwacji. Przy mniej niż 200 obserwacjach lub mniej niż 20 zdarzeniach
    klasy pozytywnej metoda przełącza się automatycznie na skalowanie Platta.
    """
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    y = np.asarray(y, dtype=int)
    if method == 'isotonic' and (len(y) < 200 or y.sum() < 20):
        method = 'platt'
    if method == 'isotonic':
        from sklearn.isotonic import IsotonicRegression
        iso = IsotonicRegression(out_of_bounds='clip', y_min=0.0, y_max=1.0)
        iso.fit(p, y)
        return ('isotonic', iso)
    from sklearn.linear_model import LogisticRegression
    lr = LogisticRegression(C=1e6, solver='lbfgs')
    lr.fit(np.log(p / (1 - p)).reshape(-1, 1), y)
    return ('platt', lr)


def apply_calibrator(cal, p: np.ndarray) -> np.ndarray:
    kind, model = cal
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    if kind == 'isotonic':
        return np.clip(model.predict(p), 0.0, 1.0)
    return model.predict_proba(np.log(p / (1 - p)).reshape(-1, 1))[:, 1]


def select_threshold(y: np.ndarray, p: np.ndarray, criterion: str = 'f1') -> float:
    """Wybiera próg decyzyjny na zbiorze walidacyjnym.

    Kryterium `f1` maksymalizuje średnią harmoniczną precyzji i czułości, `youden`
    maksymalizuje sumę czułości i swoistości pomniejszoną o jeden. Próg wybierany
    wyłącznie poza zbiorem testowym.
    """
    from sklearn.metrics import f1_score, roc_curve
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    if criterion == 'youden':
        fpr, tpr, thr = roc_curve(y, p)
        return float(thr[np.argmax(tpr - fpr)])
    grid = np.quantile(p, np.linspace(0.05, 0.95, 91))
    scores = [f1_score(y, (p >= t).astype(int), zero_division=0) for t in grid]
    return float(grid[int(np.argmax(scores))])


def evaluate(y: np.ndarray, p: np.ndarray, threshold: float = 0.5) -> dict:
    """Pełny zestaw metryk dla klasyfikacji niezbalansowanej."""
    from sklearn.metrics import (accuracy_score, average_precision_score, brier_score_loss,
                                 confusion_matrix, f1_score, precision_score, recall_score,
                                 roc_auc_score)
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    yhat = (p >= threshold).astype(int)
    base_rate = float(y.mean())
    out = dict(
        threshold=round(float(threshold), 4),
        accuracy=round(accuracy_score(y, yhat), 4),
        precision=round(precision_score(y, yhat, zero_division=0), 4),
        recall=round(recall_score(y, yhat, zero_division=0), 4),
        f1=round(f1_score(y, yhat, zero_division=0), 4),
        auc_roc=round(roc_auc_score(y, p), 4) if len(np.unique(y)) > 1 else np.nan,
        auc_pr=round(average_precision_score(y, p), 4) if len(np.unique(y)) > 1 else np.nan,
        brier=round(brier_score_loss(y, p), 4),
        base_rate=round(base_rate, 4),
        n=int(len(y)), n_positive=int(y.sum()),
    )
    # przewaga nad klasyfikatorem losowym o tej samej częstości bazowej
    out['auc_pr_lift'] = round(out['auc_pr'] / base_rate, 3) if base_rate > 0 else np.nan
    tn, fp, fn, tp = confusion_matrix(y, yhat, labels=[0, 1]).ravel()
    out.update(tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp))
    return out


# ────────────────────────────────────────────────────────────────────
# 4. Hybrydowy Indeks Ryzyka
# ────────────────────────────────────────────────────────────────────

def hri_threshold_from_distribution(hri_reference: pd.Series,
                                    quantile: float = 0.90) -> float:
    """Próg ostrzegawczy HRI jako kwantyl rozkładu z okresu referencyjnego.

    Okresem referencyjnym jest zbiór walidacyjny — próg nie może korzystać z rozkładu
    prognoz na zbiorze testowym. Kwantyl 0,90 odpowiada częstości zdarzeń przyjętej
    w definicji zmiennej objaśnianej, dzięki czemu sygnał ostrzegawczy jest tak samo
    rzadki jak zjawisko, które ma wykrywać.
    """
    return float(np.nanquantile(hri_reference.astype(float), quantile))


def identify_critical_periods(hri: pd.Series, threshold: float,
                              smooth_window: int = 5,
                              min_length: int = 3,
                              max_gap: int = 2) -> pd.DataFrame:
    """Epizody krytyczne: spójne okresy utrzymywania się HRI powyżej progu.

    Szereg wygładzany średnią ruchomą, przerwy krótsze niż `max_gap` sesji sklejane,
    epizody krótsze niż `min_length` sesji odrzucane jako szum.
    """
    s = hri.astype(float).rolling(smooth_window, min_periods=1).mean()
    above = (s > threshold).astype(int)
    # sklejanie krótkich przerw
    grp = above.copy()
    zeros = 0
    for i in range(len(grp)):
        if grp.iloc[i] == 0:
            zeros += 1
        else:
            if 0 < zeros <= max_gap and i - zeros - 1 >= 0 and grp.iloc[i - zeros - 1] == 1:
                grp.iloc[i - zeros:i] = 1
            zeros = 0
    periods, start = [], None
    for i, (ts, v) in enumerate(grp.items()):
        if v == 1 and start is None:
            start = ts
        elif v == 0 and start is not None:
            end = grp.index[i - 1]
            periods.append((start, end))
            start = None
    if start is not None:
        periods.append((start, grp.index[-1]))

    rows = []
    for s0, s1 in periods:
        seg = s.loc[s0:s1]
        n_sessions = len(seg)
        if n_sessions < min_length:
            continue
        rows.append({
            'Data początku': s0.date(), 'Data końca': s1.date(),
            'Liczba sesji': n_sessions,
            'Czas trwania (dni kalendarzowe)': (s1 - s0).days + 1,
            'Maks. HRI': round(float(hri.loc[s0:s1].max()), 4),
            'Średni HRI': round(float(seg.mean()), 4),
        })
    return pd.DataFrame(rows)


def event_study_car(signal_dates, wig_close: pd.Series, window: int = 10,
                    estimation: int = 60) -> pd.DataFrame:
    """Skumulowane ponadprzeciętne stopy zwrotu WIG20 wokół sygnałów.

    Liczone wyłącznie na kalendarzu sesyjnym GPW: pozycje sygnałów wyznaczane są jako
    indeksy porządkowe w szeregu notowań, a nie jako daty kalendarzowe. Wcześniejsza
    wersja badania mieszała kalendarz siedmiodniowy rynku kryptowalut z kalendarzem
    sesyjnym GPW, przez co znaczna część okien zdarzeń nie znajdowała odpowiedników
    i dawała wartości brakujące.

    Zwrot normalny szacowany jako średnia z `estimation` sesji poprzedzających okno.
    """
    px = wig_close.dropna().astype(float)
    ret = np.log(px / px.shift(1))
    pos_of = {ts: i for i, ts in enumerate(ret.index)}
    rows = []
    for d in pd.DatetimeIndex(signal_dates):
        i = pos_of.get(d)
        if i is None:                              # sygnał w dniu bez sesji GPW
            future = ret.index[ret.index >= d]
            if not len(future):
                continue
            i = pos_of[future[0]]
        if i - window - estimation < 0 or i + window >= len(ret):
            continue
        mu = ret.iloc[i - window - estimation:i - window].mean()
        ar = ret.iloc[i - window:i + window + 1] - mu
        car = ar.cumsum() - ar.iloc[:window].sum()   # normalizacja do t = 0
        rows.append(pd.Series(car.values * 100,
                              index=range(-window, window + 1)))
    if not rows:
        return pd.DataFrame(columns=['day_relative', 'mean_car', 'se', 'n'])
    mat = pd.DataFrame(rows)
    out = pd.DataFrame({
        'day_relative': mat.columns,
        'mean_car': mat.mean().round(4).values,
        'se': (mat.std() / np.sqrt(len(mat))).round(4).values,
        'n': len(mat),
    })
    return out


# ────────────────────────────────────────────────────────────────────
# 5. Pomocnicze
# ────────────────────────────────────────────────────────────────────

def class_weight_for(y: np.ndarray) -> float:
    """Waga klasy pozytywnej równoważąca liczebności klas."""
    y = np.asarray(y, dtype=int)
    pos = max(int(y.sum()), 1)
    return float((len(y) - pos) / pos)


def summarize_target(target: pd.Series, index: pd.DatetimeIndex | None = None) -> dict:
    t = target.dropna()
    masks = split_masks(t.index)
    out = {'n': int(len(t)), 'positive': int(t.sum()),
           'positive_pct': round(100 * float(t.mean()), 2)}
    for name, m in masks.items():
        sub = t[m.reindex(t.index).fillna(False).values]
        out[f'{name}_n'] = int(len(sub))
        out[f'{name}_pct'] = round(100 * float(sub.mean()), 2) if len(sub) else None
    return out
