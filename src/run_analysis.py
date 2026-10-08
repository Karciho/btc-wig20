#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analiza wyników pozapróbkowych: warstwa meta, Hybrydowy Indeks Ryzyka, epizody
krytyczne, przyczynowość Grangera, analiza zdarzeń, interpretowalność i kalibracja.

Skrypt korzysta z prognoz zapisanych przez run_walkforward.py, więc nie wymaga
ponownego treningu modeli. Wywoływany jest z notebooków 08–10.
"""
from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault('KERAS_BACKEND', 'torch')
os.environ.setdefault('MPLBACKEND', 'Agg')

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import models_lib as ml
import pipeline_lib as pl

BASE = Path(__file__).parent
PROC = BASE / 'data' / 'processed'
TABLES = BASE / 'results' / 'tables'
FIGS = BASE / 'results' / 'figures'

COLOR_BTC, COLOR_WIG, COLOR_HRI = '#F7931A', '#003087', '#C0392B'
BASE_NAMES = ['lstm', 'gru', 'xgb', 'lgb']
HRI_QUANTILE = pl.HRI_QUANTILE


def _style():
    """Wspólny wygląd rysunków, w tym polski zapis liczb na osiach.

    Wytyczne wymagają przecinka dziesiętnego w całej pracy. Osie wykresów podlegają
    tej zasadzie tak samo jak tekst, a kontrola tekstowa ich nie obejmuje — stąd
    ustawienie tutaj, przy jednym wspólnym stylu.
    """
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


def pl_fmt():
    """Formatter liczb dla osi, gdy w systemie brak polskiej lokalizacji."""
    from matplotlib.ticker import FuncFormatter
    return FuncFormatter(lambda v, _p: f'{v:g}'.replace('.', ','))


def load() -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    oof = pd.read_csv(PROC / 'oof_walkforward.csv', index_col=0, parse_dates=True)
    oof = ml.add_meta_variants(oof, oof['y_true'].astype(int), BASE_NAMES)
    master = pd.read_csv(PROC / 'master_dataset.csv', index_col=0, parse_dates=True)
    return oof, oof['y_true'].astype(int), master


# ────────────────────────────────────────────────────────────────────
# 1. Porównanie modeli i wag zespołu
# ────────────────────────────────────────────────────────────────────

def model_comparison(oof: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    res = dict(oof=oof, y=y, base_names=BASE_NAMES)
    cmp = ml.pooled_metrics(res)
    cmp.to_csv(TABLES / '08_model_comparison_oos.csv')
    weights = pd.DataFrame(oof.attrs.get('meta_weights', []))
    if len(weights):
        weights.to_csv(TABLES / '08_meta_weights.csv', index=False)
    return cmp


# ────────────────────────────────────────────────────────────────────
# 2. Hybrydowy Indeks Ryzyka
# ────────────────────────────────────────────────────────────────────

def build_hri(oof: pd.DataFrame) -> tuple[pd.Series, float]:
    hri = oof['p_ens'].dropna()
    reference = hri.iloc[:len(hri) // 2]        # okres referencyjny: pierwsza połowa
    thr = pl.hri_threshold_from_distribution(reference, HRI_QUANTILE)
    return hri, thr


def hri_analysis(oof: pd.DataFrame, y: pd.Series, master: pd.DataFrame) -> dict:
    _style()
    hri, thr = build_hri(oof)
    periods = pl.identify_critical_periods(hri, thr)
    periods.to_csv(TABLES / '09_critical_periods.csv', index=False)

    # rysunek: HRI na tle WIG20
    fig, ax = plt.subplots(2, 1, figsize=(12, 7), sharex=True,
                           gridspec_kw={'height_ratios': [2, 1]})
    wig = master['wig_close'].reindex(hri.index)
    ax[0].plot(wig.index, wig.values, color=COLOR_WIG, lw=1.0)
    ax[0].set_ylabel('WIG20 (pkt)')
    ax[0].set_title('Hybrydowy Indeks Ryzyka na tle notowań WIG20')
    smooth = hri.rolling(5, min_periods=1).mean()
    ax[1].plot(hri.index, hri.values, color='grey', lw=0.6, alpha=0.6, label='HRI')
    ax[1].plot(smooth.index, smooth.values, color=COLOR_HRI, lw=1.2, label='HRI (śr. 5-sesyjna)')
    ax[1].axhline(thr, color='black', ls='--', lw=0.9,
                  label=f'próg ostrzegawczy = {thr:.3f}'.replace('.', ','))
    ax[1].set_ylabel('HRI')
    ax[1].legend(fontsize=8, loc='upper right')
    for _, r in periods.iterrows():
        for a in ax:
            a.axvspan(pd.Timestamp(r['Data początku']), pd.Timestamp(r['Data końca']),
                      color=COLOR_HRI, alpha=0.12)
    plt.tight_layout()
    for ext in ('png', 'pdf'):
        plt.savefig(FIGS / f'fig_23_hri_vs_wig20.{ext}', bbox_inches='tight')
    plt.close()

    # przyczynowość Grangera
    granger = granger_tests(hri, master)
    # analiza zdarzeń
    signals = [pd.Timestamp(r['Data początku']) for _, r in periods.iterrows()]
    car = pl.event_study_car(signals, master['wig_close'])
    car.to_csv(TABLES / '09_event_study_car.csv', index=False)
    plot_event_study(car)

    return dict(hri=hri, threshold=thr, periods=periods, granger=granger, car=car)


def granger_tests(hri: pd.Series, master: pd.DataFrame, maxlag: int = 10) -> pd.DataFrame:
    from statsmodels.tsa.stattools import grangercausalitytests
    ret = np.log(master['wig_close'] / master['wig_close'].shift(1))
    df = pd.concat([hri.rename('hri'), ret.rename('wig_ret')], axis=1).dropna()
    rows = []
    for lag in range(1, maxlag + 1):
        try:
            r1 = grangercausalitytests(df[['wig_ret', 'hri']], maxlag=[lag], verbose=False)
            p_hri_wig = r1[lag][0]['ssr_ftest'][1]
        except Exception:                                    # noqa: BLE001
            p_hri_wig = np.nan
        try:
            r2 = grangercausalitytests(df[['hri', 'wig_ret']], maxlag=[lag], verbose=False)
            p_wig_hri = r2[lag][0]['ssr_ftest'][1]
        except Exception:                                    # noqa: BLE001
            p_wig_hri = np.nan
        rows.append({'Lag': lag, 'HRI → WIG20': round(p_hri_wig, 5),
                     'WIG20 → HRI': round(p_wig_hri, 5)})
    out = pd.DataFrame(rows).set_index('Lag')
    out.to_csv(TABLES / '09_granger_hri_wig20.csv')
    return out


def plot_event_study(car: pd.DataFrame) -> None:
    if not len(car):
        return
    _style()
    fig, ax = plt.subplots(figsize=(9, 5))
    x = car['day_relative'].values
    m = car['mean_car'].values
    se = car['se'].values
    ax.plot(x, m, color=COLOR_HRI, lw=1.6, marker='o', ms=3)
    ax.fill_between(x, m - 1.96 * se, m + 1.96 * se, color=COLOR_HRI, alpha=0.15)
    ax.axvline(0, color='black', lw=0.8, ls='--')
    ax.axhline(0, color='grey', lw=0.6)
    ax.set_xlabel('Sesje względem sygnału')
    ax.set_ylabel('Skumulowana ponadprzeciętna stopa zwrotu (%)')
    ax.set_title('Reakcja indeksu WIG20 na sygnały ostrzegawcze HRI')
    plt.tight_layout()
    for ext in ('png', 'pdf'):
        plt.savefig(FIGS / f'fig_25_event_study_car.{ext}', bbox_inches='tight')
    plt.close()


# ────────────────────────────────────────────────────────────────────
# 3. Kalibracja i krzywe oceny
# ────────────────────────────────────────────────────────────────────

def calibration_and_curves(oof: pd.DataFrame, y: pd.Series) -> None:
    from sklearn.calibration import calibration_curve
    from sklearn.metrics import precision_recall_curve, roc_curve
    _style()
    common = oof.dropna(subset=['p_ens']).index
    series = {'LSTM': 'p_lstm', 'GRU': 'p_gru', 'XGBoost': 'p_xgb',
              'LightGBM': 'p_lgb', 'Zespół': 'p_ens'}

    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    for label, col in series.items():
        s = oof.loc[common, col].dropna()
        yy = y.reindex(s.index).values
        fpr, tpr, _ = roc_curve(yy, s.values)
        ax[0].plot(fpr, tpr, lw=1.3, label=label)
        prec, rec, _ = precision_recall_curve(yy, s.values)
        ax[1].plot(rec, prec, lw=1.3, label=label)
    ax[0].plot([0, 1], [0, 1], 'k--', lw=0.8)
    ax[0].set_xlabel('Odsetek fałszywych alarmów')
    ax[0].set_ylabel('Czułość')
    ax[0].set_title('Krzywe ROC')
    ax[0].legend(fontsize=8)
    base_rate = float(y.reindex(common).mean())
    ax[1].axhline(base_rate, color='k', ls='--', lw=0.8,
                  label=f'częstość bazowa = {base_rate:.3f}'.replace('.', ','))
    ax[1].set_xlabel('Czułość')
    ax[1].set_ylabel('Precyzja')
    ax[1].set_title('Krzywe precyzja–czułość')
    ax[1].legend(fontsize=8)
    plt.tight_layout()
    for ext in ('png', 'pdf'):
        plt.savefig(FIGS / f'fig_26_roc_pr_curves.{ext}', bbox_inches='tight')
    plt.close()

    fig, ax = plt.subplots(figsize=(7, 6))
    for label, col in series.items():
        s = oof.loc[common, col].dropna()
        yy = y.reindex(s.index).values
        try:
            frac, mean_pred = calibration_curve(yy, s.values, n_bins=8, strategy='quantile')
            ax.plot(mean_pred, frac, marker='o', ms=4, lw=1.2, label=label)
        except Exception:                                    # noqa: BLE001
            continue
    ax.plot([0, 1], [0, 1], 'k--', lw=0.8, label='kalibracja idealna')
    ax.set_xlabel('Prognozowane prawdopodobieństwo')
    ax.set_ylabel('Zaobserwowana częstość zdarzeń')
    ax.set_title('Krzywe kalibracji prognoz')
    ax.legend(fontsize=8)
    plt.tight_layout()
    for ext in ('png', 'pdf'):
        plt.savefig(FIGS / f'fig_28_calibration_curves.{ext}', bbox_inches='tight')
    plt.close()


# ────────────────────────────────────────────────────────────────────
# 4. Interpretowalność (SHAP) na modelu dopasowanym do okresu uczącego
# ────────────────────────────────────────────────────────────────────

def shap_analysis() -> pd.DataFrame:
    import shap
    from run_walkforward import load_modelling_frame
    from sklearn.preprocessing import StandardScaler
    _style()

    f, cols = load_modelling_frame()
    train = f.loc[:pl.VAL_END]
    test = f.loc[pl.VAL_END:]
    scaler = StandardScaler().fit(train[cols].values)
    Xtr, Xte = scaler.transform(train[cols].values), scaler.transform(test[cols].values)
    ytr = train['target_liquidity_stress'].astype(int).values

    model, _, _ = ml.fit_predict_trees('xgb', Xtr, ytr, Xte[:1], Xte)
    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(Xte)
    imp = pd.DataFrame({'Cecha': cols, 'Mean |SHAP|': np.abs(sv).mean(axis=0)})
    imp = imp.sort_values('Mean |SHAP|', ascending=False).reset_index(drop=True)
    imp.to_csv(TABLES / '10_shap_importance.csv', index=False)

    plt.figure(figsize=(9, 7))
    shap.summary_plot(sv, features=Xte, feature_names=cols, show=False, max_display=15)
    plt.tight_layout()
    for ext in ('png', 'pdf'):
        plt.savefig(FIGS / f'fig_29_shap_summary.{ext}', bbox_inches='tight')
    plt.close()
    return imp


# ────────────────────────────────────────────────────────────────────
# 5. Raport zbiorczy
# ────────────────────────────────────────────────────────────────────

def write_summary(cmp: pd.DataFrame, hri_res: dict, shap_imp: pd.DataFrame,
                  oof: pd.DataFrame, y: pd.Series) -> None:
    folds = pd.read_csv(TABLES / '05_walkforward_folds.csv')
    sens = pd.read_csv(TABLES / '04_target_sensitivity.csv')
    common = oof.dropna(subset=['p_ens']).index
    lines = [
        '# Raport zbiorczy z badania',
        '',
        f'**Data wygenerowania:** {pd.Timestamp.now():%Y-%m-%d %H:%M}',
        '',
        '## 1. Dane',
        '',
        '| Seria | Źródło | Uwagi |',
        '|---|---|---|',
        '| BTC/USD | Binance REST API (BTCUSDT, interwał dzienny) | wolumen = obrót w USDT |',
        '| WIG20 | Biznesradar.pl (dane GPW) | wolumen = wartość obrotu w PLN |',
        '| VIX | Cboe za pośrednictwem Yahoo Finance | |',
        '| DXY | ICE za pośrednictwem Yahoo Finance | |',
        '| Stopa funduszy federalnych | FRED, seria dzienna DFF | |',
        '',
        'Pełny opis pochodzenia danych, kontroli krzyżowych i punktów kontrolnych: '
        '`DANE_POCHODZENIE.md`. Log maszynowy: `data/raw/_provenance_log.json`.',
        '',
        '## 2. Zmienna objaśniana',
        '',
        'Susza płynnościowa zdefiniowana jako przekroczenie przez średnie warunki '
        'płynnościowe WIG20 w kolejnych pięciu sesjach progu równego 90. percentylowi '
        'rozkładu z okresu treningowego. Wskaźnik warunków płynnościowych łączy cztery '
        'standaryzowane komponenty: niepłynność Amihuda, estymator spreadu '
        'Corwina-Schultza, niedobór obrotu względem średniej z 20 sesji oraz zmienność '
        'zrealizowaną.',
        '',
        '### Analiza wrażliwości definicji',
        '',
        sens.to_markdown(index=False),
        '',
        '## 3. Schemat walidacji',
        '',
        f'Purged walk-forward, {len(folds)} foldów, embargo {pl.EMBARGO} sesji.',
        '',
        folds.to_markdown(index=False),
        '',
        '## 4. Wyniki pozapróbkowe',
        '',
        f'Porównanie na wspólnym podzbiorze {len(common)} sesji '
        f'({int(y.reindex(common).sum())} zdarzeń, '
        f'{common.min():%Y-%m-%d} – {common.max():%Y-%m-%d}).',
        '',
        cmp.reset_index().to_markdown(index=False),
        '',
        '## 5. Hybrydowy Indeks Ryzyka',
        '',
        f'- próg ostrzegawczy (kwantyl {HRI_QUANTILE} okresu referencyjnego): '
        f'{hri_res["threshold"]:.4f}',
        f'- liczba epizodów krytycznych: {len(hri_res["periods"])}',
        '',
        hri_res['periods'].to_markdown(index=False) if len(hri_res['periods']) else '(brak)',
        '',
        '### Przyczynowość Grangera',
        '',
        hri_res['granger'].reset_index().to_markdown(index=False),
        '',
        '## 6. Najważniejsze cechy (SHAP)',
        '',
        shap_imp.head(15).to_markdown(index=False),
        '',
    ]
    (BASE / 'results' / 'summary_report.md').write_text('\n'.join(lines), encoding='utf-8')


# ────────────────────────────────────────────────────────────────────
# 6. Autoenkoder i modele drzewiaste — analizy opisowe
# ────────────────────────────────────────────────────────────────────

def autoencoder_report() -> pd.DataFrame:
    """Opisowa analiza autoenkodera dopasowanego do okresu treningowego.

    Wersja opisowa służy interpretacji; w schemacie walidacyjnym autoenkoder jest
    dopasowywany osobno w każdym foldzie (models_lib.run_walk_forward).
    """
    from run_walkforward import load_modelling_frame
    from sklearn.preprocessing import StandardScaler
    _style()

    f, cols = load_modelling_frame()
    train = f.loc[:pl.TRAIN_END]
    scaler = StandardScaler().fit(train[cols].values)
    Xall = scaler.transform(f[cols].values)
    Xtr = scaler.transform(train[cols].values)
    ytr = train['target_liquidity_stress'].astype(int).values

    ae = ml.fit_autoencoder(Xtr[ytr == 0], Xtr.shape[1])
    re_train_normal = ml.reconstruction_error(ae, Xtr[ytr == 0])
    thr = float(np.quantile(re_train_normal, 0.95))
    re_all = pd.Series(ml.reconstruction_error(ae, Xall), index=f.index, name='ae_re')
    flag = (re_all > thr).astype(int)
    y = f['target_liquidity_stress'].astype(int)

    rows = []
    for year in sorted(set(f.index.year)):
        m = f.index.year == year
        anom, tgt = flag[m], y[m]
        overlap = int(((anom == 1) & (tgt == 1)).sum())
        rows.append({
            'Rok': year, 'Sesje': int(m.sum()),
            'Anomalie AE': int(anom.sum()), 'Zdarzenia stresowe': int(tgt.sum()),
            'Pokrycie': overlap,
            'Precyzja AE': round(overlap / max(int(anom.sum()), 1), 3),
            'Czułość AE': round(overlap / max(int(tgt.sum()), 1), 3)})
    out = pd.DataFrame(rows).set_index('Rok')
    out.to_csv(TABLES / '06_autoencoder_yearly.csv')

    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.plot(re_all.index, re_all.values, color='grey', lw=0.7)
    ax.axhline(thr, color=COLOR_HRI, ls='--', lw=1.0, label=f'próg anomalii = {thr:.3f}'.replace('.', ','))
    ax.scatter(re_all.index[y == 1], re_all[y == 1], s=8, color=COLOR_HRI, alpha=0.6,
               label='sesje poprzedzające suszę płynnościową')
    ax.set_ylabel('Błąd rekonstrukcji')
    ax.set_title('Błąd rekonstrukcji autoenkodera na tle zdarzeń stresowych')
    ax.legend(fontsize=8)
    plt.tight_layout()
    for ext in ('png', 'pdf'):
        plt.savefig(FIGS / f'fig_16_autoencoder_re_timeseries.{ext}', bbox_inches='tight')
    plt.close()

    pd.DataFrame([{'Próg anomalii (95. percentyl)': round(thr, 5),
                   'Liczba anomalii': int(flag.sum()),
                   'Udział anomalii (%)': round(100 * float(flag.mean()), 2),
                   'Wymiar warstwy wąskiej': 16,
                   'Liczba cech wejściowych': len(cols)}]).to_csv(
        TABLES / '06_autoencoder_config.csv', index=False)
    return out


def tree_importance_report() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Ważność cech modeli drzewiastych dopasowanych do okresu uczącego."""
    from run_walkforward import load_modelling_frame
    from sklearn.preprocessing import StandardScaler
    _style()
    f, cols = load_modelling_frame()
    train = f.loc[:pl.VAL_END]
    scaler = StandardScaler().fit(train[cols].values)
    Xtr = scaler.transform(train[cols].values)
    ytr = train['target_liquidity_stress'].astype(int).values

    out = {}
    for kind, fname in (('xgb', '07_xgb_feature_importance.csv'),
                        ('lgb', '07_lgb_feature_importance.csv')):
        model, _, _ = ml.fit_predict_trees(kind, Xtr, ytr, Xtr[:1], Xtr[:1])
        imp = pd.DataFrame({'Cecha': cols, 'Ważność': model.feature_importances_})
        imp = imp.sort_values('Ważność', ascending=False).reset_index(drop=True)
        imp.to_csv(TABLES / fname, index=False)
        out[kind] = imp

    fig, ax = plt.subplots(1, 2, figsize=(13, 6))
    for a, (kind, title) in zip(ax, (('xgb', 'XGBoost'), ('lgb', 'LightGBM'))):
        top = out[kind].head(12).iloc[::-1]
        a.barh(top['Cecha'], top['Ważność'], color=COLOR_WIG if kind == 'xgb' else COLOR_BTC)
        a.set_title(f'Ważność cech — {title}')
        a.tick_params(axis='y', labelsize=8)
    plt.tight_layout()
    for ext in ('png', 'pdf'):
        plt.savefig(FIGS / f'fig_18_tree_feature_importance.{ext}', bbox_inches='tight')
    plt.close()
    return out['xgb'], out['lgb']


def main() -> None:
    oof, y, master = load()
    print('Porównanie modeli...')
    cmp = model_comparison(oof, y)
    print(cmp.to_string())
    print('\nAnaliza HRI...')
    hri_res = hri_analysis(oof, y, master)
    print(f'  próg = {hri_res["threshold"]:.4f}, epizodów = {len(hri_res["periods"])}')
    print('\nKrzywe oceny i kalibracji...')
    calibration_and_curves(oof, y)
    print('Analiza SHAP...')
    shap_imp = shap_analysis()
    print(shap_imp.head(10).to_string(index=False))
    write_summary(cmp, hri_res, shap_imp, oof, y)
    oof.to_csv(PROC / 'oof_walkforward.csv')
    print('\nZapisano wyniki i results/summary_report.md')


if __name__ == '__main__':
    main()
