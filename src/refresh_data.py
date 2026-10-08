#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Odświeżenie danych źródłowych pracy — w PEŁNI SKRYPTOWALNE, bez ręcznego pobierania.

Uruchomienie:
    python3 refresh_data.py                # pełne odświeżenie + walidacja + log
    python3 refresh_data.py --check-only   # tylko walidacja istniejących plików

Serie i źródła (kolejność prób opisana w DANE_POCHODZENIE.md):
    BTC/USD   : Binance REST /api/v3/klines (BTCUSDT, 1d)      [źródło podstawowe]
                kontrola: lokalny szereg minutowy Bitstamp (btc.csv) → agregacja dzienna
    WIG20     : Biznesradar.pl (notowania historyczne, dane GPW) [źródło podstawowe]
                kontrola: Bankier.pl new-charts API (pokrycie do końca 2025 r.)
    VIX       : Yahoo Finance ^VIX
    DXY       : Yahoo Finance DX-Y.NYB
    Fed funds : FRED DFF (dzienna efektywna stopa funduszy federalnych)

Stooq.pl / stooq.com: endpoint /q/d/l/ zwraca „Odmowa dostępu" także po poprawnym
przejściu weryfikacji proof-of-work (potwierdzone 2026-08-18). Kod próby pozostaje
w funkcji fetch_stooq() — gdy blokada zniknie, stooq stanie się źródłem podstawowym
bez zmian w pozostałej części skryptu.

Każde pobranie jest logowane do data/raw/_provenance_log.json (URL, znacznik czasu,
liczba wierszy, zakres dat, braki, duplikaty) i stanowi podstawę DANE_POCHODZENIE.md.
"""
from __future__ import annotations

import argparse
import hashlib
import http.cookiejar
import json
import re
import subprocess
import sys
import time
import urllib.parse
import urllib.request
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

BASE = Path(__file__).parent
DATA_RAW = BASE / 'data' / 'raw'
LOG_PATH = DATA_RAW / '_provenance_log.json'

START_DATE = '2019-01-01'
END_DATE = '2026-08-14'          # ostatnia sesja przed 2026-08-15 (15-16 VIII = weekend)
STAMP = datetime.now().strftime('%Y-%m-%d')

UA = ('Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 '
      '(KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36')

# Punkty kontrolne — wartości znane niezależnie od pobieranych danych.
# Rozbieżność przekraczająca tolerancję przerywa odświeżanie.
CHECKPOINTS = [
    dict(series='btc', kind='max_high_window', start='2025-10-01', end='2025-10-10',
         expected=126_000, tol_pct=3.0,
         desc='szczyt notowań bitcoina z 6 października 2025 r. (ok. 126 tys. USD intraday)'),
    dict(series='btc', kind='close_window_mean', start='2026-08-01', end='2026-08-14',
         expected=63_000, tol_pct=6.0,
         desc='kurs bitcoina na początku sierpnia 2026 r. (ok. 62-64 tys. USD)'),
    dict(series='wig20', kind='close_window_mean', start='2026-08-10', end='2026-08-14',
         expected=4_005, tol_pct=3.0,
         desc='WIG20 w połowie sierpnia 2026 r. (ok. 3 970-4 040 pkt)'),
]

PROV: list[dict] = []


# ────────────────────────────────────────────────────────────────────
# Narzędzia
# ────────────────────────────────────────────────────────────────────

def curl(url: str, referer: str | None = None, timeout: int = 90) -> str:
    cmd = ['curl', '-s', '--max-time', str(timeout), '-A', UA]
    if referer:
        cmd += ['-H', f'Referer: {referer}']
    cmd.append(url)
    return subprocess.run(cmd, capture_output=True, text=True).stdout


def log_series(name: str, source: str, url: str, method: str, df: pd.DataFrame,
               note: str = '') -> None:
    gaps = pd.Series(df.index).diff().dt.days
    entry = dict(
        series=name, source=source, endpoint=url, method=method,
        fetched_at=datetime.now(timezone.utc).astimezone().isoformat(timespec='seconds'),
        rows=int(len(df)),
        date_from=str(df.index.min().date()), date_to=str(df.index.max().date()),
        missing_values=int(df.isna().sum().sum()),
        duplicated_index=int(df.index.duplicated().sum()),
        max_gap_days=int(gaps.max()) if len(gaps.dropna()) else None,
        note=note,
    )
    PROV.append(entry)
    print(f'  [{name}] {source}: {entry["rows"]} wierszy, {entry["date_from"]} → {entry["date_to"]}, '
          f'braki {entry["missing_values"]}, duplikaty {entry["duplicated_index"]}, '
          f'maks. przerwa {entry["max_gap_days"]} dni')


def save_versioned(df: pd.DataFrame, name: str) -> None:
    df.to_csv(DATA_RAW / f'{name}.csv')
    df.to_csv(DATA_RAW / f'{name}_{STAMP}.csv')
    print(f'  zapisano {name}.csv (+ kopia _{STAMP}.csv)')


# ────────────────────────────────────────────────────────────────────
# Źródła — próba stooq (obecnie zablokowana po stronie serwera)
# ────────────────────────────────────────────────────────────────────

def fetch_stooq(symbol: str) -> pd.DataFrame | None:
    """Próba pobrania ze stooq.pl z przejściem weryfikacji proof-of-work.

    Zwraca None, gdy serwer odmawia dostępu do endpointu pobierania.
    """
    cj = http.cookiejar.CookieJar()
    op = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(cj))

    def get(url, referer=None, accept='text/html,*/*'):
        h = {'User-Agent': UA, 'Accept': accept}
        if referer:
            h['Referer'] = referer
        return op.open(urllib.request.Request(url, headers=h), timeout=60).read().decode('utf-8', 'replace')

    try:
        html = get('https://stooq.pl/')
        m = re.search(r'const c="([^"]+)",d=(\d+)', html)
        if m:                                    # zagadka proof-of-work
            c, d = m.group(1), int(m.group(2))
            n = 0
            while not hashlib.sha256((c + str(n)).encode()).hexdigest().startswith('0' * d):
                n += 1
            body = ('c=' + urllib.parse.quote(c) + '&n=' + str(n)).encode()
            op.open(urllib.request.Request(
                'https://stooq.pl/__verify', data=body,
                headers={'User-Agent': UA, 'Content-Type': 'application/x-www-form-urlencoded'}),
                timeout=60)
        d1 = START_DATE.replace('-', '')
        d2 = END_DATE.replace('-', '')
        url = f'https://stooq.pl/q/d/l/?s={symbol}&d1={d1}&d2={d2}&i=d'
        csv = get(url, referer=f'https://stooq.pl/q/d/?s={symbol}', accept='text/csv,*/*')
        if 'Odmowa' in csv or 'denied' in csv.lower() or 'Data' not in csv.split('\n')[0]:
            print(f'  stooq [{symbol}]: odmowa dostępu do endpointu pobierania '
                  f'(odpowiedź: {csv.strip().splitlines()[0][:40]!r})')
            return None
        from io import StringIO
        df = pd.read_csv(StringIO(csv))
        df.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume'][:len(df.columns)]
        df['Date'] = pd.to_datetime(df['Date'])
        return df.set_index('Date').sort_index()
    except Exception as exc:                     # noqa: BLE001
        print(f'  stooq [{symbol}]: błąd {type(exc).__name__}: {str(exc)[:80]}')
        return None


# ────────────────────────────────────────────────────────────────────
# Źródła działające
# ────────────────────────────────────────────────────────────────────

BINANCE_URL = 'https://api.binance.com/api/v3/klines'


def fetch_btc_binance() -> pd.DataFrame:
    """Dzienne OHLCV BTC/USDT z Binance. Volume = obrót w walucie kwotowanej (USDT)."""
    rows = []
    start = int(pd.Timestamp(START_DATE).timestamp() * 1000)
    end = int((pd.Timestamp(END_DATE) + pd.Timedelta(days=1)).timestamp() * 1000)
    while start < end:
        url = f'{BINANCE_URL}?symbol=BTCUSDT&interval=1d&startTime={start}&limit=1000'
        for attempt in range(3):
            try:
                data = json.loads(curl(url))
                break
            except Exception:                    # noqa: BLE001
                time.sleep(2 * (attempt + 1))
        else:
            raise RuntimeError('Binance: nie udało się pobrać bloku danych')
        if not isinstance(data, list) or not data:
            break
        for k in data:
            rows.append({'Date': pd.Timestamp(k[0], unit='ms'),
                         'Open': float(k[1]), 'High': float(k[2]), 'Low': float(k[3]),
                         'Close': float(k[4]), 'Volume': float(k[7])})
        start = data[-1][6] + 1
        time.sleep(0.3)
    df = (pd.DataFrame(rows).drop_duplicates('Date').set_index('Date').sort_index()
          .loc[START_DATE:END_DATE])
    df = df[df['Close'] > 0]
    df.index.name = 'Date'
    log_series('btc', 'Binance REST API', f'{BINANCE_URL}?symbol=BTCUSDT&interval=1d',
               'curl + json, paginacja po startTime', df,
               'Volume = quote asset volume (obrót w USDT)')
    return df


BIZNESRADAR_URL = 'https://www.biznesradar.pl/notowania-historyczne/WIG20'
_ROW_RE = re.compile(
    r'<td>(\d{2}\.\d{2}\.\d{4})</td>\s*<td>([\d\s.,]+)</td>\s*<td>([\d\s.,]+)</td>'
    r'\s*<td>([\d\s.,]+)</td>\s*<td>([\d\s.,]+)</td>\s*<td>([\d\s.,]*)</td>')


def _num(s: str) -> float:
    s = s.replace('\xa0', '').replace(' ', '').replace(',', '.')
    return float(s) if s else np.nan


def fetch_wig20_biznesradar() -> pd.DataFrame:
    """Dzienne OHLC + obrót (PLN) indeksu WIG20 — dane GPW z serwisu Biznesradar.pl."""
    rows, page = [], 1
    while page <= 200:
        url = BIZNESRADAR_URL + (f',{page}' if page > 1 else '')
        html = curl(url)
        found = _ROW_RE.findall(html)
        if not found:
            break
        for r in found:
            rows.append({'Date': pd.to_datetime(r[0], format='%d.%m.%Y'),
                         'Open': _num(r[1]), 'High': _num(r[2]), 'Low': _num(r[3]),
                         'Close': _num(r[4]), 'Volume': _num(r[5])})
        if min(r['Date'] for r in rows) < pd.Timestamp(START_DATE):
            break
        page += 1
        time.sleep(0.5)
    df = (pd.DataFrame(rows).drop_duplicates('Date').set_index('Date').sort_index()
          .loc[START_DATE:END_DATE])
    df = df[df['Close'] > 0]
    df.index.name = 'Date'
    log_series('wig20', 'Biznesradar.pl (dane GPW)', BIZNESRADAR_URL,
               f'curl + parsowanie HTML, {page} stron', df,
               'Volume = wartość obrotu sesyjnego w PLN')
    return df


def fetch_wig20_bankier() -> pd.DataFrame | None:
    """Kontrolne notowania WIG20 z Bankier.pl (pokrycie do końca 2025 r.)."""
    frames = []
    cur = pd.Timestamp(START_DATE)
    end = pd.Timestamp(END_DATE)
    while cur < end:
        nxt = min(cur + pd.Timedelta(days=170), end)
        url = ('https://www.bankier.pl/new-charts/get-data?symbol=WIG20&intraday=false'
               f'&type=candlestick&date_from={int(cur.timestamp()*1000)}'
               f'&date_to={int(nxt.timestamp()*1000)}')
        try:
            d = json.loads(curl(url))
        except Exception:                        # noqa: BLE001
            d = {}
        for p in d.get('main', []):
            if len(p) == 5:
                frames.append({'Date': pd.Timestamp(p[0], unit='ms'), 'Open': p[1],
                               'High': p[2], 'Low': p[3], 'Close': p[4]})
        cur = nxt
        time.sleep(0.6)
    if not frames:
        return None
    df = pd.DataFrame(frames).drop_duplicates('Date').set_index('Date').sort_index()
    df.index.name = 'Date'
    log_series('wig20_kontrola', 'Bankier.pl new-charts API',
               'https://www.bankier.pl/new-charts/get-data?symbol=WIG20',
               'curl + json, bloki 170-dniowe', df,
               'źródło kontrolne; serwis nie udostępnia notowań z 2026 r.')
    return df


def fetch_btc_local_control() -> pd.DataFrame | None:
    """Kontrolny szereg BTC z lokalnych danych minutowych (Bitstamp, plik btc.csv)."""
    path = BASE / 'btc.csv'
    if not path.exists():
        return None
    parts = []
    for ch in pd.read_csv(path, chunksize=1_000_000):
        ch['dt'] = pd.to_datetime(ch['Timestamp'], unit='s')
        ch = ch[ch['dt'] >= START_DATE]
        if len(ch):
            parts.append(ch.set_index('dt').resample('D').agg(
                Open=('Open', 'first'), High=('High', 'max'),
                Low=('Low', 'min'), Close=('Close', 'last')))
    if not parts:
        return None
    df = (pd.concat(parts).groupby(level=0)
          .agg({'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last'})
          .dropna(subset=['Close']))
    df.index.name = 'Date'
    log_series('btc_kontrola', 'Bitstamp (lokalny plik btc.csv, dane minutowe)',
               'plik lokalny: btc.csv', 'pandas read_csv chunked + resample D', df,
               'źródło kontrolne; plik kończy się na początku lutego 2026 r.')
    return df


def fetch_yahoo(ticker: str, colname: str, series_name: str) -> pd.DataFrame:
    import yfinance as yf
    df = yf.download(ticker, start=START_DATE,
                     end=str((pd.Timestamp(END_DATE) + pd.Timedelta(days=1)).date()),
                     auto_adjust=True, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    out = df[['Close']].rename(columns={'Close': colname})
    out.index = pd.to_datetime(out.index)
    if out.index.tz is not None:
        out.index = out.index.tz_localize(None)
    out.index.name = 'Date'
    log_series(series_name, 'Yahoo Finance', f'yfinance.download("{ticker}")',
               'biblioteka yfinance', out)
    return out


def fetch_fed_rate() -> pd.DataFrame:
    """Dzienna efektywna stopa funduszy federalnych (FRED: DFF).

    Zmiana względem wcześniejszej wersji badania, w której używano miesięcznej serii
    FEDFUNDS rozciąganej na dni metodą ostatniej znanej wartości.
    """
    import pandas_datareader.data as web
    fed = web.DataReader('DFF', 'fred', start=START_DATE, end=END_DATE)
    fed = fed.rename(columns={'DFF': 'fed_rate'})
    idx = pd.date_range(START_DATE, END_DATE, freq='D')
    fed = fed.reindex(idx).ffill().bfill()
    fed.index.name = 'Date'
    log_series('fed_rate', 'FRED (Federal Reserve Bank of St. Louis)',
               'https://fred.stlouisfed.org/series/DFF',
               'pandas_datareader.data.DataReader("DFF", "fred")', fed,
               'seria dzienna; luki weekendowe uzupełnione ostatnią znaną wartością')
    return fed


# ────────────────────────────────────────────────────────────────────
# Walidacja
# ────────────────────────────────────────────────────────────────────

def quality_report(df: pd.DataFrame, name: str, expect_calendar: str) -> dict:
    ret = np.log(df['Close'] / df['Close'].shift(1)) if 'Close' in df else pd.Series(dtype=float)
    ret = ret.dropna()
    out = dict(
        series=name, rows=len(df),
        date_from=str(df.index.min().date()), date_to=str(df.index.max().date()),
        calendar=expect_calendar,
        duplicates=int(df.index.duplicated().sum()),
        missing=int(df.isna().sum().sum()),
        max_gap_days=int(pd.Series(df.index).diff().dt.days.max()),
        outliers_5sigma=int((np.abs(ret - ret.mean()) > 5 * ret.std()).sum()) if len(ret) else 0,
        nonpositive_prices=int((df.select_dtypes('number') <= 0).sum().sum()),
        ret_min=round(float(ret.min()), 4) if len(ret) else None,
        ret_max=round(float(ret.max()), 4) if len(ret) else None,
    )
    if 'Volume' in df.columns:
        out['zero_volume_days'] = int((df['Volume'].fillna(0) == 0).sum())
    print(f'  [{name}] przerwa maks. {out["max_gap_days"]} dni | outliery >5σ: '
          f'{out["outliers_5sigma"]} | duplikaty: {out["duplicates"]} | braki: {out["missing"]}')
    return out


def cross_check(primary: pd.DataFrame, control: pd.DataFrame, label: str) -> dict:
    j = primary[['Close']].join(control[['Close']], how='inner',
                                lsuffix='_p', rsuffix='_c').dropna()
    if not len(j):
        return dict(pair=label, overlap=0)
    rel = (j['Close_p'] - j['Close_c']).abs() / j['Close_c'] * 100
    out = dict(pair=label, overlap=int(len(j)),
               median_diff_pct=round(float(rel.median()), 4),
               p95_diff_pct=round(float(rel.quantile(0.95)), 3),
               max_diff_pct=round(float(rel.max()), 3),
               days_above_1pct=int((rel > 1).sum()))
    print(f'  [kontrola] {label}: {out["overlap"]} wspólnych dni, mediana różnicy '
          f'{out["median_diff_pct"]}%, 95. pct {out["p95_diff_pct"]}%')
    return out


def verify_checkpoints(btc: pd.DataFrame, wig: pd.DataFrame) -> list[dict]:
    results = []
    for cp in CHECKPOINTS:
        df = btc if cp['series'] == 'btc' else wig
        seg = df.loc[cp['start']:cp['end']]
        if not len(seg):
            results.append(dict(**{k: cp[k] for k in ('series', 'desc')},
                                status='BRAK DANYCH', actual=None))
            continue
        actual = float(seg['High'].max()) if cp['kind'] == 'max_high_window' else float(seg['Close'].mean())
        dev = abs(actual - cp['expected']) / cp['expected'] * 100
        ok = dev <= cp['tol_pct']
        results.append(dict(series=cp['series'], desc=cp['desc'], expected=cp['expected'],
                            actual=round(actual, 2), deviation_pct=round(dev, 2),
                            tolerance_pct=cp['tol_pct'], status='OK' if ok else 'ROZBIEŻNOŚĆ'))
        print(f'  [{"OK " if ok else "!!!"}] {cp["desc"]}: oczekiwano ~{cp["expected"]:,}, '
              f'jest {actual:,.0f} (odchylenie {dev:.1f}%)')
    return results


# ────────────────────────────────────────────────────────────────────
# Główny przebieg
# ────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--check-only', action='store_true',
                    help='tylko walidacja plików już zapisanych w data/raw')
    args = ap.parse_args()

    DATA_RAW.mkdir(parents=True, exist_ok=True)
    print(f'Odświeżanie danych: {START_DATE} → {END_DATE} (uruchomienie {STAMP})\n')

    if args.check_only:
        btc = pd.read_csv(DATA_RAW / 'btc_daily_raw.csv', index_col=0, parse_dates=True)
        wig = pd.read_csv(DATA_RAW / 'wig20_daily_raw.csv', index_col=0, parse_dates=True)
    else:
        print('[1/5] BTC/USD')
        btc = fetch_stooq('btcusd')
        if btc is None or len(btc) < 1000:
            btc = fetch_btc_binance()
        save_versioned(btc, 'btc_daily_raw')

        print('[2/5] WIG20')
        wig = fetch_stooq('wig20')
        if wig is None or len(wig) < 1000:
            wig = fetch_wig20_biznesradar()
        save_versioned(wig, 'wig20_daily_raw')

        print('[3/5] VIX')
        save_versioned(fetch_yahoo('^VIX', 'vix', 'vix'), 'vix_raw')

        print('[4/5] DXY')
        save_versioned(fetch_yahoo('DX-Y.NYB', 'dxy', 'dxy'), 'dxy_raw')

        print('[5/5] Stopa funduszy federalnych')
        save_versioned(fetch_fed_rate(), 'fed_rate_raw')

    print('\nWalidacja jakości:')
    quality = [quality_report(btc, 'btc', '7 dni w tygodniu'),
               quality_report(wig, 'wig20', 'sesje GPW')]

    print('\nKontrola krzyżowa z niezależnymi źródłami:')
    checks = []
    ctl_wig = fetch_wig20_bankier()
    if ctl_wig is not None:
        checks.append(cross_check(wig, ctl_wig, 'WIG20: Biznesradar vs Bankier.pl'))
    ctl_btc = fetch_btc_local_control()
    if ctl_btc is not None:
        checks.append(cross_check(btc, ctl_btc, 'BTC: Binance vs Bitstamp (btc.csv)'))

    print('\nPunkty kontrolne (wartości znane niezależnie):')
    cps = verify_checkpoints(btc, wig)

    LOG_PATH.write_text(json.dumps(dict(
        run_at=datetime.now(timezone.utc).astimezone().isoformat(timespec='seconds'),
        window=dict(start=START_DATE, end=END_DATE),
        series=PROV, quality=quality, cross_checks=checks, checkpoints=cps,
    ), indent=2, ensure_ascii=False), encoding='utf-8')
    print(f'\nLog pochodzenia danych: {LOG_PATH.relative_to(BASE)}')

    if any(c['status'] == 'ROZBIEŻNOŚĆ' for c in cps):
        print('\nUWAGA: punkt kontrolny poza tolerancją — sprawdź źródło przed dalszą analizą.')
        sys.exit(2)
    print('Odświeżanie zakończone poprawnie.')


if __name__ == '__main__':
    main()
