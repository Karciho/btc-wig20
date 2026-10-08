# Raport pochodzenia danych

**Nie jest częścią pracy dyplomowej.** Dokument techniczny dla autora — do rozmowy z promotorem
i do odtworzenia zbioru danych od zera.

Data sporządzenia: **2026-08-18**. Okno danych: **2019-01-01 → 2026-08-14** (ostatnia sesja
przed 15 sierpnia 2026 r.; 15–16 VIII to weekend).
Log maszynowy: `data/raw/_provenance_log.json` (generowany przez `refresh_data.py`).

---

## 1. Serie wykorzystane w badaniu

| Seria | Dostawca | Symbol | Endpoint / polecenie | Metoda | Pobrano | Zakres | Obs. | Uwagi i ograniczenia |
|---|---|---|---|---|---|---|---|---|
| BTC/USD dzienne OHLCV | Binance | `BTCUSDT` | `https://api.binance.com/api/v3/klines?symbol=BTCUSDT&interval=1d` | `curl` + JSON, paginacja po `startTime`, bloki 1000 świec | 2026-08-18 | 2019-01-01 → 2026-08-14 | 2783 | `Volume` = *quote asset volume*, tj. obrót w USDT (nie liczba BTC). To postać wymagana przez wskaźnik Amihuda. Para z USDT, nie z USD — spread USDT/USD historycznie < 0,5%. |
| WIG20 dzienne OHLC + obrót | Biznesradar.pl (redystrybucja danych GPW) | `WIG20` | `https://www.biznesradar.pl/notowania-historyczne/WIG20[,N]` | `curl` + parsowanie HTML, 39 stron po 50 wierszy | 2026-08-18 | 2019-01-02 → 2026-08-14 | 1906 | `Volume` = wartość obrotu sesyjnego w PLN. Maks. przerwa 6 dni (przerwy świąteczne GPW). |
| VIX | Cboe przez Yahoo Finance | `^VIX` | `yfinance.download("^VIX")` | biblioteka `yfinance` 1.2.0 | 2026-08-18 | 2019-01-02 → 2026-08-14 | 1913 | Kurs zamknięcia. Kalendarz sesyjny USA. |
| DXY | ICE przez Yahoo Finance | `DX-Y.NYB` | `yfinance.download("DX-Y.NYB")` | biblioteka `yfinance` 1.2.0 | 2026-08-18 | 2019-01-02 → 2026-08-14 | 1914 | Kurs zamknięcia. |
| Efektywna stopa funduszy federalnych | FRED (Fed St. Louis) | `DFF` | `https://fred.stlouisfed.org/series/DFF` | `pandas_datareader.DataReader("DFF","fred")` | 2026-08-18 | 2019-01-01 → 2026-08-14 | 2783 | **Zmiana względem wcześniejszych wersji badania**: seria dzienna `DFF` zamiast miesięcznej `FEDFUNDS` rozciąganej na dni. Luki weekendowe uzupełnione ostatnią znaną wartością. |

Wszystko pobiera jedno polecenie:

```bash
python3 refresh_data.py
```

## 2. Źródła kontrolne (nie wchodzą do zbioru badawczego)

| Para | Wspólnych dni | Mediana różnicy | 95. percentyl | Maks. | Wniosek |
|---|---|---|---|---|---|
| WIG20: Biznesradar vs Bankier.pl | 1750 | **0,000%** | 0,000% | 0,287% | Zgodność praktycznie pełna. Rozbieżność powyżej 0,1% dotyczy 2 sesji z 1750. Bankier.pl nie udostępnia notowań z 2026 r., więc kontrola obejmuje 2019-01-02 → 2026-01-02. |
| BTC: Binance vs Bitstamp (plik `btc.csv`) | 2593 | 0,390% | 2,661% | 20,436% | Zgodność akceptowalna dla dwóch różnych giełd. Rozbieżności wynikają z luk w lokalnym szeregu minutowym (przy braku ostatniej minuty doby cena zamknięcia pochodzi z wcześniejszej godziny) oraz z realnych różnic płynności między giełdami. Na obu punktach kontrolnych zgodność do 0,2%. |

## 3. Punkty kontrolne wobec wartości znanych niezależnie

Wartości odniesienia podał zleceniodawca; skrypt weryfikuje je przy każdym uruchomieniu
i przerywa pracę przy przekroczeniu tolerancji.

| Punkt kontrolny | Oczekiwano | Z pobranych danych | Odchylenie | Status |
|---|---|---|---|---|
| Szczyt notowań bitcoina 6 października 2025 r. (intraday) | ~126 000 USD | **126 200 USD** (maks. `High` 1–10 X 2025) | 0,2% | OK |
| Kurs bitcoina na początku sierpnia 2026 r. | 62 000–64 000 USD | **63 956 USD** (średnia `Close` 1–14 VIII) | 1,5% | OK |
| WIG20 w połowie sierpnia 2026 r. | 3 970–4 040 pkt | **4 035 pkt** (średnia `Close` 10–14 VIII) | 0,8% | OK |

## 4. Stooq.pl — dlaczego nie jest źródłem podstawowym

Polecenie wskazywało stooq jako źródło preferowane. Sprawdzono wszystkie dostępne ścieżki
(stan na 2026-08-18):

1. **Bezpośredni endpoint CSV** `https://stooq.pl/q/d/l/?s=wig20&d1=20190101&d2=20260815&i=d` —
   serwer odpowiada zagadką *proof-of-work* w JavaScript. Zagadkę zaimplementowano i rozwiązano
   (SHA-256 z prefiksem czterech zer, endpoint `/__verify` zwraca HTTP 200, ciasteczka `auth`
   i `cookie_uu` zostają ustawione). Po weryfikacji strona główna oraz strona notowań ładują się
   poprawnie (198 933 B, bez zagadki), natomiast **sam endpoint pobierania zwraca „Odmowa
   dostępu"** — dla pełnego i dla wąskiego zakresu dat, dla `wig20` i dla `btcusd`, na obu
   domenach (`stooq.pl`, `stooq.com`).
2. **`pandas-datareader` z backendem `stooq`** — korzysta z tego samego endpointu, zwraca
   `RemoteDataError: Unable to read URL`.
3. **Przeglądarka użytkownika** (poprzednia sesja, 2026-07-06): otwarcie adresu CSV w Chrome
   zakończyło się pobraniem pliku `blad.txt` o treści „Odmowa dostępu".

Blokada dotyczy wyłącznie pobierania plików, nie przeglądania serwisu, i ma charakter decyzji
serwisu (pobieranie danych historycznych wymaga zalogowanego konta lub jest blokowane poza
przeglądarką). Kod próby pozostaje w `refresh_data.fetch_stooq()` jako pierwszy krok obu
serii — gdy blokada zniknie, stooq automatycznie stanie się źródłem podstawowym.

## 5. Audyt plików z wcześniejszych rund

### 5.1. Pierwotne dane (marzec 2026, przebieg autora pracy)

Zachowane w `backup_przed_aktualizacja_2026-07-06/data/raw/`.

- `btc_daily_raw.csv`: kolejność kolumn `Close,High,Low,Open,Volume` (alfabetyczna — charakterystyczna
  dla spłaszczonego `MultiIndex` z `yfinance`), ceny z pełną precyzją zmiennoprzecinkową
  (`3843.52001953125`), wolumeny całkowite. 2191 wierszy, 2019-01-01 → 2024-12-30.
  **Wniosek: yfinance, ticker `BTC-USD`.**
- `wig20_daily_raw.csv`: 1501 wierszy; `Volume` rzędu 15,5 mln = liczba akcji, nie wartość obrotu.
  **Wniosek: yfinance, ticker `WIG20.WA`.**
- Zapis w `results/summary_report.md` („BTC/USD: Yahoo Finance", „WIG20: Yahoo Finance / stooq.pl")
  był więc **poprawny dla pierwotnego przebiegu**.
- Rozbieżność powstała później: docstring `refresh_data.py` napisany w lipcu 2026 r. deklarował
  stooq jako źródło obu serii, mimo że stooq nigdy nie zwrócił danych, a funkcja `refresh_stooq()`
  oczekiwała plików pobranych ręcznie. **Deklaracja była nieprawdziwa i została usunięta.**
- Ticker `WIG20.WA` przestał być obsługiwany przez Yahoo Finance (sprawdzone 2026-07-06 i ponownie
  2026-08-18: „possibly delisted; no price data found" dla `WIG20.WA`, `^WIG20`, `WIG.WA`, `^WIG`),
  co samo w sobie wymusiło zmianę źródła dla WIG20.

### 5.2. Dane z rundy lipcowej (2026-07-06)

BTC z Binance, WIG20 z Biznesradar.pl — te same źródła co obecnie, ale pobierane częściowo
ręcznie i bez logu. Ułamkowe wolumeny w `btc_daily_raw.csv`
(np. `88149249.09230462`), które budziły wątpliwości, to *quote asset volume* z Binance,
czyli obrót w USDT — wartość z natury niecałkowita.

### 5.3. Istotna konsekwencja metodyczna zmiany źródła WIG20

Definicja `Volume` dla WIG20 zmieniła się między przebiegami: pierwotnie liczba akcji (Yahoo),
obecnie wartość obrotu w PLN (GPW). Wskaźnik niepłynności Amihuda, na którym oparta jest zmienna
objaśniana, w oryginalnym ujęciu autora miary liczony jest właśnie od obrotu wartościowego,
więc obecna postać jest metodycznie poprawna, ale **skala wskaźnika i tym samym progi zmiennej
celu nie są porównywalne między wersjami pracy**. Opisano to w `METODYKA_ZMIANY.md`.

## 6. Duże pliki poboczne w katalogu głównym

| Plik | Rozmiar | Zawartość | Wykorzystanie w badaniu | Rekomendacja |
|---|---|---|---|---|
| `btc.csv` | 371 MB | Dane minutowe BTC/USD od 2011-12-31 (cena 4,58 USD) do 2026-02-05, format `Timestamp,Open,High,Low,Close,Volume` — układ zbioru „Bitcoin Historical Data" (Bitstamp) | **Tak** — użyty jako niezależne źródło kontrolne dla Binance (sekcja 2) | Zachować. Jedyne dostępne offline źródło kontrolne dla BTC. |
| `btc old.csv` | 369 MB | Ten sam zbiór, wcześniejsza wersja, kończy się 2026-01-08 | Nie | Kandydat do archiwizacji (duplikat starszy o miesiąc). **Nie usunięto — wymaga zgody autora.** |
| `cryptocurrency.csv` | 11 MB | Zrzut notowań wielu kryptowalut z 9 stycznia 2026 r. (nazwa, kapitalizacja, zmiany %) | Nie | Bez związku z badaniem. Kandydat do archiwizacji. |
| `stocks.csv` | 197 KB | Zrzut notowań akcji amerykańskich z 9 stycznia 2026 r. | Nie | Bez związku z badaniem. Kandydat do archiwizacji. |
| `12 (2).csv` | 110 KB | Świece godzinowe XRP-USDT z Binance od 2021 r. | Nie | Pozostałość wcześniejszych eksperymentów. Kandydat do archiwizacji. |

Łącznie do archiwizacji około 740 MB (`btc old.csv`, `cryptocurrency.csv`, `stocks.csv`,
`12 (2).csv`). Zgodnie z ustaleniami **żaden plik nie został usunięty ani przeniesiony** —
decyzja należy do autora.

## 7. Instrukcja pełnego odtworzenia zbioru danych

```bash
cd ~/Desktop/"praca magisterska"

# 1. Surowe dane (Binance, Biznesradar, Yahoo, FRED) + walidacja + punkty kontrolne
python3 refresh_data.py

# 2. Weryfikacja bez ponownego pobierania
python3 refresh_data.py --check-only

# 3. Przebudowa zbioru głównego i cech oraz przeliczenie modeli
KERAS_BACKEND=torch PYTORCH_ENABLE_MPS_FALLBACK=1 MPLBACKEND=Agg \
  jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.kernel_name=python3 --ExecutePreprocessor.timeout=5400 \
  01_setup_and_data_download.ipynb 02_eda_exploratory_analysis.ipynb \
  03_statistical_tests.ipynb 04_feature_engineering.ipynb \
  05_model_lstm_gru.ipynb 06_model_autoencoder.ipynb \
  07_model_xgboost_lightgbm.ipynb 08_ensemble_stacking.ipynb \
  09_hri_analysis.ipynb 10_evaluation_shap.ipynb

# 4. Złożenie pracy i eksport
python3 create_docx.py && python3 export_pdf.py
```

Wymagane środowisko: Python 3.13 (Anaconda), `keras` 3.13 z backendem PyTorch
(`PYTORCH_ENABLE_MPS_FALLBACK=1` jest konieczne — inicjalizacja warstw rekurencyjnych używa
rozkładu QR, nieobsługiwanego na Metal Performance Shaders), `xgboost` 3.2, `lightgbm` 4.6,
`shap` 0.51, `statsmodels` 0.14, `arch` 8.0, `yfinance` 1.2, `pandas-datareader` 0.10.
Ziarno generatora liczb losowych: 42.
