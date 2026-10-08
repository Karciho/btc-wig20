# Bitcoin market stress as a leading signal of liquidity stress on the Warsaw Stock Exchange

Code and results from my master's thesis in Economic Analytics at the Cracow University of Economics (defended September 2026).

The question behind the project: does information from the Bitcoin market arrive before liquidity dries up on the Polish equity market? The short answer is yes, but a correlation will not find it.

## The main result

Raw Bitcoin returns Granger-cause WIG20 liquidity stress at a few lags, until the Benjamini-Hochberg correction for multiple testing is applied. After the correction nothing survives: 0 of 10 lags.

The same information, combined with liquidity, volatility and global financing conditions and aggregated non-linearly into one index, keeps 8 of 10 lags significant after the same correction.

| Series | Significant lags before correction | After Benjamini-Hochberg |
|---|---|---|
| Raw BTC returns | 3 of 10 | 0 of 10 |
| Non-linear index, 52 features | 5 of 10 | 0 of 10 |
| Non-linear index, 34 crypto and macro features | 9 of 10 | 8 of 10 |

Out of sample, on 748 sessions where liquidity droughts cover 2.4 percent of days:

| Model | AUC-PR | AUC-ROC |
|---|---|---|
| XGBoost | 0.437 | 0.861 |
| LightGBM | 0.437 | 0.826 |
| Ensemble, non-negative stacking | 0.396 | 0.809 |
| LSTM | 0.256 | 0.725 |
| GRU | 0.206 | 0.814 |

An AUC-PR of 0.437 against a base rate of 0.024 is an 18x lift. The bootstrap interval on 5,000 resamples runs from 0.21 to 0.66.

## Method in short

- **Data**: 1,906 common trading sessions, January 2019 to August 2026. BTC/USD from Binance, WIG20 from the Warsaw Stock Exchange, VIX, DXY and the federal funds rate. Every series was cross-checked against an independent source.
- **Target**: a liquidity drought is a five-session average of a composite stress index above its 90th training-set percentile. The index combines Amihud illiquidity, the Corwin-Schultz spread estimator, a turnover shortfall measure and realised volatility. Positive class: 6.98 percent in sample, 2.26 percent out of sample.
- **Validation**: purged walk-forward with 10 folds and a 10-session embargo, following Lopez de Prado, so no information from the future leaks backwards.
- **Models**: LSTM and GRU, an autoencoder for unsupervised anomaly detection, XGBoost and LightGBM, and a stacking layer with non-negative weights fitted by NNLS.
- **Robustness**: 15 random seeds, 5,000 bootstrap resamples, an ablation of the autoencoder block, longer embargo windows, and a sensitivity analysis of the target definition.
- **Interpretation**: SHAP values on the tree models.

## What I do not claim

The relationship runs in both directions. Reverse-direction Granger causality holds at all 10 lags, which points to coupled markets rather than Bitcoin leading Warsaw. The event study rests on five episodes, so it says little about the size of the price reaction. The index is built to monitor vulnerability, not to forecast prices.

## Repository layout

```
notebooks/   10 Jupyter notebooks, from data download to SHAP evaluation
src/         reusable modules and the scripts that produce the results
results/     40 result tables (CSV) and 43 figures (PNG)
docs/        notes on data provenance and quality checks
```

## Running the code

```bash
pip install -r requirements.txt
python src/refresh_data.py     # downloads the market data
python src/run_analysis.py     # full pipeline: features, models, evaluation
```

A full run takes a few hours on a laptop, mostly in the recurrent models.

Market data is not included in this repository. Binance and the data provider for WIG20 have their own terms, so the download script is here instead of the files.

## Author

Tsikhan Karoukin, Cracow. MSc in Economic Analytics.
Portfolio of data analyses: https://rpubs.com/Karciho

## Licence

MIT, see LICENSE.
