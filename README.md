# Demand Forecasting & Inventory Optimization

An end-to-end mini supply-chain analytics project:
**forecast demand -> decide how much to stock -> show it in a dashboard.**

## Problem
A retailer sells 20 products across 10 stores. Planners need to know
(1) how much each product will sell over the next 7 days and
(2) how much stock to hold so they rarely run out without tying up too much money.

## Data
Daily sales per store and product, 2022-2025 (292,200 rows): `date, store, item, sales, promo, is_holiday`.
> TODO (before publishing): state clearly whether you used the synthetic data (`01_generate_data.py`)
> or a real dataset (e.g. Kaggle "Store Item Demand Forecasting"). Never present synthetic data as real.

## Approach
1. **Baselines** - "same as last week" and 28-day moving average.
2. **Features** - calendar (day of week, month, holiday, promo) + lags (7/14/28/364 days) + rolling mean/std.
   All lags are >= 7 days, matching the 7-day horizon, so there is no data leakage.
3. **Model** - LightGBM with Poisson objective (sales are counts), early stopping on a validation window.
4. **Validation** - time-based split: train -> validation (60 days) -> test (last 60 days). No random shuffling.
5. **Metric** - WAPE = sum(|actual - forecast|) / sum(actual).
6. **Experiment tracking** - MLflow logs parameters and metrics.
7. **Inventory** - safety stock, reorder point, EOQ per store-item; simulation of two policies on test-period demand.
8. **Dashboard** - CSVs in `outputs/` feed a Power BI report (see `POWERBI_GUIDE.md`).

## Results (fill in YOUR numbers after running)
| Model | WAPE |
|---|---|
| Naive (last week) | _fill_ |
| 28-day moving average | _fill_ |
| LightGBM | _fill_ |

| Inventory policy | Stockout days | Fill rate | Avg stock value |
|---|---|---|---|
| Without safety stock | _fill_ | _fill_ | _fill_ |
| With safety stock | _fill_ | _fill_ | _fill_ |

Charts are in `outputs/`.

## Assumptions & limitations (important - be honest)
- Lead time (7 days), order cost ($50), holding cost (20%/year) and unit costs are **assumed**, not real.
- Forecast errors are assumed roughly normal when sizing safety stock.
- The simulation covers only 60 days and uses a fixed reorder point.
- Possible next steps: add price/weather data, hierarchical forecasting, probabilistic forecasts, PySpark version.

## How to run
```bash
pip install -r requirements.txt
cd src
python 01_generate_data.py   # creates data/sales.csv  (skip if using real data)
python 02_forecast.py        # baselines + model + charts + MLflow
python 03_inventory.py       # safety stock, ROP, EOQ, simulation
mlflow ui --backend-store-uri ../outputs/mlruns   # optional: view experiments
```

## Tech
Python, pandas, NumPy, LightGBM, scikit-learn, MLflow, matplotlib, Power BI.
