"""
Step 1 - Create a realistic SYNTHETIC daily sales dataset.

Why synthetic? So the project runs with no download. The real-data version
(Kaggle 'Store Item Demand Forecasting') has the same columns:
    date, store, item, sales
Swap it in by saving it as data/sales.csv with those columns
(+ promo and is_holiday, or just fill them with 0).

Patterns built in (so the model has something to learn):
  - weekly cycle (weekends sell more)
  - yearly cycle (seasonality)
  - slow upward trend
  - holidays
  - promotions (known in advance, boost sales)
  - random noise
"""
import numpy as np, pandas as pd
from config import *

rng = np.random.default_rng(SEED)
dates = pd.date_range(START, END, freq="D")

HOLIDAYS = [(1, 1), (2, 14), (7, 4), (10, 31), (11, 25), (12, 24), (12, 25), (12, 31)]
is_hol = np.array([(d.month, d.day) in HOLIDAYS for d in dates])

dow_effect = np.array([0.85, 0.85, 0.90, 0.95, 1.10, 1.30, 1.20])      # Mon..Sun
t = np.arange(len(dates))

rows = []
for store in range(1, N_STORES + 1):
    store_size = rng.uniform(0.6, 1.6)
    for item in range(1, N_ITEMS + 1):
        base = rng.uniform(8, 60)
        season_amp = rng.uniform(0.05, 0.35)
        season_phase = rng.uniform(0, 2 * np.pi)
        trend = rng.uniform(0.0, 0.25)
        promo = (rng.random(len(dates)) < 0.06).astype(int)

        level = base * store_size
        yearly = 1 + season_amp * np.sin(2 * np.pi * t / 365.25 + season_phase)
        growth = 1 + trend * t / len(dates)
        mean = level * dow_effect[dates.dayofweek] * yearly * growth
        mean = mean * (1 + 0.35 * promo) * (1 + 0.25 * is_hol)

        sales = rng.poisson(mean * rng.gamma(shape=20, scale=1 / 20, size=len(dates)))
        rows.append(pd.DataFrame({"date": dates, "store": store, "item": item,
                                  "sales": sales, "promo": promo,
                                  "is_holiday": is_hol.astype(int)}))

df = pd.concat(rows, ignore_index=True)
df.to_csv(DATA / "sales.csv", index=False)
print(f"Saved {len(df):,} rows -> data/sales.csv")
print(df.head())
