"""
Step 2 - Demand forecasting.

Goal: predict daily sales 7 days into the future for every store-item.

Key ideas
  * Baselines first (naive + moving average) so we can say "we beat X by Y%".
  * Features: calendar + lags + rolling averages (all shifted >= 7 days,
    so we never use information from the future = no data leakage).
  * Time-based split: train (oldest) -> validation -> test (newest).
  * Metric: WAPE = sum(|actual - forecast|) / sum(actual). Lower is better.
  * LightGBM model (falls back to scikit-learn's similar model if not installed).
  * MLflow tracking (skipped automatically if not installed).
"""
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from config import *

try:
    import lightgbm as lgb
    USE_LGBM = True
except ImportError:
    from sklearn.ensemble import HistGradientBoostingRegressor
    USE_LGBM = False
    print("NOTE: lightgbm not installed -> using sklearn HistGradientBoostingRegressor instead.")
try:
    import mlflow
    USE_MLFLOW = True
except ImportError:
    USE_MLFLOW = False
    print("NOTE: mlflow not installed -> skipping experiment tracking.")


def wape(actual, pred):
    return np.abs(actual - pred).sum() / actual.sum()


# ---------- 1. load ----------
df = pd.read_csv(DATA / "sales.csv", parse_dates=["date"])
df = df.sort_values(["store", "item", "date"]).reset_index(drop=True)

# ---------- 2. features ----------
g = df.groupby(["store", "item"])["sales"]
for lag in (7, 14, 28, 364):                      # all >= HORIZON -> no leakage
    df[f"lag_{lag}"] = g.shift(lag)
for w in (7, 28):
    df[f"roll_mean_{w}"] = g.transform(lambda s: s.shift(HORIZON).rolling(w).mean())
df["roll_std_28"] = g.transform(lambda s: s.shift(HORIZON).rolling(28).std())

df["dow"] = df["date"].dt.dayofweek
df["month"] = df["date"].dt.month
df["dayofyear"] = df["date"].dt.dayofyear
df["weekofyear"] = df["date"].dt.isocalendar().week.astype(int)
df["is_weekend"] = (df["dow"] >= 5).astype(int)

FEATURES = ["store", "item", "promo", "is_holiday", "dow", "month", "dayofyear",
            "weekofyear", "is_weekend", "lag_7", "lag_14", "lag_28", "lag_364",
            "roll_mean_7", "roll_mean_28", "roll_std_28"]

# ---------- 3. time-based split ----------
last = df["date"].max()
test_start = last - pd.Timedelta(days=TEST_DAYS - 1)
valid_start = test_start - pd.Timedelta(days=VALID_DAYS)
train = df[(df["date"] < valid_start) & df["lag_364"].notna()]
valid = df[(df["date"] >= valid_start) & (df["date"] < test_start)]
test = df[df["date"] >= test_start].copy()
print(f"Train: {len(train):,} rows | Valid: {len(valid):,} | Test: {len(test):,}")
print(f"Test period: {test_start.date()} -> {last.date()}")

# ---------- 4. baselines ----------
test["pred_naive"] = test["lag_7"]
test["pred_ma28"] = test["roll_mean_28"]
b_naive = wape(test["sales"], test["pred_naive"])
b_ma = wape(test["sales"], test["pred_ma28"])

# ---------- 5. model ----------
params = dict(n_estimators=600, learning_rate=0.05, num_leaves=63,
              min_child_samples=40, subsample=0.8, colsample_bytree=0.8)

if USE_MLFLOW:
    mlflow.set_tracking_uri(f"sqlite:///{OUT / 'mlflow.db'}")
    mlflow.set_experiment("demand_forecasting")
    mlflow.start_run(run_name="lightgbm_v1" if USE_LGBM else "histgb_v1")

if USE_LGBM:
    model = lgb.LGBMRegressor(objective="poisson", random_state=SEED, verbose=-1,
                              subsample_freq=1, **params)
    model.fit(train[FEATURES], train["sales"],
              eval_set=[(valid[FEATURES], valid["sales"])],
              callbacks=[lgb.early_stopping(50, verbose=False)],
              categorical_feature=["store", "item"])
    importances = pd.Series(model.feature_importances_, FEATURES)
else:
    model = HistGradientBoostingRegressor(loss="poisson", learning_rate=0.05, max_iter=300,
                                          max_leaf_nodes=63, random_state=SEED,
                                          early_stopping=False)
    model.fit(train[FEATURES], train["sales"])
    from sklearn.inspection import permutation_importance
    sample = valid.sample(min(len(valid), 6000), random_state=SEED)
    pi = permutation_importance(model, sample[FEATURES], sample["sales"], n_repeats=2,
                                random_state=SEED, n_jobs=-1)
    importances = pd.Series(pi.importances_mean, FEATURES)

test["pred_model"] = np.clip(model.predict(test[FEATURES]), 0, None)
valid_pred = np.clip(model.predict(valid[FEATURES]), 0, None)

w_model = wape(test["sales"], test["pred_model"])
metrics = pd.DataFrame({
    "model": ["Naive (same as last week)", "28-day moving average",
              "LightGBM" if USE_LGBM else "Gradient boosting (sklearn)"],
    "WAPE": [b_naive, b_ma, w_model]})
metrics["improvement_vs_ma28_%"] = (1 - metrics["WAPE"] / b_ma) * 100
print("\n", metrics.round(4).to_string(index=False))

if USE_MLFLOW:
    mlflow.log_params(params)
    mlflow.log_metrics({"wape_test": w_model, "wape_naive": b_naive, "wape_ma28": b_ma})
    mlflow.end_run()

# ---------- 6. save outputs ----------
metrics.to_csv(OUT / "metrics.csv", index=False)
test[["date", "store", "item", "sales", "promo", "pred_naive", "pred_ma28", "pred_model"]] \
    .to_csv(OUT / "forecast_results.csv", index=False)
vv = valid[["date", "store", "item", "sales"]].copy()
vv["pred_model"] = valid_pred
vv.to_csv(OUT / "validation_errors.csv", index=False)

# ---------- 7. charts ----------
fig, ax = plt.subplots(figsize=(10, 4))
daily = test.groupby("date")[["sales", "pred_model", "pred_ma28"]].sum()
ax.plot(daily.index, daily["sales"], label="Actual", lw=2, color="black")
ax.plot(daily.index, daily["pred_model"], label="Model forecast", lw=2, color="tab:blue")
ax.plot(daily.index, daily["pred_ma28"], label="Moving average (baseline)", ls="--", color="tab:orange")
ax.set_title("Total daily sales: actual vs forecast (test period)")
ax.legend(); ax.grid(alpha=.3); fig.autofmt_xdate(); fig.tight_layout()
fig.savefig(OUT / "chart_forecast_vs_actual.png", dpi=130); plt.close(fig)

fig, ax = plt.subplots(figsize=(7, 4))
importances.sort_values().tail(10).plot.barh(ax=ax, color="tab:blue")
ax.set_title("Top 10 most important features"); fig.tight_layout()
fig.savefig(OUT / "chart_feature_importance.png", dpi=130); plt.close(fig)

fig, ax = plt.subplots(figsize=(6, 3.5))
ax.bar(["Naive", "Moving avg", "Model"], metrics["WAPE"] * 100,
       color=["tab:gray", "tab:orange", "tab:blue"])
ax.set_ylabel("WAPE %  (lower is better)"); ax.set_title("Forecast error comparison")
fig.tight_layout(); fig.savefig(OUT / "chart_wape_comparison.png", dpi=130); plt.close(fig)
print("\nSaved outputs/ (metrics, forecast_results, validation_errors, charts)")
