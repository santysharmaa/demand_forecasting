"""
Step 3 - Inventory optimisation on top of the forecasts.

For every store-item we compute:
  * Safety stock   = Z * (std of daily forecast error) * sqrt(lead time)
  * Reorder point  = avg daily demand * lead time + safety stock
  * EOQ            = sqrt(2 * yearly demand * order cost / holding cost per unit)

Then we SIMULATE the test period with two policies:
  A) reorder point WITHOUT safety stock
  B) reorder point WITH safety stock
and compare stockouts vs average stock held.

ALL costs / lead time are ASSUMPTIONS (see config.py). Say that clearly.
"""
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from config import *

rng = np.random.default_rng(SEED)
fc = pd.read_csv(OUT / "forecast_results.csv", parse_dates=["date"])
val = pd.read_csv(OUT / "validation_errors.csv", parse_dates=["date"])

unit_cost = pd.Series(rng.uniform(5, 50, N_ITEMS + 1), index=range(N_ITEMS + 1)).round(2)

# ---------- policy per store-item ----------
val["err"] = val["sales"] - val["pred_model"]
err = val.groupby(["store", "item"])["err"].std().rename("err_std")
avg_d = fc.groupby(["store", "item"])["pred_model"].mean().rename("avg_daily_forecast")
pol = pd.concat([avg_d, err], axis=1).reset_index()
pol["unit_cost"] = pol["item"].map(unit_cost)
pol["safety_stock"] = SERVICE_Z * pol["err_std"] * np.sqrt(LEAD_TIME_DAYS)
pol["reorder_point_no_ss"] = pol["avg_daily_forecast"] * LEAD_TIME_DAYS
pol["reorder_point"] = pol["reorder_point_no_ss"] + pol["safety_stock"]
holding = HOLDING_RATE * pol["unit_cost"]
pol["eoq"] = np.sqrt(2 * pol["avg_daily_forecast"] * 365 * ORDER_COST / holding)
pol = pol.round(2)
pol.to_csv(OUT / "inventory_policy.csv", index=False)


# ---------- simulation on the real test-period demand ----------
def simulate(demand, rop, q, lead):
    on_hand = rop + q
    pipeline = {}
    stockout_days, lost, stock_sum = 0, 0.0, 0.0
    for day, d in enumerate(demand):
        on_hand += pipeline.pop(day, 0)
        if d > on_hand:
            stockout_days += 1
            lost += d - on_hand
            on_hand = 0
        else:
            on_hand -= d
        position = on_hand + sum(pipeline.values())
        if position <= rop:
            pipeline[day + lead] = q
        stock_sum += on_hand
    return stockout_days, lost, stock_sum / len(demand)


demand_map = {k: g.sort_values("date")["sales"].values for k, g in fc.groupby(["store", "item"])}
res = []
for _, r in pol.iterrows():
    dem = demand_map[(r["store"], r["item"])]
    q = max(r["eoq"], 1)
    for name, rop in (("A: no safety stock", r["reorder_point_no_ss"]),
                      ("B: with safety stock", r["reorder_point"])):
        so, lost, avg_stock = simulate(dem, rop, q, LEAD_TIME_DAYS)
        res.append(dict(store=r["store"], item=r["item"], policy=name,
                        stockout_days=so, lost_units=lost, avg_stock=avg_stock,
                        avg_stock_value=avg_stock * r["unit_cost"], total_demand=dem.sum()))
res = pd.DataFrame(res)
res.to_csv(OUT / "simulation_detail.csv", index=False)

n_series = len(demand_map)
summ = res.groupby("policy").agg(
    stockout_days=("stockout_days", "sum"), lost_units=("lost_units", "sum"),
    total_demand=("total_demand", "sum"), avg_stock_value=("avg_stock_value", "sum")).reset_index()
summ["stockout_day_rate_%"] = summ["stockout_days"] / (n_series * TEST_DAYS) * 100
summ["fill_rate_%"] = (1 - summ["lost_units"] / summ["total_demand"]) * 100
summ = summ.drop(columns=["stockout_days"]).round(2)
summ.to_csv(OUT / "simulation_summary.csv", index=False)
print(summ.to_string(index=False))

fig, ax = plt.subplots(1, 2, figsize=(9, 3.6))
ax[0].bar(["No safety\nstock", "With safety\nstock"], summ["stockout_day_rate_%"], color=["tab:red", "tab:green"])
ax[0].set_title("% of days with a stockout")
ax[1].bar(["No safety\nstock", "With safety\nstock"], summ["avg_stock_value"], color=["tab:red", "tab:green"])
ax[1].set_title("Average stock value held ($)")
fig.tight_layout(); fig.savefig(OUT / "chart_inventory_tradeoff.png", dpi=130); plt.close(fig)

risk = res[res["policy"].str.startswith("B")].copy()
risk["stockout_risk"] = risk["stockout_days"] / TEST_DAYS
risk.sort_values("stockout_risk", ascending=False).to_csv(OUT / "stockout_risk.csv", index=False)
print("\nSaved inventory_policy.csv, simulation_summary.csv, stockout_risk.csv, chart")
