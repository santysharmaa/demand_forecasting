# Demand Forecasting & Inventory Optimization

**Predict demand. Plan inventory. Make better stocking decisions.**

What if a retailer could estimate next week's sales before placing an order, instead of relying entirely on guesswork?

That's the idea behind this project. I built an end-to-end supply chain analytics pipeline that forecasts product demand, compares forecasting models, and uses the predictions to explore how inventory decisions affect stockouts and holding costs.

The project combines Python, machine learning, inventory management concepts, experiment tracking, and Power BI reporting in one workflow.

## What does the project solve?

Imagine managing 10 stores that sell 20 different products. Some products sell consistently, while others experience changes because of weekends, promotions, holidays, or seasonal demand.

Ordering too little can mean lost sales and unhappy customers. Ordering too much ties up money in inventory that may not sell quickly.

The goal of this project is to answer two practical questions:

- How much of each product might a store sell over the next 7 days?
- How much inventory should it maintain to reduce stockouts without unnecessarily increasing stock levels?

Rather than evaluating a machine learning model in isolation, the project connects its forecasts to a simulated inventory planning process.

## How the project works

```text
Historical Sales Data
        |
        v
Data Preparation & Feature Engineering
        |
        v
Time-Based Train / Validation / Test Split
        |
        v
Naive Baseline vs Moving Average vs LightGBM
        |
        v
Forecast Evaluation using WAPE
        |
        v
7-Day Demand Forecast
        |
        v
Safety Stock, Reorder Point & EOQ
        |
        v
Inventory Policy Simulation
        |
        v
Results, Charts & Power BI Dashboard
```

Each stage has a specific purpose. The forecasting stage estimates future demand, while the inventory stage uses those estimates to explore stocking decisions.

## Dataset

The project works with daily sales records for 20 products across 10 stores, covering 2022–2025.

The dataset contains approximately 292,200 rows and the following fields:

| Column | Description |
|---|---|
| `date` | Date of the sales observation |
| `store` | Store identifier |
| `item` | Product identifier |
| `sales` | Number of units sold |
| `promo` | Indicates whether a promotion was active |
| `is_holiday` | Indicates whether the date was a holiday |

**Data source:** The repository includes `01_generate_data.py`, which generates a sample dataset. Before treating the results as real-world evidence, the dataset source should be verified and documented. If the reported results come from the generated data, they should be described as results on a synthetic retail dataset, not actual retailer sales.

The forecasting pipeline expects data in this general format. Real-world data may require additional cleaning, validation, and preprocessing.

## Forecasting approach

I compared three approaches to understand whether a machine learning model provides a meaningful improvement over simpler forecasting methods.

### 1. Naive baseline

The baseline uses sales from seven days earlier as the prediction for the corresponding day.

For example, if a store sold 40 units of a product last Monday, the model predicts 40 units for the following Monday.

It's simple, but useful as a starting point because it captures a basic weekly pattern.

### 2. 28-day moving average

This method uses recent sales history to estimate future demand. Averaging a longer period can reduce the influence of unusually high or low sales on individual days.

It provides a stronger baseline when demand fluctuates from one day to another.

### 3. LightGBM

For the main model, I used LightGBM with a Poisson objective, which is suitable for modelling non-negative count-like targets such as units sold.

Instead of relying on a single historical value or average, the model learns relationships between sales and multiple input features.

The feature set includes:

- **Lag features:** Sales from 7, 14, 28, and 364 days earlier.
- **Rolling statistics:** Recent 7-day and 28-day average sales, plus 28-day rolling standard deviation.
- **Calendar features:** Day of the week, month, day of the year, week of the year, and weekend indicator.
- **Store and product information:** Store and item identifiers.
- **External indicators available in the dataset:** Promotions and holidays.

Rolling features are shifted so that the calculation uses historical observations rather than the target day's sales.

The model uses a learning rate of 0.05, 600 estimators, and 63 leaves, alongside other configured parameters. Early stopping is used with a validation window to help select an appropriate number of boosting iterations.

## Model evaluation

A forecasting model should be evaluated on dates it was not trained on. Randomly shuffling time-series observations can produce an unrealistic evaluation because it ignores the order in which information becomes available.

I therefore used a chronological split:

- **Training:** Older historical observations used to fit the model.
- **Validation:** The preceding 60-day window used for model selection and tuning.
- **Testing:** The final 60-day window reserved for evaluating performance.

The forecast horizon is 7 days.

### Evaluation metric: WAPE

Weighted Absolute Percentage Error (WAPE) measures the total absolute forecasting error relative to the total actual demand.

\[
\text{WAPE} =
\frac{\sum |\text{Actual}-\text{Predicted}|}
{\sum \text{Actual}}\times 100
\]

Lower WAPE indicates lower aggregate absolute error relative to actual demand.

### Forecasting results

The following are the results previously obtained on the project's original held-out test set. They should be reproduced from the current code and output files before being treated as final published results.

| Forecasting method | WAPE |
|---|---:|
| Naive baseline | 31.52% |
| 28-day moving average | 26.24% |
| LightGBM | **21.92%** |

LightGBM reduced WAPE by approximately **16.46% relative to the moving-average baseline**.

These results suggest that incorporating multiple historical and calendar features helped the model outperform the simpler approaches on this particular test set. They do not guarantee the same performance on a different dataset, time period, or retailer.

## From forecasts to inventory decisions

A demand forecast becomes more useful when it helps answer operational questions. The inventory module translates demand estimates and historical forecast errors into stocking parameters.

### Safety stock

Safety stock is additional inventory kept as a buffer against unexpected demand and forecasting errors.

The project estimates it using the variability of validation-period forecast errors, a service-level Z value of 1.65, and a lead time of 7 days.

The calculation assumes that forecast errors are sufficiently well behaved for this approximation to be useful. Actual safety-stock requirements depend on the error distribution, demand patterns, replenishment process, and the service level a business wants to achieve.

### Reorder point (ROP)

The reorder point is the inventory level at which a replenishment order should be triggered.

The basic idea is:

**Reorder point = Expected demand during lead time + Safety stock**

For example, if a product is expected to sell 10 units per day and replenishment takes 7 days, expected lead-time demand is 70 units. Any additional safety stock provides a buffer against uncertainty.

This is an illustrative example, not a specific result from the dataset.

### Economic order quantity (EOQ)

EOQ estimates an order quantity that balances ordering costs against inventory holding costs.

The project uses annualized demand, an assumed ordering cost of $50, and an annual holding-rate assumption of 20%. Unit costs are assigned using a seeded random generator within the configured $5–$50 range.

These costs are modelling assumptions, not actual supplier quotations or observed business expenses. In a real deployment, they should be replaced with reliable purchasing and carrying-cost data.

## Inventory policy simulation

To see how safety stock changes inventory performance, I compared two policies over the test-period demand:

- **Policy A — Without safety stock:** Replenishment planning does not include the additional safety-stock buffer.
- **Policy B — With safety stock:** Replenishment planning includes the calculated buffer.

The simulation tracks demand that could not be fulfilled, stockout frequency, fill rate, and average inventory value.

### Simulation results

| Metric | Without safety stock | With safety stock |
|---|---:|---:|
| Lost units | 7,692.67 | 898.33 |
| Total demand | 506,442 | 506,442 |
| Stockout-day rate | 2.25% | 0.38% |
| Fill rate | 98.48% | 99.82% |
| Average stock value | $1,775,753.86 | $2,074,759.94 |

Under the project's simulation assumptions, adding safety stock reduced lost units by approximately **88.3%** and improved the fill rate from 98.48% to 99.82%.

That improvement came with a trade-off: average inventory value increased by approximately **16.8%**.

This is the central inventory-planning trade-off explored by the project. Holding more stock can help meet demand, but it also ties up more capital.

These are simulation results, not a guarantee of the same outcome in a real business. The fractional lost-unit values indicate that some simulation quantities may be averaged or aggregated; the implementation should be checked to understand precisely how those figures are calculated.

## Tech stack

| Technology | How it is used |
|---|---|
| Python | Pipeline implementation |
| Pandas | Data preparation and time-series feature engineering |
| LightGBM | Demand forecasting |
| MLflow | Experiment and metric tracking |
| Matplotlib / Seaborn, if used in the current code | Result visualizations |
| Power BI | Dashboard and business-facing reporting |

The exact visualization libraries and dashboard setup should match the dependencies and files in the repository.

## Project structure

```text
demand_forecasting_project/
├── data/
│   └── sales.csv
├── src/
│   ├── 01_generate_data.py
│   ├── 02_forecast.py
│   ├── 03_inventory.py
│   └── config.py
├── outputs/
├── README.md
├── POWERBI_GUIDE.md
├── INTERVIEW_PREP.md
└── requirements.txt
```

**What each file does**

- `01_generate_data.py` — Generates the sample sales dataset.
- `02_forecast.py` — Builds forecasting features, trains and evaluates the models, records experiment metrics, and produces prediction outputs.
- `03_inventory.py` — Calculates inventory parameters, simulates the policies, and compares their performance.
- `config.py` — Stores configurable parameters such as the forecast horizon, validation and test windows, lead time, ordering cost, and holding-rate assumption.
- `outputs/` — Stores generated predictions, metrics, charts, and other results.
- `POWERBI_GUIDE.md` — Explains how to use the outputs in Power BI.
- `INTERVIEW_PREP.md` — Contains project-related interview preparation material.

## Getting started

### 1. Clone the repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd demand_forecasting_project
```

Replace the placeholder with the actual repository URL.

### 2. Create and activate a virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

On macOS or Linux:

```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Generate the sample dataset

From the project root:

```bash
python src/01_generate_data.py
```

Skip this step when using an existing, properly prepared dataset.

### 5. Run the forecasting pipeline

```bash
python src/02_forecast.py
```

This step generates the forecasts, compares model performance, and saves the configured outputs.

### 6. Run the inventory simulation

```bash
python src/03_inventory.py
```

This step calculates inventory parameters and evaluates the configured stocking policies.

### 7. Inspect experiment runs (optional)

If MLflow is configured to use the local tracking directory shown by the project:

```bash
mlflow ui --backend-store-uri ./outputs/mlruns
```

Open the local URL printed by MLflow in your browser.

The tracking path must match the actual URI configured in the code. If the project uses a different tracking URI, use that instead.

## Visualizations and Power BI

The generated outputs can be used to build a dashboard that brings the modelling results closer to the decisions a planner needs to make.

Useful dashboard views include:

- Actual versus forecast sales over time.
- WAPE comparison across the three forecasting methods.
- Demand forecasts by store and product.
- Products with the highest predicted demand.
- Safety stock and reorder-point comparisons.
- Fill rate, stockout frequency, and lost-unit comparisons across policies.
- The relationship between service performance and inventory value.

Refer to `POWERBI_GUIDE.md` for the expected output files and dashboard setup.

## Assumptions and limitations

This is a learning and portfolio project designed to demonstrate an end-to-end workflow. Several choices simplify the real-world problem:

- Lead time, ordering cost, holding-rate assumptions, and unit costs are configured or simulated rather than sourced from a real retailer.
- Safety-stock calculations rely on historical forecast-error variability and simplifying distribution assumptions.
- The inventory simulation covers a limited test period and uses the replenishment logic implemented in the project.
- Promotions and holidays are represented by available indicators; richer information about discounts, weather, prices, or local events could improve the feature set.
- The reported WAPE reflects one held-out test period and should not be treated as universal model accuracy.
- The validity of the results depends on correct time-based feature generation, leakage-free evaluation, and realistic inventory simulation rules.

Before using the pipeline with a new dataset, check the column names, date coverage, missing values, duplicate store-product-date records, sales types, and availability of enough history for the 364-day lag feature.

## What I learned

The most important part of this project was connecting forecasting performance to the decisions made after a prediction is produced.

A model with a lower forecasting error is useful, but inventory planning also depends on uncertainty, replenishment lead time, costs, and the service level a business wants to maintain.

Comparing simple baselines with LightGBM helped me evaluate whether additional model complexity was worthwhile. The inventory simulation then showed how a stocking policy can trade higher inventory investment for fewer lost sales.

The project also highlighted why evaluation methodology matters: historical data must be split in time order, and features used at prediction time must not rely on information that would only become available later.

## Possible next steps

There are several ways I would extend the project:

- Evaluate forecast performance separately by store and product.
- Add prediction intervals or probabilistic demand forecasts.
- Incorporate actual supplier lead times, product costs, and ordering constraints.
- Compare additional inventory policies under different service-level targets.
- Add external variables such as prices, discounts, and weather where relevant.
- Test the pipeline on a separate real-world dataset.
- Improve the Power BI dashboard with filters and product-level inventory recommendations.

## Final note

This project is an exploration of how demand forecasting and inventory planning can work together. It is not intended to represent a fully validated production supply-chain system.

The results are useful for understanding the workflow and comparing approaches under stated assumptions. A real deployment would require reliable operational data, further validation, and inventory rules adapted to the business.

**Built with Python, Pandas, LightGBM, MLflow, and Power BI.**
