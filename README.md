# Retail Demand Forecasting (ML) — full working project

An end-to-end machine learning demand forecasting system for a multi-store retail network: synthetic data, feature engineering, model training and validation, and two live interfaces (a web app and an API) that serve the trained model in real time.

All data is synthetic and represents no real company or product.

---

## What it does

Forecasts weekly units sold using the drivers that shape demand: recent sales momentum, price and discount, a competitor's price, promotions, seasonality, holidays, weather, store format and region. The best model reaches about **7.5% WMAPE on held-out weeks, roughly 65% better than a naive baseline**, with low bias.

The app shows a **4-week forecast per product**, summed across all stores. It is produced by **recursive forecasting**: the model predicts the next week, that prediction is fed back in as an input, and it rolls forward 4 times. Discount and temperature sliders let you test a promotion or weather scenario, and the forecast re-projects live.

The framing is **predict then decide**: the model predicts demand, and that prediction supports pricing, promotion and ordering decisions.

---

## Project layout

```
forecast_project/
├── 01_build_dataset.py      # generate the synthetic SKU x store x week data
├── 02_build_features.py     # turn raw data into leakage-safe model features
├── 03_train_validate.py     # train 4 models, validate, save the best
├── forecast_core.py         # shared live-prediction logic (used by app + api)
├── app_streamlit.py         # interactive web app (pure Python)
├── serve_api.py             # production-style serving API (FastAPI)
├── requirements.txt
├── data/                    # created by the scripts (raw_dataset.csv, features.csv)
└── models/                  # created by training (best_model.joblib)
```

---

## Setup (once)

```bash
# from inside forecast_project/
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Build the model from scratch

Run the three steps in order:

```bash
python 01_build_dataset.py       # writes data/raw_dataset.csv  (18,720 rows)
python 02_build_features.py      # writes data/features.csv     (17,280 rows)
python 03_train_validate.py      # trains, validates, saves models/best_model.joblib
```

Step 3 prints the model comparison and saves the winner (LightGBM).

---

## Run the interactive app (easiest)

```bash
streamlit run app_streamlit.py
```

Open the URL it prints (usually http://localhost:8501). Pick a product, set a discount and a temperature, and the 4-week forecast re-projects live. Every number is a fresh call to the trained model.

## Or run the API (production-style)

```bash
uvicorn serve_api:app --reload
```

- Interactive docs: open http://localhost:8000/docs
- Example request to `POST /predict`:

```json
{ "sku_id": "SKU215", "store_id": "ST01", "discount_pct": 20, "temperature_c": 24 }
```

Response:

```json
{ "forecast_units": 179, "baseline_units": 119, "uplift_pct": 50.8, "price": 288, "mrp": 360 }
```

---

## How the pieces fit together

```
trained model (.joblib)  ->  forecast_core.py  ->  app_streamlit.py  (web UI)
                                               \->  serve_api.py      (REST API)
```

- `forecast_core.py` loads the model and the latest feature context per SKU-store, and exposes one `predict_units(...)` function. Both interfaces call it, so there is a single source of truth.
- The Streamlit app is the quickest interface: pure Python, no HTML or JavaScript.
- The FastAPI service is the production shape: the model is loaded once at startup and answers prediction requests over HTTP, which any other system can call.

---

## Modelling notes

- **One global model** across all SKUs and stores (identity encoded as features), so a sparse series borrows patterns from similar ones.
- **Time-based validation** (train on earlier weeks, test on the most recent 16), never random, to avoid leaking the future.
- **Leakage-safe features**: anything built from past sales is shifted so a row only sees earlier weeks; planned drivers (price, promo, calendar) are used for the target week directly.
- **Metrics**: WMAPE (volume-weighted accuracy) and bias (systematic over/under-forecast), both compared against naive baselines.
- **Model choice**: gradient-boosted trees (LightGBM / XGBoost) beat linear regression and neural networks on this tabular data, while staying fast and explainable.

## Tech stack

Python, pandas, NumPy, scikit-learn, XGBoost, LightGBM, joblib, Streamlit, FastAPI.
