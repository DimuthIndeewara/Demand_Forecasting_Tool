"""
forecast_core.py
Shared logic for the demand forecaster.
  - predict_units(): single next-week prediction (what-if).
  - forecast_product_horizon(): recursive multi-week forecast at PRODUCT level
    (summed across stores), for a 3-month (13-week) horizon.
Used by the Streamlit app and the API. One source of truth.
"""
import numpy as np
import pandas as pd
import joblib
from pathlib import Path

ROOT = Path(__file__).resolve().parent
FEATURES_CSV = ROOT / "data" / "features.csv"
MODEL_PATH   = ROOT / "models" / "best_model.joblib"

CATS = ["region","store_id","store_format","sku_id","brand","category","subcategory"]
NUM  = ["lag_1","lag_2","lag_4","lag_52","roll_mean_4","roll_mean_12","roll_std_4",
        "month","week_sin","week_cos","is_quarter_end","price","price_index",
        "competitor_price","price_vs_comp","promo_flag","discount_pct","promo_lag1",
        "avg_temp_c","holiday_flag","stockout_lag1"]
FEATURES = NUM + [c + "_code" for c in CATS]
HOLIDAY_WEEKS = {1, 13, 30, 47, 51, 52}   # same calendar as the data generator


def ensure_ready():
    """
    Build the data and train the model if they are not already present.
    Lets the app run on a fresh deployment (e.g. Streamlit Cloud) where only
    the code was uploaded, no data or model files.
    """
    import pipeline
    if not FEATURES_CSV.exists():
        pipeline.build_raw()
        pipeline.build_features()
    if not MODEL_PATH.exists():
        pipeline.train_and_save()


def load_model():
    return joblib.load(MODEL_PATH)


def load_history(features_csv: Path = FEATURES_CSV):
    """Return everything needed for both single and multi-week forecasting."""
    df = pd.read_csv(features_csv, parse_dates=["week_start_date"])
    for c in CATS:
        df[c + "_code"] = df[c].astype("category").cat.codes
    df = df.sort_values(["sku_id","store_id","week_start_date"]).reset_index(drop=True)

    products = (df.groupby("sku_id")["sku_name"].first()
                  .reset_index().sort_values("sku_name"))
    product_list = list(zip(products["sku_id"], products["sku_name"]))
    return df, product_list


def _series_context(df, sku, store):
    g = df[(df.sku_id == sku) & (df.store_id == store)].sort_values("week_start_date")
    last = g.iloc[-1]
    mrp = last["price"] / last["price_index"] if last["price_index"] else last["price"]
    codes = {c + "_code": last[c + "_code"] for c in CATS}
    seasonal_temp = g.groupby(g["week_start_date"].dt.isocalendar().week)["avg_temp_c"].mean().to_dict()
    return {
        "units": list(g["units_sold"].values),
        "dates": list(g["week_start_date"].values),
        "mrp": float(mrp), "comp": float(last["competitor_price"]),
        "codes": codes, "seasonal_temp": seasonal_temp,
        "last_promo": int(last["promo_flag"]),
    }


def forecast_product_horizon(model, df, sku, horizon=4, discount_pct=0,
                             temperature_c=None, hist_weeks=16):
    """
    Recursive multi-week forecast for ONE product, summed across all its stores.
    - horizon: number of future weeks (default 4).
    - discount_pct: applied across the whole horizon (a promo scenario).
    - temperature_c: if given, used for every future week; if None, each week
      follows its seasonal norm.
    - competitor price follows recent norm; holidays come from the calendar.
    Returns (history_df, future_df) at product (all-stores) level.
    """
    stores = sorted(df[df.sku_id == sku]["store_id"].unique())
    per_store_future = []
    hist_by_week = None

    for store in stores:
        ctx = _series_context(df, sku, store)
        units = list(ctx["units"]); dates = list(ctx["dates"])
        mrp, comp, codes = ctx["mrp"], ctx["comp"], ctx["codes"]
        last_date = pd.Timestamp(dates[-1])
        fut_dates, fut_units = [], []
        prev_promo = ctx["last_promo"]

        for step in range(1, horizon + 1):
            d = last_date + pd.Timedelta(weeks=step)
            wk = int(d.isocalendar().week); mo = d.month
            def back(n): return units[-n] if len(units) >= n else np.mean(units[-12:])
            roll4 = np.mean(units[-4:]); roll12 = np.mean(units[-12:]); std4 = np.std(units[-4:])
            price = round(mrp * (1 - discount_pct / 100), 0)
            row = {
                "lag_1": back(1), "lag_2": back(2), "lag_4": back(4), "lag_52": back(52),
                "roll_mean_4": roll4, "roll_mean_12": roll12, "roll_std_4": std4,
                "month": mo, "week_sin": np.sin(2*np.pi*wk/52), "week_cos": np.cos(2*np.pi*wk/52),
                "is_quarter_end": 1 if wk in (13,26,39,52) else 0,
                "price": price, "price_index": round(1-discount_pct/100,3),
                "competitor_price": comp,
                "price_vs_comp": round(price/comp,3) if comp else 1.0,
                "promo_flag": 1 if discount_pct>0 else 0, "discount_pct": discount_pct,
                "promo_lag1": prev_promo,
                "avg_temp_c": (temperature_c if temperature_c is not None
                               else ctx["seasonal_temp"].get(wk, np.mean(list(ctx["seasonal_temp"].values())))),
                "holiday_flag": 1 if wk in HOLIDAY_WEEKS else 0,
                "stockout_lag1": 0, **codes,
            }
            yhat = max(0.0, float(model.predict(pd.DataFrame([row])[FEATURES])[0]))
            units.append(yhat); dates.append(d)
            fut_dates.append(d); fut_units.append(yhat)
            prev_promo = 1 if discount_pct > 0 else 0

        per_store_future.append(pd.DataFrame({"date": fut_dates, "units": fut_units}))
        # product-level history (sum across stores), last hist_weeks
        h = df[(df.sku_id == sku) & (df.store_id == store)].sort_values("week_start_date").tail(hist_weeks)
        hs = h.set_index("week_start_date")["units_sold"]
        hist_by_week = hs if hist_by_week is None else hist_by_week.add(hs, fill_value=0)

    future = (pd.concat(per_store_future).groupby("date")["units"].sum().round().reset_index())
    history = hist_by_week.reset_index(); history.columns = ["date", "units"]
    return history, future


def predict_units(model, df, sku, store, discount_pct=0, temperature_c=16):
    """Single next-week what-if for one SKU in one store (used by the API)."""
    ctx = _series_context(df, sku, store)
    def _p(disc, temp):
        row = {f: None for f in FEATURES}; row.update(ctx["codes"])
        u = ctx["units"]; mrp, comp = ctx["mrp"], ctx["comp"]
        price = round(mrp*(1-disc/100),0)
        row.update({"lag_1":u[-1],"lag_2":u[-2],"lag_4":u[-4],
                    "lag_52":u[-52] if len(u)>=52 else np.mean(u[-12:]),
                    "roll_mean_4":np.mean(u[-4:]),"roll_mean_12":np.mean(u[-12:]),
                    "roll_std_4":np.std(u[-4:]),"month":6,
                    "week_sin":0,"week_cos":1,"is_quarter_end":0,"price":price,
                    "price_index":round(1-disc/100,3),"competitor_price":comp,
                    "price_vs_comp":round(price/comp,3) if comp else 1.0,
                    "promo_flag":1 if disc>0 else 0,"discount_pct":disc,"promo_lag1":0,
                    "avg_temp_c":temp,"holiday_flag":0,"stockout_lag1":0})
        return max(0.0,float(model.predict(pd.DataFrame([row])[FEATURES])[0]))
    f=_p(discount_pct,temperature_c); b=_p(0,temperature_c)
    return {"forecast_units":round(f),"baseline_units":round(b),
            "uplift_pct":round((f/b-1)*100,1) if b>0 else 0.0,
            "price":round(ctx["mrp"]*(1-discount_pct/100)),"mrp":round(ctx["mrp"])}
