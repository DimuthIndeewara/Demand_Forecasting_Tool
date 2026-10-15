"""
02_build_features.py  —  turn the raw data into model features.
Reads : data/raw_dataset.csv
Writes: data/features.csv
Leakage-safe: past-sales features are shifted per (sku, store) series.
Run:  python 02_build_features.py
"""
import numpy as np, pandas as pd
from pathlib import Path

D = Path(__file__).resolve().parent / "data"
df = pd.read_csv(D/"raw_dataset.csv", parse_dates=["week_start_date"])
df = df.sort_values(["sku_id","store_id","week_start_date"]).reset_index(drop=True)
g = df.groupby(["sku_id","store_id"], group_keys=False)

# past-sales features (shifted = safe)
df["lag_1"]=g["units_sold"].shift(1); df["lag_2"]=g["units_sold"].shift(2)
df["lag_4"]=g["units_sold"].shift(4); df["lag_52"]=g["units_sold"].shift(52)
df["roll_mean_4"]=g["units_sold"].transform(lambda s:s.shift(1).rolling(4).mean())
df["roll_mean_12"]=g["units_sold"].transform(lambda s:s.shift(1).rolling(12).mean())
df["roll_std_4"]=g["units_sold"].transform(lambda s:s.shift(1).rolling(4).std())
# calendar
df["month"]=df["week_start_date"].dt.month
df["week_sin"]=np.sin(2*np.pi*df["week_no"]/52); df["week_cos"]=np.cos(2*np.pi*df["week_no"]/52)
df["is_quarter_end"]=df["week_no"].isin([13,26,39,52]).astype(int)
# price & competition
df["price_index"]=(df["price"]/df["mrp"]).round(3)
df["price_vs_comp"]=(df["price"]/df["competitor_price"]).round(3)
# promo & stockout (outcome -> only lag is usable)
df["promo_lag1"]=g["promo_flag"].shift(1).fillna(0).astype(int)
df["stockout_lag1"]=g["stockout_flag"].shift(1).fillna(0).astype(int)

need=["lag_1","lag_2","lag_4","roll_mean_4","roll_mean_12","roll_std_4"]
before=len(df); df=df.dropna(subset=need).reset_index(drop=True)
df["lag_52"]=df["lag_52"].fillna(df["roll_mean_12"])
for c in ("roll_mean_4","roll_mean_12","roll_std_4","lag_52"): df[c]=df[c].round(1)

df.to_csv(D/"features.csv", index=False)
print(f"Dropped {before-len(df):,} warm-up rows. Wrote {D/'features.csv'} ({len(df):,} rows)")
