"""
03_train_validate.py  —  train several models, validate, save the best.
Reads : data/features.csv
Writes: models/best_model.joblib  +  prints a comparison table
Split : time-based (last 16 weeks held out).  Metric: WMAPE + bias.
Run:  python 03_train_validate.py
"""
import numpy as np, pandas as pd, joblib, time
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from forecast_core import CATS, NUM, FEATURES

ROOT = Path(__file__).resolve().parent
feat = pd.read_csv(ROOT/"data"/"features.csv", parse_dates=["week_start_date"])
feat = feat.sort_values(["sku_id","store_id","week_start_date"]).reset_index(drop=True)
for c in CATS: feat[c+"_code"]=feat[c].astype("category").cat.codes
TARGET="units_sold"

cutoff=feat["week_start_date"].max()-pd.Timedelta(weeks=16)
train=feat[feat.week_start_date<=cutoff]; test=feat[feat.week_start_date>cutoff]
Xtr,ytr=train[FEATURES],train[TARGET]; Xte,yte=test[FEATURES],test[TARGET]
print(f"TRAIN {len(train):,} rows (to {cutoff.date()}) | TEST {len(test):,} rows (last 16 weeks)\n")

def wmape(y,p): return np.sum(np.abs(y-p))/np.sum(np.abs(y))*100
def bias(y,p):  return np.sum(p-y)/np.sum(y)*100

rows=[("Naive (last week)", wmape(yte,test["lag_1"]), bias(yte,test["lag_1"]), None),
      ("Moving avg (4wk)",  wmape(yte,test["roll_mean_4"]), bias(yte,test["roll_mean_4"]), None)]
models={
 "Linear Regression": make_pipeline(StandardScaler(), LinearRegression()),
 "Random Forest": RandomForestRegressor(n_estimators=300,max_depth=16,min_samples_leaf=3,random_state=42,n_jobs=-1),
 "XGBoost": XGBRegressor(n_estimators=700,learning_rate=0.05,max_depth=7,subsample=0.8,colsample_bytree=0.8,random_state=42,n_jobs=-1),
 "LightGBM": LGBMRegressor(n_estimators=900,learning_rate=0.05,num_leaves=63,subsample=0.8,colsample_bytree=0.8,random_state=42,n_jobs=-1,verbose=-1),
}
fitted={}
for name,m in models.items():
    t0=time.time(); m.fit(Xtr,ytr); dt=time.time()-t0
    p=np.clip(m.predict(Xte),0,None); fitted[name]=m
    rows.append((name, wmape(yte,p), bias(yte,p), dt))
    print(f"  {name:18s} WMAPE {wmape(yte,p):5.2f}%  bias {bias(yte,p):+5.2f}%  ({dt:.1f}s)")

res=pd.DataFrame(rows,columns=["Model","WMAPE_%","Bias_%","Train_s"]).round(2)
print("\n=== VALIDATION (held-out last 16 weeks) ===")
print(res.to_string(index=False))
ml=res[res.Train_s.notna()]; best=ml.loc[ml["WMAPE_%"].idxmin(),"Model"]
print(f"\nBest model: {best}  ({ml['WMAPE_%'].min():.2f}% WMAPE, "
      f"{(1-ml['WMAPE_%'].min()/res.loc[0,'WMAPE_%'])*100:.0f}% better than naive)")

(ROOT/"models").mkdir(exist_ok=True)
joblib.dump(fitted[best], ROOT/"models"/"best_model.joblib")
res.to_csv(ROOT/"models"/"validation_results.csv", index=False)
print(f"Saved models/best_model.joblib ({best})")
