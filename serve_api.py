"""
serve_api.py  —  production-style serving layer (FastAPI).
Loads the trained model once and serves forecasts.
  POST /forecast  -> 4-week product-level forecast
  POST /predict   -> single next-week what-if (one SKU, one store)
Run:   uvicorn serve_api:app --reload
Docs:  http://localhost:8000/docs
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import forecast_core as fc

app = FastAPI(title="Demand Forecast API", version="2.0")
MODEL = fc.load_model()
DF, PRODUCTS = fc.load_history()
VALID_SKUS = {s for s, _ in PRODUCTS}


class ForecastRequest(BaseModel):
    sku_id: str
    discount_pct: float = 0
    horizon_weeks: int = 4
    temperature_c: float | None = None


class PredictRequest(BaseModel):
    sku_id: str
    store_id: str
    discount_pct: float = 0
    temperature_c: float = 16


@app.get("/")
def health():
    return {"status": "ok", "products": len(PRODUCTS)}


@app.get("/products")
def products():
    return {"products": [{"sku_id": s, "name": n} for s, n in PRODUCTS]}


@app.post("/forecast")
def forecast(req: ForecastRequest):
    if req.sku_id not in VALID_SKUS:
        raise HTTPException(404, f"Unknown sku_id {req.sku_id}")
    _, fut = fc.forecast_product_horizon(MODEL, DF, req.sku_id,
                                         horizon=req.horizon_weeks,
                                         discount_pct=req.discount_pct,
                                         temperature_c=req.temperature_c)
    return {"sku_id": req.sku_id, "discount_pct": req.discount_pct,
            "horizon_weeks": req.horizon_weeks,
            "total_units": int(fut["units"].sum()),
            "weekly": [{"date": d.strftime("%Y-%m-%d"), "units": int(u)}
                       for d, u in zip(fut["date"], fut["units"])]}


@app.post("/predict")
def predict(req: PredictRequest):
    if req.sku_id not in VALID_SKUS:
        raise HTTPException(404, f"Unknown sku_id {req.sku_id}")
    return {"sku_id": req.sku_id, "store_id": req.store_id,
            **fc.predict_units(MODEL, DF, req.sku_id, req.store_id,
                               req.discount_pct, req.temperature_c)}
