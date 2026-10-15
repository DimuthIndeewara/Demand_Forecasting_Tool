"""Precompute 3-month (13-week) product-level forecasts for the web app."""
import json, numpy as np
import forecast_core as fc

m = fc.load_model(); df, plist = fc.load_history()
DISCOUNTS = [0, 10, 20, 30]
cats = df.groupby("sku_id")["category"].first().to_dict()

products = []
for sku, name in plist:
    hist, _ = fc.forecast_product_horizon(m, df, sku, horizon=13, discount_pct=0)
    hist = hist.tail(26)
    hrec = [{"date": d.strftime("%Y-%m-%d"), "units": int(u)}
            for d, u in zip(hist["date"], hist["units"])]
    futures = {}
    for disc in DISCOUNTS:
        _, fut = fc.forecast_product_horizon(m, df, sku, horizon=13, discount_pct=disc)
        futures[str(disc)] = [{"date": d.strftime("%Y-%m-%d"), "units": int(u)}
                              for d, u in zip(fut["date"], fut["units"])]
    products.append({"sku": sku, "name": name, "category": cats[sku],
                     "history": hrec, "futures": futures})
    print(f"  {name:26s} done")

out = {"discounts": DISCOUNTS,
       "categories": sorted(df["category"].unique().tolist()),
       "products": products}
json.dump(out, open("/home/claude/app_data3.json", "w"))
print("\nwrote app_data3.json  kb:", round(len(json.dumps(out))/1024))
