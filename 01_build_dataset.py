"""
01_build_dataset.py  —  generate the synthetic multi-store retail dataset.
Writes: data/raw_dataset.csv   (one row per SKU per store per week)
Fully synthetic: no real company or product.
Run:  python 01_build_dataset.py
"""
import numpy as np, pandas as pd
from datetime import date, timedelta
from pathlib import Path

OUT = Path(__file__).resolve().parent / "data"; OUT.mkdir(exist_ok=True)
rng = np.random.default_rng(7)

weeks = [date(2023,1,2) + timedelta(weeks=i) for i in range(156)]
week_no = np.array([d.isocalendar().week for d in weeks]); year = np.array([d.year for d in weeks]); t = np.arange(156)

stores = [("ST01","North","Hypermarket",1.7),("ST02","North","Supermarket",1.0),
          ("ST03","Central","Hypermarket",1.6),("ST04","Central","Convenience",0.5),
          ("ST05","South","Supermarket",1.1),("ST06","South","Convenience",0.45)]
sm = pd.DataFrame(stores, columns=["store_id","region","store_format","size_mult"])
region_temp_off = {"North":-2.5,"Central":0.0,"South":2.8}

products = [
 ("SKU201","Aurora Shampoo 200ml","Aurora","Personal Care","Hair Care","200ml",190,180,520,"mild","none"),
 ("SKU202","Aurora Conditioner 200ml","Aurora","Personal Care","Hair Care","200ml",200,190,380,"mild","none"),
 ("SKU203","Lumen Hand Soap 250ml","Lumen","Personal Care","Skin Care","250ml",70,65,900,"none","none"),
 ("SKU204","Lumen Body Wash 400ml","Lumen","Personal Care","Skin Care","400ml",160,150,540,"none","none"),
 ("SKU205","Verda Toothpaste 120g","Verda","Personal Care","Oral Care","120g",110,100,760,"none","none"),
 ("SKU206","Verda Mouthwash 500ml","Verda","Personal Care","Oral Care","500ml",180,170,300,"none","none"),
 ("SKU207","Brisk Detergent 2L","Brisk","Home Care","Fabric Wash","2L",300,280,360,"none","none"),
 ("SKU208","Brisk Fabric Softener 1L","Brisk","Home Care","Fabric Wash","1L",180,168,320,"none","none"),
 ("SKU209","Gleam Dishwash 750ml","Gleam","Home Care","Dish Wash","750ml",160,150,600,"none","none"),
 ("SKU210","Gleam Surface Cleaner 1L","Gleam","Home Care","Surface Care","1L",150,140,420,"none","none"),
 ("SKU211","Hearth Black Tea 250g","Hearth","Beverages","Hot Drinks","250g",130,120,820,"winter","cold_up"),
 ("SKU212","Hearth Green Tea 100g","Hearth","Beverages","Hot Drinks","100g",150,140,440,"winter","cold_up"),
 ("SKU213","Crisp Sparkling 1.5L","Crisp","Beverages","Cold Drinks","1.5L",120,110,780,"summer","warm_up"),
 ("SKU214","Crisp Cola 1.5L","Crisp","Beverages","Cold Drinks","1.5L",130,120,920,"summer","warm_up"),
 ("SKU215","Glacier Ice Cream 1L","Glacier","Frozen","Desserts","1L",360,340,360,"summer","warm_up"),
 ("SKU216","Glacier Frozen Peas 500g","Glacier","Frozen","Vegetables","500g",120,112,420,"none","none"),
 ("SKU217","Harvest Tomato Soup 400g","Harvest","Packaged Food","Ready Meals","400g",100,92,300,"winter","cold_up"),
 ("SKU218","Harvest Baked Beans 400g","Harvest","Packaged Food","Ready Meals","400g",95,88,500,"none","none"),
 ("SKU219","Nomad Muesli 500g","Nomad","Packaged Food","Breakfast","500g",240,225,330,"mild","none"),
 ("SKU220","Nomad Oats 1kg","Nomad","Packaged Food","Breakfast","1kg",260,245,300,"winter","cold_up"),
]
pm = pd.DataFrame(products, columns=["sku_id","sku_name","brand","category","subcategory","pack_size","mrp","base_price","base_units","season","weather_sensitivity"])

base_temp = 15 + 11*np.sin(2*np.pi*(week_no-16)/52) + rng.normal(0,1.0,156)

holiday_map={}
def mark(y,w,n):
    for i,(yy,ww) in enumerate(zip(year,week_no)):
        if yy==y and ww==w: holiday_map[i]=n
for y in (2023,2024,2025):
    mark(y,1,"New Year"); mark(y,13,"Spring Holiday"); mark(y,30,"Summer Sale")
    mark(y,47,"Autumn Festival"); mark(y,51,"Year End"); mark(y,52,"Year End")
holiday_flag=np.array([1 if i in holiday_map else 0 for i in range(156)])

promo_lookup={}
for _,p in pm.iterrows():
    for w0 in sorted(rng.choice(np.arange(2,152),size=rng.integers(10,16),replace=False)):
        dur=int(rng.integers(1,3)); disc=int(rng.choice([10,15,20,25,30,35])); up=1+disc/100*rng.uniform(4,6.5)
        for w in range(w0,min(w0+dur,156)): promo_lookup[(p.sku_id,w)]=(disc,up)

rows=[]
for _,p in pm.iterrows():
    trend=1+0.0009*t
    seas = (1+0.9*np.sin(2*np.pi*(week_no-16)/52) if p.season=="summer" else
            1+0.8*np.sin(2*np.pi*(week_no-42)/52) if p.season=="winter" else
            1+0.13*np.sin(2*np.pi*(week_no-20)/52) if p.season=="mild" else np.ones(156))
    seas=np.clip(seas,0.25,None)
    for _,s in sm.iterrows():
        temp=base_temp+region_temp_off[s.region]
        weath=(1+0.045*(temp-15) if p.weather_sensitivity=="warm_up" else
               1-0.030*(temp-15) if p.weather_sensitivity=="cold_up" else np.ones(156))
        weath=np.clip(weath,0.3,None); snoise=rng.uniform(0.9,1.1)
        for i in range(156):
            disc,up=promo_lookup.get((p.sku_id,i),(0,1.0))
            price=round(p.base_price*(1-disc/100),0) if disc else float(p.base_price)
            comp=round(p.base_price*rng.uniform(0.95,1.08),0)
            hol=(1.25 if p.category in ("Beverages","Packaged Food","Frozen","Personal Care") else 1.10) if holiday_flag[i] else 1.0
            mean=p.base_units*s.size_mult*snoise*trend[i]*seas[i]*weath[i]*up*hol
            mean*=np.clip(1+0.25*((comp-price)/p.base_price),0.6,1.4)
            true=int(rng.poisson(max(mean,1)))
            so=1 if rng.random()<0.03 else 0
            units=int(true*rng.uniform(0.55,0.8)) if so else true
            rows.append({"week_start_date":weeks[i],"year":int(year[i]),"week_no":int(week_no[i]),
                "region":s.region,"store_id":s.store_id,"store_format":s.store_format,
                "sku_id":p.sku_id,"sku_name":p.sku_name,"brand":p.brand,"category":p.category,"subcategory":p.subcategory,
                "units_sold":units,"price":price,"mrp":float(p.mrp),"competitor_price":comp,
                "promo_flag":1 if disc else 0,"discount_pct":disc,
                "avg_temp_c":round(temp[i],1),"holiday_flag":int(holiday_flag[i]),"stockout_flag":so})
df=pd.DataFrame(rows)
df.to_csv(OUT/"raw_dataset.csv", index=False)
print(f"Wrote {OUT/'raw_dataset.csv'}  ({len(df):,} rows)")
