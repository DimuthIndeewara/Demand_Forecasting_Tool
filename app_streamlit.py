"""
app_streamlit.py  —  4-week demand forecast interface (pure Python, live).
Product level (all stores combined). Pick a product, set a promo discount and
a temperature, and see the next 4 weeks forecast update live.

Run:  streamlit run app_streamlit.py
"""
import streamlit as st
import pandas as pd
import altair as alt
import forecast_core as fc

st.set_page_config(page_title="Demand Forecast", page_icon="📈", layout="wide")

HORIZON = 4  # weeks ahead


@st.cache_resource
def _load():
    # build data + model on first run if they are not shipped with the code
    with st.spinner("Preparing the model (first run only, ~30s)…"):
        fc.ensure_ready()
    model = fc.load_model()
    df, product_list = fc.load_history()
    return model, df, product_list

model, df, product_list = _load()

# category -> products, for a cleaner two-step picker
meta = (df.groupby("sku_id").agg(name=("sku_name", "last"),
        category=("category", "last")).reset_index())

st.title("Demand Forecast Explorer")
st.caption("Next 4 weeks · product level (all stores combined) · live LightGBM model · synthetic data")

# ----------------- controls -----------------
left, right = st.columns([1, 2.4], gap="large")

with left:
    st.subheader("Scenario")
    category = st.selectbox("Category", ["All"] + sorted(meta["category"].unique()))
    pool = meta if category == "All" else meta[meta["category"] == category]
    product_name = st.selectbox("Product", sorted(pool["name"]))
    sku = pool[pool["name"] == product_name]["sku_id"].iloc[0]

    st.markdown("**What-if levers**")
    discount = st.slider("Promotion discount (%)", 0, 40, 0, step=5,
                         help="Applied to every week in the 4-week horizon.")
    temp = st.slider("Temperature (°C)", 0, 30, 18, step=1,
                     help="Assumed weekly temperature over the horizon.")
    st.caption("Move the sliders to test a promotion or a weather scenario. "
               "The forecast updates instantly.")

# ----------------- forecast (live) -----------------
hist, fut = fc.forecast_product_horizon(model, df, sku, horizon=HORIZON,
                                        discount_pct=discount, temperature_c=temp)
base_hist, base_fut = fc.forecast_product_horizon(model, df, sku, horizon=HORIZON,
                                                  discount_pct=0, temperature_c=temp)
total = int(fut["units"].sum())
base_total = int(base_fut["units"].sum())
uplift = (total / base_total - 1) * 100 if base_total else 0
mrp = int(round(df[df.sku_id == sku]["price"].iloc[-1] / df[df.sku_id == sku]["price_index"].iloc[-1]))
price = round(mrp * (1 - discount / 100))

with right:
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Next 4 weeks", f"{total:,}", "units total")
    k2.metric("Avg per week", f"{round(total/HORIZON):,}", "units")
    k3.metric("Promo uplift", "—" if discount == 0 else f"+{uplift:.0f}%",
              None if discount == 0 else "vs no promo")
    k4.metric("Price", f"€{price}", None if discount == 0 else f"was €{mrp}")

    # ---- chart: recent actuals + 4-week forecast (and the no-promo path) ----
    h = hist.rename(columns={"units": "value"}); h["series"] = "Actual (recent)"
    f = fut.rename(columns={"units": "value"}); f["series"] = "Forecast (your scenario)"
    # connect the lines: prepend the last actual point to the forecast
    bridge = h.tail(1).copy(); bridge["series"] = "Forecast (your scenario)"
    f = pd.concat([bridge, f])
    plot = pd.concat([h, f])

    base_line = base_fut.rename(columns={"units": "value"}); base_line["series"] = "Forecast (no promo)"
    base_line = pd.concat([h.tail(1).assign(series="Forecast (no promo)"), base_line])
    plot = pd.concat([plot, base_line])

    color = alt.Color("series:N", title="",
                      scale=alt.Scale(domain=["Actual (recent)", "Forecast (your scenario)", "Forecast (no promo)"],
                                      range=["#2C2C2A", "#4b43c4", "#b0b0b0"]))
    dash = alt.StrokeDash("series:N",
                          scale=alt.Scale(domain=["Actual (recent)", "Forecast (your scenario)", "Forecast (no promo)"],
                                          range=[[1, 0], [1, 0], [5, 4]]), legend=None)
    chart = (alt.Chart(plot).mark_line(point=True, strokeWidth=2.5)
             .encode(x=alt.X("date:T", title="Week"),
                     y=alt.Y("value:Q", title="Units / week"),
                     color=color, strokeDash=dash)
             .properties(height=340))
    st.altair_chart(chart, use_container_width=True)

    # ---- the 4 weekly numbers, plainly ----
    tbl = fut.copy()
    tbl["week"] = [f"Week +{i}" for i in range(1, HORIZON + 1)]
    tbl["date"] = pd.to_datetime(tbl["date"]).dt.strftime("%d %b %Y")
    tbl["units"] = tbl["units"].astype(int)
    st.dataframe(tbl[["week", "date", "units"]].rename(
        columns={"week": "", "date": "Week starting", "units": "Forecast units"}),
        hide_index=True, use_container_width=True)

st.divider()
st.caption("Recursive 4-week forecast: the model predicts week 1, feeds that back in to "
           "predict week 2, and so on. Every number is a live model call. In production the same "
           "model would be served through the API in serve_api.py.")
