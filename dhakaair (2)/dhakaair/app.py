"""
DhakaAir — PM2.5 / AQI Trend, Location-Aware Prediction & 7-Day Forecast
Trained on real Dhaka air-quality station data via OpenAQ:
  - US Embassy Baridhara reference station (2016-2025, long history, city-wide model)
  - Multi-station low-cost sensor network across 7 Dhaka neighbourhoods (2025-2026, location-aware model)
"""

import base64

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------
st.set_page_config(page_title="DhakaAir", page_icon="dhakaair_logo.png", layout="wide")


@st.cache_data
def load_logo_b64():
    with open("dhakaair_logo.png", "rb") as f:
        return base64.b64encode(f.read()).decode()


LOGO_B64 = load_logo_b64()

# City-wide long-history model (Linear Regression — see Model Info tab for why)
FEATURES = joblib.load("features_final.pkl")
MODEL = joblib.load("model_final.pkl")
METRICS = joblib.load("metrics.pkl")

# Neighbourhood-aware model (Random Forest, trained on multi-station data)
LOC_FEATURES = joblib.load("location_features.pkl")
LOC_MODEL = joblib.load("model_location.pkl")
LOC_AREAS = joblib.load("location_areas.pkl")
LOC_METRICS = joblib.load("location_metrics.pkl")

# Comparison of several model types on the long-history dataset
MODEL_COMPARISON = joblib.load("model_comparison.pkl")

BASE_FEATURES = ['month', 'day_of_week', 'is_weekend', 'is_brick_kiln_season',
                  'temp_mean', 'humidity_mean', 'wind_speed_mean', 'pressure_mean',
                  'precip_sum', 'is_rainy_day', 'blh_mean', 'ventilation_index',
                  'pm25_lag1', 'pm25_lag2', 'pm25_lag3', 'pm25_roll7_mean', 'pm25_roll30_mean']


@st.cache_data
def load_data():
    df = pd.read_csv("app_data.csv")
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").reset_index(drop=True)


@st.cache_data
def load_area_history():
    df = pd.read_csv("area_history.csv")
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values(["area", "date"]).reset_index(drop=True)


DATA = load_data()
AREA_HIST = load_area_history()

AQI_BREAKPOINTS = [
    (0, 50, "Good", "#00e400"),
    (51, 100, "Moderate", "#ffff00"),
    (101, 150, "Unhealthy for Sensitive Groups", "#ff7e00"),
    (151, 200, "Unhealthy", "#ff0000"),
    (201, 300, "Very Unhealthy", "#8f3f97"),
    (301, 500, "Hazardous", "#7e0023"),
]

ADVICE = {
    "Good": "Air quality is fine. Enjoy outdoor activities as usual.",
    "Moderate": "Acceptable air quality. Unusually sensitive people should consider limiting prolonged outdoor exertion.",
    "Unhealthy for Sensitive Groups": "Children, elderly, and people with asthma or heart/lung conditions should reduce prolonged outdoor exertion. A mask (N95) helps outdoors.",
    "Unhealthy": "Everyone may experience health effects. Limit outdoor exertion. Sensitive groups should avoid it. Wear a mask outdoors.",
    "Very Unhealthy": "Health alert. Avoid outdoor exertion. Keep windows closed. Use an air purifier indoors if possible.",
    "Hazardous": "Health emergency. Stay indoors, avoid all outdoor activity, and wear a well-fitted N95/KN95 mask if you must go out.",
}


def aqi_category(aqi):
    for lo, hi, cat, color in AQI_BREAKPOINTS:
        if lo <= aqi <= hi:
            return cat, color
    return "Hazardous", "#7e0023"


def pm25_to_aqi(pm25):
    """Simplified US EPA PM2.5 -> AQI conversion (2024 breakpoints, approx)."""
    bps = [
        (0.0, 9.0, 0, 50), (9.1, 35.4, 51, 100), (35.5, 55.4, 101, 150),
        (55.5, 125.4, 151, 200), (125.5, 225.4, 201, 300), (225.5, 500.4, 301, 500),
    ]
    pm25 = max(0.0, min(pm25, 500.4))
    for c_lo, c_hi, a_lo, a_hi in bps:
        if c_lo <= pm25 <= c_hi:
            return round(a_lo + (a_hi - a_lo) / (c_hi - c_lo) * (pm25 - c_lo))
    return 500


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown(
    f"""
    <div style="
        background: linear-gradient(90deg, #00e400 0%, #ffff00 25%, #ff7e00 50%, #ff0000 75%, #8f3f97 100%);
        padding: 14px 24px; border-radius: 14px; margin-bottom: 10px;
        display: flex; align-items: center; gap: 16px;">
        <img src="data:image/png;base64,{LOGO_B64}" style="height: 56px; width: 56px; border-radius: 50%;
             box-shadow: 0 2px 8px rgba(0,0,0,0.35);">
        <span style="font-size: 34px; font-weight: 800; color: white; text-shadow: 1px 1px 4px rgba(0,0,0,0.5);">
            DhakaAir
        </span>
    </div>
    """,
    unsafe_allow_html=True,
)
st.caption(
    "Historical PM2.5/AQI trends, a neighbourhood-aware prediction model, and a 7-day "
    "forecast for Dhaka — trained on real air-quality station data via OpenAQ, 2016–2026."
)

tab1, tab2, tab3 = st.tabs(["📈 Trends", "🔮 Predict & Forecast", "📊 Model Info"])

# ---------------------------------------------------------------------------
# Tab 1 — Trends
# ---------------------------------------------------------------------------
with tab1:
    col1, col2 = st.columns([1, 3])
    with col1:
        years = sorted(DATA["date"].dt.year.unique())
        year_sel = st.multiselect("Year", years, default=years[-3:])
        season_sel = st.multiselect(
            "Season", sorted(DATA["season"].unique()), default=list(DATA["season"].unique())
        )

    filtered = DATA[DATA["date"].dt.year.isin(year_sel) & DATA["season"].isin(season_sel)]

    with col2:
        latest = DATA.iloc[-1]
        cat, color = aqi_category(latest["aqi_pm25"])
        m1, m2, m3 = st.columns(3)
        m1.metric("Latest recorded PM2.5", f"{latest['pm25']:.0f} µg/m³", latest["date"].strftime("%d %b %Y"))
        m2.metric("Latest AQI", f"{latest['aqi_pm25']:.0f}", cat)
        m3.metric("Days in dataset", f"{len(DATA):,}")

    if filtered.empty:
        st.warning("No data for the selected filters.")
    else:
        fig = px.line(filtered, x="date", y="pm25", title="Daily PM2.5 (µg/m³) — Dhaka")
        fig.add_hline(y=35.4, line_dash="dot", line_color="orange", annotation_text="Unhealthy for Sensitive Groups")
        fig.add_hline(y=55.4, line_dash="dot", line_color="red", annotation_text="Unhealthy")
        st.plotly_chart(fig, use_container_width=True)

        c1, c2 = st.columns(2)
        with c1:
            monthly = DATA.groupby("month")["pm25"].mean().reset_index()
            fig2 = px.bar(monthly, x="month", y="pm25", title="Average PM2.5 by Month (all years)")
            st.plotly_chart(fig2, use_container_width=True)
        with c2:
            seasonal = DATA.groupby("season")["pm25"].mean().reset_index()
            fig3 = px.bar(seasonal, x="season", y="pm25", title="Average PM2.5 by Season", color="season")
            st.plotly_chart(fig3, use_container_width=True)

        st.info(
            "Dhaka's air quality is worst in **winter and the brick-kiln season "
            "(Nov–Apr)**, and improves during the monsoon when rain washes out particulates."
        )
        st.caption(
            "2016–2025 data is from the US Embassy reference-grade station (Baridhara). "
            "2025 onward is a daily average across several low-cost sensors placed around Dhaka "
            "(Uttara, Mirpur, Dhanmondi, Gulshan, Badda, Ramna, Hazaribagh, Baridhara)."
        )

# ---------------------------------------------------------------------------
# Tab 2 — Predict & Forecast
# ---------------------------------------------------------------------------
with tab2:
    st.subheader("Predict tomorrow's PM2.5 & AQI")

    area_choice = st.selectbox(
        "Location",
        ["Dhaka Average (10-year city-wide model)"] + LOC_AREAS,
        help="City-wide uses 10 years of reference-station history. Neighbourhood models use "
             "a shorter (6-12 month) multi-sensor history, so treat them as a rougher estimate.",
    )
    use_location = area_choice != "Dhaka Average (10-year city-wide model)"

    if use_location:
        area_df = AREA_HIST[AREA_HIST["area"] == area_choice].sort_values("date")
        last = area_df.iloc[-1]
        st.caption(
            f"Neighbourhood model for **{area_choice}** — based on {len(area_df)} days of "
            f"low-cost sensor data (latest: {last['date'].strftime('%d %b %Y')})."
        )
    else:
        last = DATA.iloc[-1]

    st.caption(
        "Defaults are pre-filled from the most recent recorded day. Adjust the weather "
        "forecast values below (e.g. from a weather app) to see how air quality might change."
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        temp = st.slider("Forecast temperature (°C)", 5.0, 40.0, float(last["temp_mean"]), 0.5)
        humidity = st.slider("Forecast humidity (%)", 10.0, 100.0, float(last["humidity_mean"]), 1.0)
    with c2:
        wind = st.slider("Forecast wind speed (km/h)", 0.0, 40.0, float(last["wind_speed_mean"]), 0.5)
        pressure = st.slider("Forecast pressure (hPa)", 995.0, 1025.0, float(last["pressure_mean"]), 0.5)
    with c3:
        precip = st.slider("Forecast rainfall (mm)", 0.0, 100.0, float(last["precip_sum"]), 1.0)
        month = st.selectbox("Month", list(range(1, 13)), index=int(last["month"]) - 1)

    is_rainy = 1 if precip > 1 else 0
    is_brick_kiln = 1 if month in [11, 12, 1, 2, 3, 4] else 0
    day_of_week = int((last["day_of_week"] + 1) % 7)
    is_weekend = 1 if day_of_week in [4, 5] else 0  # Bangladesh weekend: Fri/Sat
    blh = float(last["blh_mean"])
    ventilation = blh * wind

    def build_row(lag1, lag2, lag3, roll7, roll30):
        base = {
            "month": month, "day_of_week": day_of_week, "is_weekend": is_weekend,
            "is_brick_kiln_season": is_brick_kiln, "temp_mean": temp, "humidity_mean": humidity,
            "wind_speed_mean": wind, "pressure_mean": pressure, "precip_sum": precip,
            "is_rainy_day": is_rainy, "blh_mean": blh, "ventilation_index": ventilation,
            "pm25_lag1": lag1, "pm25_lag2": lag2, "pm25_lag3": lag3,
            "pm25_roll7_mean": roll7, "pm25_roll30_mean": roll30,
        }
        if use_location:
            for a in LOC_AREAS:
                base[f"area_{a}"] = 1 if a == area_choice else 0
            return pd.DataFrame([base])[LOC_FEATURES]
        return pd.DataFrame([base])[FEATURES]

    active_model = LOC_MODEL if use_location else MODEL

    row = build_row(last["pm25"], last["pm25_lag1"], last["pm25_lag2"],
                     last["pm25_roll7_mean"], last["pm25_roll30_mean"])
    pred_pm25 = max(0.0, active_model.predict(row)[0])
    pred_aqi = pm25_to_aqi(pred_pm25)
    cat, color = aqi_category(pred_aqi)

    st.divider()
    r1, r2 = st.columns([1, 2])
    with r1:
        st.metric("Predicted PM2.5 (tomorrow)", f"{pred_pm25:.0f} µg/m³")
        st.metric("Predicted AQI", f"{pred_aqi}")
        st.markdown(
            f"<div style='background-color:{color};padding:10px;border-radius:8px;"
            f"text-align:center;font-weight:bold;color:black'>{cat}</div>",
            unsafe_allow_html=True,
        )
    with r2:
        st.markdown("**Health advice**")
        st.write(ADVICE[cat])
        st.caption(
            "This is a statistical estimate based on historical patterns, weather inputs, "
            "and recent PM2.5 levels — not an official government forecast."
        )

    # -- 7-day recursive forecast -------------------------------------------
    st.divider()
    st.subheader("📅 7-day forecast")
    st.caption(
        "Assumes the weather conditions set above hold for the whole week. Each day's prediction "
        "feeds into the next, so uncertainty grows further into the week — treat days 4-7 as rough."
    )

    lag1, lag2, lag3 = float(last["pm25"]), float(last["pm25_lag1"]), float(last["pm25_lag2"])
    roll30 = float(last["pm25_roll30_mean"])
    recent_window = [lag3, lag2, lag1]  # oldest to newest, used to roll the 7-day mean forward

    forecast_dates, forecast_vals, forecast_cats = [], [], []
    base_date = last["date"]
    for i in range(7):
        roll7 = float(np.mean(recent_window[-7:]))
        f_row = build_row(lag1, lag2, lag3, roll7, roll30)
        pred = max(0.0, active_model.predict(f_row)[0])

        forecast_dates.append(base_date + pd.Timedelta(days=i + 1))
        forecast_vals.append(pred)
        f_cat, _ = aqi_category(pm25_to_aqi(pred))
        forecast_cats.append(f_cat)

        lag3, lag2, lag1 = lag2, lag1, pred
        recent_window.append(pred)
        roll30 = roll30 * 29 / 30 + pred / 30

    forecast_df = pd.DataFrame({"date": forecast_dates, "pm25": forecast_vals, "category": forecast_cats})

    fig_fc = go.Figure()
    fig_fc.add_trace(go.Scatter(
        x=forecast_df["date"], y=forecast_df["pm25"], mode="lines+markers",
        line=dict(color="#ff7e00", width=3), marker=dict(size=9),
    ))
    fig_fc.add_hline(y=35.4, line_dash="dot", line_color="orange", annotation_text="Unhealthy for Sensitive Groups")
    fig_fc.add_hline(y=55.4, line_dash="dot", line_color="red", annotation_text="Unhealthy")
    fig_fc.update_layout(title=f"7-day PM2.5 forecast — {area_choice}", yaxis_title="PM2.5 (µg/m³)")
    st.plotly_chart(fig_fc, use_container_width=True)

    cols = st.columns(7)
    for i, c in enumerate(cols):
        _, color_i = aqi_category(pm25_to_aqi(forecast_df["pm25"].iloc[i]))
        c.markdown(
            f"<div style='background-color:{color_i};padding:8px;border-radius:8px;text-align:center;color:black'>"
            f"<b>{forecast_df['date'].iloc[i].strftime('%a %d')}</b><br>{forecast_df['pm25'].iloc[i]:.0f} µg/m³</div>",
            unsafe_allow_html=True,
        )

# ---------------------------------------------------------------------------
# Tab 3 — Model info
# ---------------------------------------------------------------------------
with tab3:
    st.subheader("City-wide model performance")
    c1, c2 = st.columns(2)
    c1.metric("Mean Absolute Error", f"{METRICS['mae']:.1f} µg/m³")
    c2.metric("R² score", f"{METRICS['r2']:.2f}")

    st.markdown(
        f"""
- **Data source:** OpenAQ — US Embassy Dhaka reference-grade station, Baridhara (2016–2025).
  The **Trends** tab also blends in 2025–2026 multi-station low-cost sensor averages across Dhaka.
- **Rows used for training:** {METRICS['n_train'] + METRICS['n_test']:,} daily records
  ({METRICS['n_train']:,} train / {METRICS['n_test']:,} test, split chronologically — no shuffling, to avoid leaking future data into training)
- **Model used:** {METRICS['model_name']} — selected because it had the best held-out accuracy
  among the models compared below, not because it's the most complex one.
        """
    )

    st.subheader("Model comparison (same train/test split)")
    comp_df = pd.DataFrame(
        [{"Model": k, "MAE (µg/m³)": v[0], "R²": v[1]} for k, v in MODEL_COMPARISON.items()]
    ).sort_values("MAE (µg/m³)")
    st.dataframe(comp_df.style.format({"MAE (µg/m³)": "{:.2f}", "R²": "{:.3f}"}), use_container_width=True)
    fig_cmp = px.bar(comp_df, x="Model", y="MAE (µg/m³)", color="Model", title="Lower is better")
    fig_cmp.update_layout(showlegend=False)
    st.plotly_chart(fig_cmp, use_container_width=True)
    st.caption(
        "Linear Regression outperformed the tree-based models here — a reminder that a simpler "
        "model can beat a more complex one when the relationship between weather/lag features "
        "and next-day PM2.5 is largely linear and the dataset isn't huge."
    )

    st.subheader(f"What the city-wide model relies on")
    if hasattr(MODEL, "feature_importances_"):
        imp = pd.Series(MODEL.feature_importances_, index=FEATURES).sort_values(ascending=False)
        imp_label = "Importance"
    else:
        imp = pd.Series(np.abs(MODEL.coef_), index=FEATURES).sort_values(ascending=False)
        imp_label = "|Coefficient| (relative weight)"
    fig_imp = px.bar(imp, orientation="h", title=imp_label)
    fig_imp.update_layout(showlegend=False, yaxis_title="", xaxis_title=imp_label)
    st.plotly_chart(fig_imp, use_container_width=True)

    st.divider()
    st.subheader("Neighbourhood (location-aware) model")
    lc1, lc2, lc3 = st.columns(3)
    lc1.metric("Mean Absolute Error", f"{LOC_METRICS['mae']:.1f} µg/m³")
    lc2.metric("R² score", f"{LOC_METRICS['r2']:.2f}")
    lc3.metric("Areas covered", f"{len(LOC_METRICS['areas'])}")
    st.markdown(
        f"- **Model:** {LOC_METRICS['model_name']}, with neighbourhood as a feature "
        f"({', '.join(LOC_METRICS['areas'])})\n"
        f"- **Rows:** {LOC_METRICS['n_train']:,} train / {LOC_METRICS['n_test']:,} test "
        f"— only 6-12 months of history per neighbourhood (vs. 10 years for the city-wide model)"
    )
    st.warning(
        "**Honest limitation:** with only 6-12 months of data per neighbourhood, this model is "
        "close to a naive 'tomorrow = today' guess and doesn't yet reliably beat it on every split. "
        "It's included to make the point meaningful — real hyperlocal air-quality prediction needs "
        "at least 1-2 full seasonal cycles per station. Treat its predictions as a rough signal, "
        "not a precise forecast, until more data accumulates."
    )

    st.warning(
        "**City-wide model limitation:** trained on one station (Baridhara, a diplomatic-zone "
        "location), so it may not reflect industrial or high-traffic areas as accurately. It also "
        "can't see sudden events (fires, dust storms) that aren't in the historical pattern."
    )

st.divider()
st.caption("Built with Python, scikit-learn, Plotly & Streamlit · Data via OpenAQ")
