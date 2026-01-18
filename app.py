import streamlit as st
import pandas as pd
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

st.set_page_config(page_title="AQI Prediction (Random Forest)", page_icon="🌀")

# ---------- AQI Category (Dataset-based) ----------
def aqi_category(aqi: float):
    if aqi <= 20:
        return "Low Pollution", "Air quality is generally safe."
    elif aqi <= 40:
        return "Moderate Pollution", "Sensitive individuals should take care."
    else:
        return "High Pollution", "Reduce outdoor activity if you feel discomfort."

# ---------- Load + Train Model ONCE ----------
@st.cache_resource
def train_rf_model():
    df = pd.read_csv("Dataset/Air_Quality_History.csv")

    # Standardize columns
    df.columns = (
        df.columns.str.strip().str.lower().str.replace(" ", "_")
        .str.replace("(", "", regex=False)
        .str.replace(")", "", regex=False)
        .str.replace("/", "_", regex=False)
    )

    # Filter NO2 records
    df = df[df["parameter_name"].str.contains("nitrogen dioxide", case=False)]

    # Keep only needed columns
    df = df[["datetime_local", "arithmetic_mean", "first_max_value", "aqi"]]

    # Time features
    df["datetime_local"] = pd.to_datetime(df["datetime_local"], errors="coerce")
    df["hour"] = df["datetime_local"].dt.hour
    df["day"] = df["datetime_local"].dt.day
    df["month"] = df["datetime_local"].dt.month

    # Clean
    df = df.dropna().reset_index(drop=True)

    # Rename features
    df.rename(
        columns={"arithmetic_mean": "no2_mean", "first_max_value": "no2_max"},
        inplace=True
    )

    features = ["no2_mean", "no2_max", "hour", "day", "month"]
    X = df[features]
    y = df["aqi"]

    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)

    rf = RandomForestRegressor(n_estimators=200, random_state=42)
    rf.fit(X_train, y_train)

    # Dataset ranges for safe inputs
    ranges = {
        "no2_mean_min": float(df["no2_mean"].min()),
        "no2_mean_max": float(df["no2_mean"].max()),
        "no2_max_min": float(df["no2_max"].min()),
        "no2_max_max": float(df["no2_max"].max()),
    }

    return rf, features, ranges

# ---------- UI ----------
st.title("🌀 AQI Prediction")
st.caption("Model: Random Forest Regression")

st.write(
    "This application predicts **AQI** using **Random Forest Regression**. "
    "You enter NO₂ mean and NO₂ max (from monitoring data), while time features "
    "(hour/day/month) are taken automatically from the system."
)

st.info(
    "**Input source:** NO₂ mean/max are typically taken from air-quality monitoring stations, sensors, or datasets.\n\n"
    "- **NO₂ mean** = average NO₂ concentration over a period\n"
    "- **NO₂ max** = peak (highest) NO₂ value during the same period"
)

rf, features, ranges = train_rf_model()

st.caption(
    f"Dataset ranges — NO₂ mean: {ranges['no2_mean_min']:.1f} to {ranges['no2_mean_max']:.1f} | "
    f"NO₂ max: {ranges['no2_max_min']:.1f} to {ranges['no2_max_max']:.1f}"
)
st.caption("Note: Pollution categories are defined relative to the AQI range in the training dataset.")

st.subheader("Enter NO₂ values")

no2_mean = st.number_input(
    "NO₂ mean (average)",
    min_value=ranges["no2_mean_min"],
    max_value=ranges["no2_mean_max"],
    value=max(ranges["no2_mean_min"], min(25.0, ranges["no2_mean_max"])),
    step=0.1,
)

no2_max = st.number_input(
    "NO₂ max (peak)",
    min_value=ranges["no2_max_min"],
    max_value=ranges["no2_max_max"],
    value=max(ranges["no2_max_min"], min(35.0, ranges["no2_max_max"])),
    step=0.1,
)

# System time features
now = datetime.now()
hour, day, month = now.hour, now.day, now.month
st.caption(f"⏱️ Auto time features: hour={hour}, day={day}, month={month}")

if st.button("Predict AQI"):
    input_df = pd.DataFrame([[no2_mean, no2_max, hour, day, month]], columns=features)
    predicted_aqi = float(rf.predict(input_df)[0])

    category, advice = aqi_category(predicted_aqi)

    st.success(f"Predicted AQI: {predicted_aqi:.2f}")
    st.info(f"Category: {category}")
    st.warning(f"Health Advice: {advice}")
