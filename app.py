"""Streamlit web app for the California Housing Price Predictor.

Run locally:  streamlit run app.py
"""
import json
import sys
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.append(str(PROJECT_ROOT / "src"))

from feature_engineering import engineer_features  # noqa: E402

MODEL_DIR = PROJECT_ROOT / "outputs" / "models"

st.set_page_config(page_title="CA Housing Price Predictor", page_icon="🏠",
                   layout="centered")


@st.cache_resource
def load_model():
    model = joblib.load(MODEL_DIR / "best_model.pkl")
    with open(MODEL_DIR / "preprocessing.json") as f:
        stats = json.load(f)
    return model, stats


model, stats = load_model()

st.title("🏠 California Housing Price Predictor")
st.caption(
    f"Predicts a district's median house value with a trained "
    f"**{stats['model']}** model (1990 census data, Kaggle "
    f"`camnugent/california-housing-prices`)."
)

# ---------------- Sidebar inputs ----------------
st.sidebar.header("District details")

median_income = st.sidebar.slider(
    "Median household income ($)", 5_000, 150_000, 35_000, step=1_000
)
housing_median_age = st.sidebar.slider("Median house age (years)", 1, 52, 25)
ocean_proximity = st.sidebar.selectbox(
    "Ocean proximity",
    ["<1H OCEAN", "INLAND", "NEAR BAY", "NEAR OCEAN", "ISLAND"],
)

st.sidebar.subheader("Size of the district")
total_rooms = st.sidebar.number_input("Total rooms", 100, 40_000, 2_500, step=100)
total_bedrooms = st.sidebar.number_input("Total bedrooms", 50, 7_000, 500, step=50)
population = st.sidebar.number_input("Population", 50, 36_000, 1_400, step=50)
households = st.sidebar.number_input("Households", 50, 6_000, 450, step=50)

st.sidebar.subheader("Location")
latitude = st.sidebar.slider("Latitude", 32.5, 42.0, 34.05, step=0.01)
longitude = st.sidebar.slider("Longitude", -124.5, -114.3, -118.24, step=0.01)

# ---------------- Prediction ----------------
district = pd.DataFrame(
    [{
        "longitude": longitude,
        "latitude": latitude,
        "housing_median_age": housing_median_age,
        "total_rooms": total_rooms,
        "total_bedrooms": total_bedrooms,
        "population": population,
        "households": households,
        "median_income": median_income / 10_000,  # model expects $10k units
        "ocean_proximity": ocean_proximity,
    }]
)

features = engineer_features(district, bedrooms_median=stats["bedrooms_median"])
price = float(model.predict(features)[0])
price = max(price, 0)

st.metric("Predicted median house value", f"${price:,.0f}")

col1, col2, col3 = st.columns(3)
col1.metric("Rooms / household", f"{total_rooms / households:.1f}")
col2.metric("People / household", f"{population / households:.1f}")
col3.metric("Income", f"${median_income:,.0f}")

st.subheader("District location")
st.map(pd.DataFrame({"lat": [latitude], "lon": [longitude]}), zoom=6)

with st.expander("How this works"):
    st.markdown(
        """
        - Trained on 20,640 California districts from the 1990 census.
        - Features include income, house age, room/occupancy ratios,
          one-hot ocean proximity, and distance to LA / SF / San Diego /
          Sacramento.
        - Best model selected from Linear Regression, Ridge, Random Forest,
          and Gradient Boosting by test RMSE.
        - Note: values reflect 1990 prices and are capped at $500k in the
          training data.
        """
    )