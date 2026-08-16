import streamlit as st
from pathlib import Path

from src.constants import YES_NO_COLUMNS
from src.model_io import load_model_package
from src.preprocess import build_inference_row

st.set_page_config(page_title="House Price Predictor", page_icon="🏠")
st.title("House Price Predictor")
st.caption("Estimate a home's value based on your preferred property features.")

model_path = Path(__file__).resolve().parent / "model" / "house_price_model.pkl"
model_package = load_model_package(model_path)
model = model_package["model"]

area = st.number_input(
    "Total area of the property (sq ft)",
    min_value=0.0,
    value=3000.0,
    step=100.0,
    format="%.0f",
)

bedrooms = st.number_input(
    "Number of bedrooms",
    min_value=0,
    value=3,
    step=1,
)

bathrooms = st.number_input(
    "Number of bathrooms",
    min_value=0,
    value=2,
    step=1,
)

parking = st.number_input(
    "Parking spaces",
    min_value=0,
    max_value=3,
    value=1,
    step=1,
)

stories = st.number_input(
    "Number of floors/stories",
    min_value=0,
    max_value=4,
    value=2,
    step=1,
)

furnishingstatus = st.selectbox(
    "Furnishing status",
    ("Unfurnished", "Semi-furnished", "Furnished"),
)

yes_no_values = {}
for col in YES_NO_COLUMNS:
    yes_no_values[col] = st.selectbox(
        f"{col.replace('_', ' ').title()} included?",
        ("No", "Yes"),
    )

if st.button("Predict Home Price"):
    model_input = build_inference_row(
        area=area,
        bedrooms=bedrooms,
        stories=stories,
        bathrooms=bathrooms,
        parking=parking,
        furnishingstatus=furnishingstatus,
        yes_no_values=yes_no_values,
    )
    result = model.predict(model_input)
    st.success(f"Estimated Home Price: {float(result[0]):,.0f} INR")