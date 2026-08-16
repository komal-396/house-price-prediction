import streamlit as st
from pathlib import Path

from src.constants import YES_NO_COLUMNS
from src.model_io import load_model_package
from src.preprocess import build_inference_row

model_path = Path(__file__).resolve().parent / "model" / "house_price_model.pkl"
model_package = load_model_package(model_path)
model = model_package["model"]

area = st.number_input(
    "Enter the area of house you are dreaming about, Uh I mean area that you can afford !",
    min_value=0.0,
    value=3000.0,
    step=100.0,
    format="%.0f",
)
bedrooms = st.number_input(
    "How many bedrooms do you want in your home, Human, \n Maybe an extra for unwanted guests? ",
    min_value=0,
    value=3,
    step=1,
)
bathrooms = st.number_input(
    " Think how many bathrooms you need?",
    min_value=0,
    value=2,
    step=1
)
parking = st.number_input(
    "How many parking spaces do you want? \n If you don't want any, enter 0. \n If you want 1, enter 1. \n If you want more than 1, enter 2.",
    min_value=0,
    max_value=3,
    value=1,
    step=1
)
stories = st.number_input(
    "How many stories do you want in your home?",
    min_value=0,
    max_value=4,
    value=2,
    step=1
)

furnishingstatus = st.selectbox(
    "What is your furnishing status preference?",
    ("Unfurnished", "Semi-furnished", "Furnished")
)

yes_no_values = {}
for col in YES_NO_COLUMNS:
    yes_no_values[col] = st.selectbox(
        f"Do you want {col.replace('_', ' ')}?",
        ("No", "Yes")
    )


if st.button("PREDICT YOUR Future I mean JUST HOME for now :D "):
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
    # st.success(f"Your Dream Home Price is: {result[0][0]:,.0f} INR")
    st.success(f"Your Dream Home Price is: {float(result[0]):,.0f} INR")