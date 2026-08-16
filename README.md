# House Price Prediction

A machine learning project that predicts house prices based on property features such as area, number of bedrooms, bathrooms, parking, story count, furnishing status, and nearby/home feature preferences.

This project combines a regression model with a Streamlit web interface so users can estimate a property price from a few inputs.

## Project Overview

This project was built to explore the end-to-end workflow of a predictive ML application:

- data preprocessing and feature engineering
- model training and evaluation
- model persistence
- deployment as a simple interactive user-facing app

## Tech Stack

- Python
- Pandas
- NumPy
- scikit-learn
- Matplotlib
- Streamlit

## Project Structure

```text
Project 1- House Prediction/
├── Backend/
│   ├── app.py
│   ├── traintest.py
│   ├── Housing.csv
│   ├── model/
│   │   └── house_price_model.pkl
│   └── src/
│       ├── __init__.py
│       ├── constants.py
│       ├── model_io.py
│       ├── preprocess.py
│       └── train_pipeline.py
├── requirements.txt
├── README.md
└── .gitignore
```

## Key Components

- `Backend/traintest.py`: trains the model and evaluates performance
- `Backend/app.py`: interactive Streamlit UI for predictions
- `Backend/src/preprocess.py`: handles data transformation and feature encoding
- `Backend/src/train_pipeline.py`: defines model training, evaluation, and comparison logic
- `Backend/src/model_io.py`: saves and loads the trained model package
- `Backend/src/constants.py`: stores feature names and column mappings

## Features

- Home price prediction using property characteristics
- Multiple regression model comparison (Linear Regression, Random Forest, Gradient Boosting)
- Feature engineering for categorical home attributes
- Streamlit interface for user-friendly prediction input

## Setup

1. Create and activate a virtual environment.
2. Install dependencies:

```powershell
pip install -r requirements.txt
```

## Train the Model

From the `Backend` folder, run:

```powershell
python traintest.py
```

This script will:

- load the dataset
- preprocess the data
- split the data into train and test sets
- train and compare models
- evaluate metrics such as MAE, MSE, and R²
- save the best-performing model to `Backend/model/house_price_model.pkl`

## Run the App

From the `Backend` folder, run:

```powershell
streamlit run app.py
```

The app allows users to input home preferences and receive an estimated price.

## Model Notes

- Training and inference use consistent feature ordering to avoid mismatches.
- Retrain the model whenever feature engineering or feature columns change.
- The best-performing model is selected based on the strongest validation score during training.

## Portfolio Notes

This project is a strong beginner-to-intermediate machine learning portfolio project because it demonstrates:

- Python-based ML workflow
- data preprocessing
- model evaluation
- practical deployment through an app
- a clean end-to-end project structure

## License

This project is intended for learning and portfolio use. Add a license if you plan to share it publicly beyond personal use.

## Future Improvements

- add more advanced model tuning
- include more data features and external market signals
- improve UI styling and app UX
- add deployment to a cloud service or web hosting platform
- add tests and automated validation
