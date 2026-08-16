# House Price Prediction

This project trains a house-price regression model and serves predictions through a Streamlit app.

## Project Structure

```
Project 1- House Prediction/
	Backend/
		app.py
		traintest.py
		Housing.csv
		model/
			house_price_model.pkl
		src/
			__init__.py
			constants.py
			preprocess.py
			model_io.py
			train_pipeline.py
	requirements.txt
	README.md
```

## What Each File Does

- `Backend/traintest.py`: entry point for model training and evaluation.
- `Backend/app.py`: Streamlit UI for user input and prediction.
- `Backend/src/preprocess.py`: encoding logic for training and inference.
- `Backend/src/train_pipeline.py`: data loading, training, metrics, and plotting.
- `Backend/src/model_io.py`: save/load helpers for model package.
- `Backend/src/constants.py`: shared feature lists and order.

## Setup

1. Create and activate a virtual environment.
2. Install dependencies:

```powershell
pip install -r requirements.txt
```

## Train the Model

Run from the `Backend` folder:

```powershell
python traintest.py
```

This command will:

- load and preprocess data,
- train a linear regression model,
- print MAE, MSE, and R2,
- save the trained model to `Backend/model/house_price_model.pkl`.

## Run the App

Run from the `Backend` folder:

```powershell
streamlit run app.py
```

## Notes

- Keep training and app feature order consistent (handled by shared `src` modules).
- Retrain the model after changing feature engineering logic.
