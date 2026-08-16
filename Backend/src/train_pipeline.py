from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

'''For linear regression model, we will use the following features:'''
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

'''For random forest regression model, we will use the following features:'''
from sklearn.ensemble import RandomForestRegressor

'''For Gradient Boosting Regression model:'''
from sklearn.ensemble import GradientBoostingRegressor

from .constants import FEATURE_COLUMNS
from .model_io import save_model_package
from .preprocess import encode_training_dataframe


def run_training(data_path: Path, model_path: Path, show_plots: bool = True) -> dict[str, float]:
    data = pd.read_csv(data_path)

    print(data.head())
    print(data.shape)
    print(data.isnull().sum())
    print(data.describe())

    encoded = encode_training_dataframe(data)
    print(encoded.columns)

    X = encoded[FEATURE_COLUMNS]
    y = encoded[["price"]]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )

    # model = LinearRegression()
    # model.fit(X_train, y_train)

    #Linear Regression model comparision:
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    lr_preds = lr_model.predict(X_test)
    lr_r2 = r2_score(y_test, lr_preds)
    lr_mae = mean_absolute_error(y_test, lr_preds)
    print(f"Linear Regression - R2 score: {lr_r2:.4f} | MAE : {lr_mae:,.0f}")

# Random Forest Regression
    rf_model = RandomForestRegressor(
        n_estimators = 100,
        max_depth= 10,
        random_state = 42,
        oob_score = True,
    )
    rf_model.fit(X_train, y_train.values.ravel())
    rf_preds = rf_model.predict(X_test)
    rf_r2 = r2_score(y_test, rf_preds)
    rf_mae = mean_absolute_error(y_test,rf_preds)
    print(f"Random Forest Regression - R2 Score: {rf_r2:.4f} | MAE : {rf_mae:,.0f}")
    print(f"OOB Score: {rf_model.oob_score_:.4f}")

    '''Gradient Boosting Regression model comparision:'''
    gb_model = GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        random_state=42,
    )
    gb_model.fit(X_train, y_train.values.ravel())
    gb_preds = gb_model.predict(X_test)
    gb_r2 = r2_score(y_test, gb_preds)
    gb_mae = mean_absolute_error(y_test, gb_preds)
    print(f"Gradient Boosting Regression - R2 Score: {gb_r2:.4f} | MAE : {gb_mae:,.0f}")


    '''Feature importance analysis:'''
    importances = pd.Series(rf_model.feature_importances_,
                            index= FEATURE_COLUMNS).sort_values(ascending=False)
    print("\n Feature Importances:")
    print(importances)

    '''Choose best model based on R2 score and MAE:'''
    best_r2 = max(lr_r2, rf_r2, gb_r2)

    if best_r2 == gb_r2:
        model = gb_model
        predictions = gb_preds
        print("\n Saving: Gradient Boosting Regression model as it has better R2 score")
    elif best_r2 == rf_r2:
        model = rf_model
        predictions = rf_preds
        print("\n Saving: Random Forest Regression model as it has better R2 score")
    else:
        model = lr_model
        predictions = lr_preds
        print("\n Saving: Linear Regression model as it has better R2 score")



    save_model_package(model, model_path)
    print(f"Model saved successfully at: {model_path}")

    predictions = model.predict(X_test)
    print(predictions[:5])

    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    print(
        "Metrics are:",
        f"Mean Absolute Error: {mae}",
        f"Mean Squared Error: {mse}",
        f"R2 Score: {r2}",
        sep="\n",
    )

    comparison = pd.DataFrame({"Actual": y_test["price"], "Predicted": predictions.flatten()})
    print(comparison.head())

    if show_plots:
        _plot_area_vs_price(encoded)
        _plot_actual_vs_predicted(y_test["price"], predictions.flatten())

    return {
        "mae": mae,
        "mse": mse,
        "r2": r2,
    }


def _plot_area_vs_price(data: pd.DataFrame) -> None:
    kg = plt.gca()
    kg.ticklabel_format(style="plain", axis="both")
    plt.scatter(data["area"], data["price"])
    plt.xlabel("AREA")
    plt.ylabel("PRICE")
    plt.title("AREA VS PRICE")
    kg.format_coord = lambda x, y: f"x={x:.0f}, y={y:,.0f}"
    plt.show()


def _plot_actual_vs_predicted(actual: pd.Series, predicted: pd.Series) -> None:
    plt.scatter(actual, predicted, alpha=0.5)
    plt.plot([actual.min(), actual.max()], [actual.min(), actual.max()], color="green", lw=2)
    plt.xlabel("Actual Prices")
    plt.ylabel("Predicted Prices")
    plt.title("Actual vs Predicted Prices")
    plt.ticklabel_format(style="plain", axis="both")
    plt.show()


def _plot_feature_importance(importances: pd.Series) -> None:
    importances.plot(kind="bar")
    plt.xlabel("Importance Score")
    plt.ylabel("Features Importances(RF)")
    plt.tight_layout()
    plt.show()


