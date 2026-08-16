from pathlib import Path

from src.train_pipeline import run_training


if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent
    data_path = base_dir / "Housing.csv"
    model_path = base_dir / "model" / "house_price_model.pkl"

    run_training(data_path=data_path, model_path=model_path, show_plots=True)