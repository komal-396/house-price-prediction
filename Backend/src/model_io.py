from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

from .constants import FEATURE_COLUMNS


def save_model_package(model: Any, model_path: Path) -> None:
    model_path.parent.mkdir(parents=True, exist_ok=True)
    package = {
        "model": model,
        "feature_columns": FEATURE_COLUMNS,
    }
    with model_path.open("wb") as f:
        pickle.dump(package, f)


def load_model_package(model_path: Path) -> dict[str, Any]:
    with model_path.open("rb") as f:
        loaded = pickle.load(f)

    # Backward compatibility with older plain-model pickle files.
    if isinstance(loaded, dict) and "model" in loaded:
        return loaded

    return {
        "model": loaded,
        "feature_columns": FEATURE_COLUMNS,
    }
