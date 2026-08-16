from __future__ import annotations

import pandas as pd

from .constants import FEATURE_COLUMNS, YES_NO_COLUMNS


def encode_training_dataframe(data: pd.DataFrame) -> pd.DataFrame:
    """Encode raw training data into numeric features used by the model."""
    encoded = data.copy()

    for col in YES_NO_COLUMNS:
        encoded[col] = encoded[col].map({"yes": 1, "no": 0})

    encoded = pd.get_dummies(
        encoded,
        columns=["furnishingstatus"],
        drop_first=True,
    )

    # Ensure expected one-hot columns always exist.
    for col in FEATURE_COLUMNS:
        if col not in encoded.columns:
            encoded[col] = 0

    return encoded


def build_inference_row(
    *,
    area: float,
    bedrooms: int,
    stories: int,
    bathrooms: int,
    parking: int,
    furnishingstatus: str,
    yes_no_values: dict[str, str],
) -> pd.DataFrame:
    """Build a single-row dataframe in exact training feature order."""
    row = {col: 0 for col in FEATURE_COLUMNS}

    row["area"] = float(area)
    row["bedrooms"] = int(bedrooms)
    row["stories"] = int(stories)
    row["bathrooms"] = int(bathrooms)
    row["parking"] = int(parking)

    for col in YES_NO_COLUMNS:
        row[col] = 1 if yes_no_values.get(col, "No") == "Yes" else 0

    row["furnishingstatus_semi-furnished"] = 1 if furnishingstatus == "Semi-furnished" else 0
    row["furnishingstatus_unfurnished"] = 1 if furnishingstatus == "Unfurnished" else 0

    return pd.DataFrame([row], columns=FEATURE_COLUMNS)
