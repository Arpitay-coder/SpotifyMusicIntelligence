"""
ml/classification_target.py
============================
Defines the ONE binary classification target shared by Logistic Regression,
XGBoost, and AdaBoost so their metrics are directly comparable on the
Model Comparison page.

High_Popularity = 1 if popularity >= the dataset's 75th percentile, else 0.

The 75th percentile is computed from the actual cleaned dataset (not a
hard-coded number) and is exposed here so every page that trains a
classifier can display the exact threshold used.
"""

import pandas as pd

FEATURES = [
    "danceability", "energy", "loudness", "speechiness", "acousticness",
    "instrumentalness", "liveness", "valence", "tempo", "duration_ms",
]
PERCENTILE = 0.75


def build_target(df: pd.DataFrame) -> dict:
    """Returns X (features), y (0/1 target), and the threshold used."""
    threshold = float(df["popularity"].quantile(PERCENTILE))
    y = (df["popularity"] >= threshold).astype(int)
    X = df[FEATURES].copy()
    return {
        "X": X, "y": y, "threshold": round(threshold, 2),
        "percentile": int(PERCENTILE * 100),
        "positive_rate": round(float(y.mean()), 4),
        "features": FEATURES,
    }
