"""
ml/clustering_utils.py
=======================
Shared helpers for all four clustering modules (kmeans, kmeans_plusplus,
hierarchical, dbscan). Mirrors the pattern in the supplied
clustering_functions.py reference: scale numeric features with
StandardScaler before any distance-based clustering, since raw-scale
features like tempo/loudness would otherwise dominate the distance metric.

Clustering here is computed on a random sample of the cleaned dataset
(default 6,000 tracks) rather than the full ~90k rows. This is disclosed
directly on every clustering page. It is a deliberate performance choice so
that changing K / eps / linkage in the UI recomputes and re-renders in
under a second or two instead of tens of seconds — full-dataset distance
matrices (hierarchical) or repeated K-Means re-inits (interactive K slider)
are not practical to recompute on every request at 90k rows.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

ALL_CLUSTER_FEATURES = [
    "danceability", "energy", "valence", "acousticness",
    "instrumentalness", "speechiness", "liveness", "tempo", "loudness",
]
DEFAULT_FEATURES = ["danceability", "energy", "valence"]
DEFAULT_SAMPLE_SIZE = 6000


def sample_for_clustering(df: pd.DataFrame, n=DEFAULT_SAMPLE_SIZE, random_state=42) -> pd.DataFrame:
    if len(df) <= n:
        return df.reset_index(drop=True)
    return df.sample(n=n, random_state=random_state).reset_index(drop=True)


def prepare_features(df: pd.DataFrame, feature_cols: list):
    """Drop rows with missing feature values, scale features.
    Returns (clean_df, X_scaled, fitted_scaler)."""
    clean_df = df.dropna(subset=feature_cols).reset_index(drop=True)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(clean_df[feature_cols])
    return clean_df, X_scaled, scaler


def style_fig(fig, title=None, height=460):
    fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                       plot_bgcolor="rgba(0,0,0,0)",
                       font=dict(color="#E6EDF3", family="Inter, sans-serif"),
                       title=title, height=height, margin=dict(l=40, r=20, t=50, b=40))
    return fig
