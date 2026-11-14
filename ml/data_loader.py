"""
ml/data_loader.py
==================
Loads the raw Spotify dataset from disk exactly once per process and exposes
small helpers for computing the headline dashboard statistics.

No cleaning happens here — that is preprocessing.py's job. This module only
reads the CSV and reports what is actually in it (rows, columns, missing
values, duplicates) so the "before preprocessing" numbers on the dashboard
reflect the real, untouched file.
"""

import os
import pandas as pd

DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "spotify.csv")


def load_raw_data(path: str = DATA_PATH) -> pd.DataFrame:
    """Read the raw Spotify CSV as-is (no cleaning)."""
    df = pd.read_csv(path)
    return df


def raw_data_summary(df: pd.DataFrame) -> dict:
    """Basic 'before preprocessing' facts about the raw file."""
    return {
        "rows": int(len(df)),
        "columns": int(df.shape[1]),
        "missing_values": int(df.isna().sum().sum()),
        "missing_by_column": {k: int(v) for k, v in df.isna().sum().items() if v > 0},
        "duplicate_track_ids": int(df["track_id"].duplicated().sum()) if "track_id" in df.columns else 0,
        "duplicate_rows": int(df.duplicated().sum()),
        "column_names": list(df.columns),
    }


def headline_stats(df: pd.DataFrame) -> dict:
    """Numbers shown as the big cards on the main dashboard.
    Expects a *cleaned* dataframe (post track_id de-duplication) so that
    counts of artists/albums/genres/tracks are meaningful."""
    return {
        "n_tracks": int(len(df)),
        "n_artists": int(df["artists"].nunique()),
        "n_albums": int(df["album_name"].nunique()),
        "n_genres": int(df["track_genre"].nunique()),
        "avg_popularity": round(float(df["popularity"].mean()), 2),
        "avg_danceability": round(float(df["danceability"].mean()), 3),
        "avg_energy": round(float(df["energy"].mean()), 3),
        "avg_valence": round(float(df["valence"].mean()), 3),
        "avg_tempo": round(float(df["tempo"].mean()), 1),
    }
