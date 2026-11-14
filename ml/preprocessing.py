"""
ml/preprocessing.py
====================
Turns the raw Spotify CSV into a clean, model-ready dataframe and records
exactly what changed, so the Preprocessing page can show real
before/after numbers instead of a vague description.

Steps performed (in order):
    1. Drop the unnamed pandas index column ("Unnamed: 0") if present.
    2. Drop rows missing artists / album_name / track_name (a track with no
       name/artist is not usable for display or analysis).
    3. De-duplicate on track_id (this dataset lists the same track once per
       genre it's tagged with, which would badly bias any model that treats
       rows as independent observations).
    4. Cast `explicit` to int (0/1) — it arrives as a Python bool.
    5. Clip physically-impossible values (e.g. negative duration) — inspect
       only, does not silently drop legitimate outliers such as very long
       tracks or very high-tempo tracks, since those are real music.
    6. Leave categorical text columns (artists, album_name, track_name,
       track_genre) untouched as text — they are never blindly scaled.
       Numeric feature scaling happens inside each model module right before
       training, using only the numeric feature columns needed for that
       model (see linear_regression.py, logistic_regression.py, the
       clustering modules, etc.) so training/test splits stay independent.

Returns the cleaned dataframe plus a report dict used directly by the
Preprocessing page template.
"""

import numpy as np
import pandas as pd

NUMERIC_AUDIO_FEATURES = [
    "danceability", "energy", "loudness", "speechiness", "acousticness",
    "instrumentalness", "liveness", "valence", "tempo", "duration_ms",
]

CATEGORICAL_COLUMNS = ["artists", "album_name", "track_name", "track_genre"]


def _outlier_report(df: pd.DataFrame, cols) -> dict:
    """IQR-based outlier counts, for display only (nothing is dropped)."""
    report = {}
    for col in cols:
        q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        n_out = int(((df[col] < lower) | (df[col] > upper)).sum())
        report[col] = {"count": n_out, "pct": round(100 * n_out / len(df), 2),
                        "lower_bound": round(float(lower), 3), "upper_bound": round(float(upper), 3)}
    return report


def preprocess(df_raw: pd.DataFrame) -> dict:
    before = {
        "rows": int(len(df_raw)),
        "columns": int(df_raw.shape[1]),
        "missing_values": int(df_raw.isna().sum().sum()),
        "duplicate_track_ids": int(df_raw["track_id"].duplicated().sum()),
    }

    df = df_raw.copy()

    # 1. Drop unnamed index column
    unnamed_cols = [c for c in df.columns if c.startswith("Unnamed")]
    df = df.drop(columns=unnamed_cols, errors="ignore")

    # 2. Drop rows missing essential text identifiers
    essential = ["artists", "album_name", "track_name"]
    rows_before_dropna = len(df)
    df = df.dropna(subset=essential)
    dropped_missing = rows_before_dropna - len(df)

    # 3. De-duplicate on track_id, keep first occurrence
    rows_before_dedup = len(df)
    df = df.drop_duplicates(subset="track_id", keep="first")
    dropped_duplicates = rows_before_dedup - len(df)

    # 4. Encode boolean -> int
    if df["explicit"].dtype == bool:
        df["explicit"] = df["explicit"].astype(int)
    else:
        df["explicit"] = df["explicit"].map({True: 1, False: 0, "True": 1, "False": 0}).fillna(0).astype(int)

    # 5. Sanity clip: duration must be positive; tempo must be > 0
    invalid_duration = int((df["duration_ms"] <= 0).sum())
    invalid_tempo = int((df["tempo"] <= 0).sum())
    df = df[(df["duration_ms"] > 0) & (df["tempo"] > 0)].reset_index(drop=True)

    df = df.reset_index(drop=True)

    outliers = _outlier_report(df, NUMERIC_AUDIO_FEATURES + ["popularity"])

    after = {
        "rows": int(len(df)),
        "columns": int(df.shape[1]),
        "missing_values": int(df.isna().sum().sum()),
        "duplicate_track_ids": int(df["track_id"].duplicated().sum()),
    }

    report = {
        "before": before,
        "after": after,
        "steps": [
            {"step": "Dropped unnamed index column", "detail": f"{len(unnamed_cols)} column(s): {unnamed_cols}"},
            {"step": "Dropped rows with missing artist/album/track name",
             "detail": f"{dropped_missing} row(s) removed"},
            {"step": "De-duplicated on track_id (kept first occurrence)",
             "detail": f"{dropped_duplicates} duplicate row(s) removed — this dataset repeats a track once per genre tag"},
            {"step": "Encoded `explicit` as 0/1 integer", "detail": "boolean → int"},
            {"step": "Removed physically invalid rows",
             "detail": f"{invalid_duration} row(s) with duration_ms<=0, {invalid_tempo} row(s) with tempo<=0"},
            {"step": "Categorical columns left as text",
             "detail": f"{CATEGORICAL_COLUMNS} — never blindly scaled; only numeric audio features are scaled, and only inside each model right before training"},
        ],
        "outliers": outliers,
        "numeric_features": NUMERIC_AUDIO_FEATURES,
        "categorical_features": CATEGORICAL_COLUMNS,
    }

    return {"data": df, "report": report}
