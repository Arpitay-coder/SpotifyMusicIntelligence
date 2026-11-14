# Spotify Music Intelligence — ML/Data Science Dashboard

An interactive Flask dashboard that explores a real ~90,000-track Spotify
dataset through exploratory data analysis, regression, classification,
boosting, and clustering — with a hand-built, Spotify-themed animation of
DBSCAN discovering "music communities" track by track.

---

## 1. Project Overview

This is Project 12: a Spotify/music data-science and machine-learning
dashboard. Each concept in the sidebar (Preprocessing, EDA, Linear
Regression, Logistic Regression, XGBoost, AdaBoost, K-Means, K-Means++,
Hierarchical Clustering, DBSCAN, Cluster Animation, Model Comparison, Music
Insights) opens its own dedicated page — nothing is dumped onto one giant
scrolling page, and every chart, metric and table is computed from the real
dataset, not placeholder data.

## 2. Features

- Dark, glassmorphism "music dashboard" UI with an animated hero equalizer,
  floating music-note particles, and a pulsing epsilon ring in the DBSCAN
  animation.
- Real before/after preprocessing report (rows, columns, missing values,
  duplicates) computed from the actual CSV.
- 15-chart interactive EDA page (Plotly): distributions, top artists/tracks,
  genre comparisons, a correlation heatmap, explicit-vs-not comparison, and
  audio feature boxplots.
- Linear / Ridge / Lasso / Elastic Net regression predicting `popularity`,
  with actual-vs-predicted and residual plots.
- Logistic Regression, XGBoost, and AdaBoost all classifying the same,
  transparently-defined `High_Popularity` target (popularity ≥ the 75th
  percentile of the cleaned dataset — the exact number is shown on-page),
  each with confusion matrix, ROC curve (logistic) or feature importance
  (XGBoost/AdaBoost).
- K-Means, K-Means++, Hierarchical Clustering, and DBSCAN, all with live
  controls (K slider, eps/min_samples sliders, a feature checklist) that
  recompute and redraw in well under a second.
- A from-scratch, server-computed DBSCAN trace animated in the browser:
  play/pause/step/reset controls, a pulsing epsilon ring around the "active"
  track, and a live info panel showing the real track name/artist DBSCAN is
  currently visiting.
- A Model Comparison page stacking every classifier, every regressor, and
  every clustering algorithm's metrics side by side.
- A Music Insights page whose bullet points are generated live from the
  actual dataset and cached model results — nothing hard-coded.

## 3. Dataset

`data/spotify.csv` — ~114,000 raw rows (~90,000 after cleaning) with
columns: `track_id, artists, album_name, track_name, popularity,
duration_ms, explicit, danceability, energy, key, loudness, mode,
speechiness, acousticness, instrumentalness, liveness, valence, tempo,
time_signature, track_genre`.

This dataset repeats a track once per genre it is tagged with, which is why
de-duplicating on `track_id` (see Preprocessing) roughly matches the ~24k
duplicate count you'll see reported on the Dataset page.

## 4. ML Algorithms (Supervised)

| Page | Algorithm | Target | Notes |
|---|---|---|---|
| Linear Regression | Linear, Ridge, Lasso, Elastic Net | `popularity` (continuous) | StandardScaler on features; same train/test split across all 4 |
| Logistic Regression | Logistic Regression | `High_Popularity` (0/1) | Threshold = 75th percentile of popularity, shown on-page |
| XGBoost | Gradient-boosted trees | `High_Popularity` (0/1) | 100 trees, depth 5 — kept modest for practicality on ~90k rows |
| AdaBoost | Adaptive Boosting (decision-stump weak learners) | `High_Popularity` (0/1) | 100 estimators, depth-2 stumps |

Honest note: audio features alone are a **weak** predictor of popularity
(this is realistic — popularity is driven far more by marketing, artist
fame, and playlist placement than by danceability/energy/etc.). The
dashboard reports the real, modest R²/recall numbers rather than dressing
them up.

## 5. Clustering Algorithms (Unsupervised)

| Page | Algorithm | Notes |
|---|---|---|
| K-Means | `KMeans(init="random", n_init=10)` | Elbow curve, WCSS, silhouette |
| K-Means++ | `KMeans(init="k-means++", n_init=10)` | Compared directly against random-init WCSS |
| Hierarchical | `AgglomerativeClustering` + real `scipy.cluster.hierarchy` dendrogram | Runs on a further sample (300 tracks) — a full 90k×90k distance matrix isn't practical |
| DBSCAN | `DBSCAN(eps, min_samples)` | Label -1 = noise; eps/min_samples adjustable via sliders |

All four scale numeric features with `StandardScaler` first (mirroring the
supplied `clustering_functions.py` reference), and all run on a random
6,000-track sample of the cleaned dataset (except Hierarchical, sampled
further to 300) so the interactive controls stay responsive — this is
disclosed directly on each page.

## 6. Architecture

```
spotify_ml_lab/
├── app.py                    # Flask routes + in-memory result cache
├── main.py                   # `python main.py` entry point
├── requirements.txt
├── data/
│   └── spotify.csv
├── ml/
│   ├── data_loader.py        # raw CSV load + headline stats
│   ├── preprocessing.py      # cleaning pipeline + before/after report
│   ├── eda.py                 # 15 Plotly figures
│   ├── classification_target.py  # shared High_Popularity definition
│   ├── linear_regression.py
│   ├── logistic_regression.py
│   ├── xgboost_model.py
│   ├── adaboost_model.py
│   ├── clustering_utils.py   # shared scaling/sampling helpers
│   ├── kmeans.py
│   ├── kmeans_plusplus.py
│   ├── hierarchical.py
│   ├── dbscan.py
│   ├── dbscan_animation.py   # step-by-step DBSCAN trace generator
│   ├── model_comparison.py
│   └── insights.py
├── templates/                # one Jinja template per route
└── static/
    ├── css/style.css
    └── js/{main.js, dbscan_animation.js}
```

## 7. Installation

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

## 8. How to Run

```bash
python app.py
# or
python main.py
```

Then open **http://127.0.0.1:5000** in your browser.

## 9. Requirements

See `requirements.txt` — Flask, pandas, numpy, scikit-learn, scipy,
xgboost, plotly.

## 10. Routes

| Route | Page |
|---|---|
| `/` | Dashboard |
| `/dataset` | Dataset overview |
| `/preprocessing` | Preprocessing before/after report |
| `/eda` | Exploratory Data Analysis |
| `/linear-regression` | Linear/Ridge/Lasso/ElasticNet regression |
| `/logistic-regression` | Logistic Regression classifier |
| `/xgboost` | XGBoost classifier |
| `/adaboost` | AdaBoost classifier |
| `/clustering` | Clustering overview |
| `/clustering/kmeans` | K-Means (query: `k`, `features`) |
| `/clustering/kmeans-plus-plus` | K-Means++ (query: `k`, `features`) |
| `/clustering/hierarchical` | Hierarchical (query: `k`, `linkage`, `features`) |
| `/clustering/dbscan` | DBSCAN (query: `eps`, `min_samples`, `features`) |
| `/clustering/animation` | DBSCAN "music communities" animation |
| `/model-comparison` | Cross-model metric comparison |
| `/music-insights` | Auto-generated data-driven insights |

## 11. Algorithm Explanations

See the "concept box" at the top of each model/clustering page — every page
explains what the algorithm is, why it's used, and (for regression/logistic
regression) the underlying equation, right above the real results.

## 12. Dataset Preprocessing

1. Drop the unnamed pandas index column.
2. Drop rows missing `artists`/`album_name`/`track_name`.
3. De-duplicate on `track_id` (keep first) — removes the genre-repeat rows.
4. Encode `explicit` as 0/1.
5. Remove physically invalid rows (`duration_ms <= 0`, `tempo <= 0`).
6. Report IQR-based outliers per numeric feature — informational only,
   nothing is dropped, since these are real musical variation.
7. Categorical text columns are never scaled; numeric scaling happens only
   inside each model, right before training.

Full before/after numbers are on the `/preprocessing` page.

## 13. Performance Considerations

- The dataset (raw CSV → cleaned dataframe) is loaded and cleaned **once**
  per process and cached in memory (`app.py`'s `CACHE` dict).
- Linear/Logistic Regression, XGBoost, and AdaBoost are trained **once** and
  cached — they are not retrained on every page view.
- Clustering pages (K-Means, K-Means++, Hierarchical, DBSCAN) intentionally
  retrain per request, because their whole point is live, interactive
  parameter controls (K slider, eps/min_samples sliders) — but they run on a
  sampled subset of the data (disclosed on each page) specifically so that
  retraining per request stays fast (well under a second in testing).
- The EDA page's histograms are pre-binned with `numpy.histogram` server-side
  rather than shipping all ~90k raw values to the browser per chart, keeping
  page weight reasonable.
