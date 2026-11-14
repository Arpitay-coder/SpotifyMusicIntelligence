"""
ml/kmeans.py
============
Plain K-Means with random centroid initialization, following the same
core config as the supplied clustering_functions.py reference:
    KMeans(n_clusters=k, init="random", n_init=10, random_state=42)
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from ml.clustering_utils import sample_for_clustering, prepare_features, style_fig, DEFAULT_SAMPLE_SIZE


def run_kmeans(df: pd.DataFrame, feature_cols: list, k: int = 4, random_state: int = 42) -> dict:
    sampled = sample_for_clustering(df)
    clean_df, X_scaled, scaler = prepare_features(sampled, feature_cols)

    model = KMeans(n_clusters=k, init="random", n_init=10, random_state=random_state)
    labels = model.fit_predict(X_scaled)

    clean_df = clean_df.copy()
    clean_df["cluster"] = labels

    wcss = float(model.inertia_)
    sil = float(silhouette_score(X_scaled, labels)) if k > 1 else None

    # Elbow curve: WCSS for k = 2..9 on the same sample/features
    elbow_k = list(range(2, 10))
    elbow_wcss = []
    for kk in elbow_k:
        m = KMeans(n_clusters=kk, init="random", n_init=10, random_state=random_state)
        m.fit(X_scaled)
        elbow_wcss.append(float(m.inertia_))
    fig_elbow = go.Figure()
    fig_elbow.add_trace(go.Scatter(x=elbow_k, y=elbow_wcss, mode="lines+markers",
                                    line=dict(color="#1DB954", width=3), marker=dict(size=8)))
    fig_elbow.add_vline(x=k, line_dash="dash", line_color="#E6EDF3")
    fig_elbow.update_layout(xaxis_title="K (number of clusters)", yaxis_title="WCSS (Inertia)")
    style_fig(fig_elbow, "Elbow Curve", height=380)

    fig_scatter = _scatter_fig(clean_df, feature_cols, labels, model.cluster_centers_, scaler,
                                title=f"K-Means Clusters (K={k}, random init)")

    counts = pd.Series(labels).value_counts().sort_index()

    return {
        "wcss": round(wcss, 2), "silhouette": round(sil, 4) if sil is not None else None,
        "k": k, "cluster_counts": {int(c): int(n) for c, n in counts.items()},
        "n_points": int(len(clean_df)), "sample_size": DEFAULT_SAMPLE_SIZE,
        "figs": {"scatter": fig_scatter.to_json(), "elbow": fig_elbow.to_json()},
        "feature_cols": feature_cols,
    }


def _scatter_fig(clean_df, feature_cols, labels, centers_scaled, scaler, title):
    """2D scatter using the first two selected features (raw scale for
    readability), with centroids projected back to raw scale."""
    fx, fy = feature_cols[0], feature_cols[1] if len(feature_cols) > 1 else feature_cols[0]
    plot_df = clean_df.copy()
    plot_df["cluster"] = labels.astype(str)

    fig = px.scatter(plot_df, x=fx, y=fy, color="cluster",
                      color_discrete_sequence=px.colors.qualitative.Set2,
                      opacity=0.65, hover_data=["track_name", "artists"] if "track_name" in plot_df.columns else None)

    # project centroids back to raw units for the two plotted dims
    centers_raw = scaler.inverse_transform(centers_scaled)
    fx_idx, fy_idx = feature_cols.index(fx), feature_cols.index(fy)
    fig.add_trace(go.Scatter(
        x=centers_raw[:, fx_idx], y=centers_raw[:, fy_idx], mode="markers",
        marker=dict(color="white", size=16, symbol="x", line=dict(color="#1DB954", width=2)),
        name="Centroids",
    ))
    fig.update_layout(xaxis_title=fx, yaxis_title=fy)
    style_fig(fig, title)
    return fig
