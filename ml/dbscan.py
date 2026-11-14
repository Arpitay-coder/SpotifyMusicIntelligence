"""
ml/dbscan.py
============
Density-based clustering, following the supplied clustering_functions.py
reference:
    DBSCAN(eps=eps, min_samples=min_samples)

Label -1 means noise (a track that doesn't belong to any dense region).
StandardScaler is applied first (same as the reference), since eps is a
distance threshold and raw-scale features would distort it.
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score

from ml.clustering_utils import sample_for_clustering, prepare_features, style_fig, DEFAULT_SAMPLE_SIZE


def run_dbscan(df: pd.DataFrame, feature_cols: list, eps: float = 0.5, min_samples: int = 5) -> dict:
    sampled = sample_for_clustering(df)
    clean_df, X_scaled, scaler = prepare_features(sampled, feature_cols)

    model = DBSCAN(eps=eps, min_samples=min_samples)
    labels = model.fit_predict(X_scaled)

    clean_df = clean_df.copy()
    clean_df["cluster"] = labels

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = int(np.sum(labels == -1))

    sil = None
    if n_clusters > 1:
        mask = labels != -1
        if mask.sum() > 1:
            sil = float(silhouette_score(X_scaled[mask], labels[mask]))

    fx, fy = feature_cols[0], feature_cols[1] if len(feature_cols) > 1 else feature_cols[0]
    plot_df = clean_df.copy()
    plot_df["cluster_label"] = np.where(labels == -1, "Noise", "Cluster " + pd.Series(labels).astype(str))

    color_seq = px.colors.qualitative.Set2
    fig_scatter = px.scatter(plot_df, x=fx, y=fy, color="cluster_label",
                              color_discrete_map={"Noise": "#555555"},
                              color_discrete_sequence=color_seq, opacity=0.7,
                              hover_data=["track_name", "artists"] if "track_name" in plot_df.columns else None)
    fig_scatter.update_layout(xaxis_title=fx, yaxis_title=fy)
    style_fig(fig_scatter, f"DBSCAN Clusters (eps={eps}, min_samples={min_samples})")

    counts = pd.Series(labels).value_counts().sort_index()
    fig_dist = go.Figure()
    x_labels = ["Noise" if c == -1 else f"Cluster {c}" for c in counts.index]
    fig_dist.add_trace(go.Bar(x=x_labels, y=counts.values,
                               marker_color=["#555555" if c == -1 else "#1DB954" for c in counts.index]))
    style_fig(fig_dist, "Cluster / Noise Distribution", height=360)

    return {
        "n_clusters": n_clusters, "n_noise": n_noise,
        "silhouette": round(sil, 4) if sil is not None else None,
        "eps": eps, "min_samples": min_samples,
        "cluster_counts": {("noise" if c == -1 else int(c)): int(n) for c, n in counts.items()},
        "n_points": int(len(clean_df)), "sample_size": DEFAULT_SAMPLE_SIZE,
        "figs": {"scatter": fig_scatter.to_json(), "distribution": fig_dist.to_json()},
        "feature_cols": feature_cols,
    }
