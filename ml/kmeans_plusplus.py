"""
ml/kmeans_plusplus.py
======================
K-Means with K-Means++ initialization:
    KMeans(n_clusters=k, init="k-means++", n_init=10, random_state=42)

Unlike plain random init, K-Means++ picks the first centroid uniformly at
random, then picks each subsequent centroid with probability proportional
to its squared distance from the nearest centroid already chosen. This
spreads the starting centroids out across the data, which typically gives
faster, more consistent convergence and avoids poor random starts (e.g. two
initial centroids landing in the same dense region).
"""

import pandas as pd
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from ml.clustering_utils import sample_for_clustering, prepare_features, style_fig, DEFAULT_SAMPLE_SIZE
from ml.kmeans import _scatter_fig


def run_kmeans_plusplus(df: pd.DataFrame, feature_cols: list, k: int = 4, random_state: int = 42) -> dict:
    sampled = sample_for_clustering(df)
    clean_df, X_scaled, scaler = prepare_features(sampled, feature_cols)

    model = KMeans(n_clusters=k, init="k-means++", n_init=10, random_state=random_state)
    labels = model.fit_predict(X_scaled)

    clean_df = clean_df.copy()
    clean_df["cluster"] = labels

    wcss = float(model.inertia_)
    sil = float(silhouette_score(X_scaled, labels)) if k > 1 else None

    # Also fit plain-random K-Means on the same data/k for a direct comparison
    from sklearn.cluster import KMeans as _KM
    rand_model = _KM(n_clusters=k, init="random", n_init=10, random_state=random_state)
    rand_labels = rand_model.fit_predict(X_scaled)
    rand_wcss = float(rand_model.inertia_)
    rand_sil = float(silhouette_score(X_scaled, rand_labels)) if k > 1 else None

    fig_scatter = _scatter_fig(clean_df, feature_cols, labels, model.cluster_centers_, scaler,
                                title=f"K-Means++ Clusters (K={k})")

    fig_compare = go.Figure()
    fig_compare.add_trace(go.Bar(x=["WCSS"], y=[rand_wcss], name="Random Init", marker_color="#7F77DD"))
    fig_compare.add_trace(go.Bar(x=["WCSS"], y=[wcss], name="K-Means++ Init", marker_color="#1DB954"))
    fig_compare.update_layout(barmode="group", yaxis_title="WCSS (lower is better)")
    style_fig(fig_compare, "K-Means++ vs Random Init — WCSS", height=360)

    counts = pd.Series(labels).value_counts().sort_index()

    return {
        "wcss": round(wcss, 2), "silhouette": round(sil, 4) if sil is not None else None,
        "random_init_wcss": round(rand_wcss, 2),
        "random_init_silhouette": round(rand_sil, 4) if rand_sil is not None else None,
        "k": k, "cluster_counts": {int(c): int(n) for c, n in counts.items()},
        "n_points": int(len(clean_df)), "sample_size": DEFAULT_SAMPLE_SIZE,
        "figs": {"scatter": fig_scatter.to_json(), "comparison": fig_compare.to_json()},
        "feature_cols": feature_cols,
    }
