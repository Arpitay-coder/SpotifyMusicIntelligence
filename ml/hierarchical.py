"""
ml/hierarchical.py
===================
Agglomerative (bottom-up) hierarchical clustering with a real
scipy.cluster.hierarchy dendrogram — not a fabricated tree image.

90k (or even the 6,000-track clustering sample) rows is still too many to
render a legible dendrogram or fit within reasonable memory for
AgglomerativeClustering's distance matrix, so a further, smaller sample
(default 300 tracks) is drawn specifically for the dendrogram and cluster
assignment on this page. This is disclosed directly in the returned dict
and shown on the page.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage as scipy_linkage

from ml.clustering_utils import sample_for_clustering, prepare_features, style_fig

DENDROGRAM_SAMPLE_SIZE = 300


def run_hierarchical(df: pd.DataFrame, feature_cols: list, k: int = 4, linkage: str = "ward") -> dict:
    # Smaller sample specifically for this page (distance-matrix cost)
    sampled = sample_for_clustering(df, n=DENDROGRAM_SAMPLE_SIZE, random_state=42)
    clean_df, X_scaled, scaler = prepare_features(sampled, feature_cols)

    model = AgglomerativeClustering(n_clusters=k, linkage=linkage)
    labels = model.fit_predict(X_scaled)

    clean_df = clean_df.copy()
    clean_df["cluster"] = labels

    sil = float(silhouette_score(X_scaled, labels)) if k > 1 else None
    counts = pd.Series(labels).value_counts().sort_index()

    # Real dendrogram via scipy
    Z = scipy_linkage(X_scaled, method=linkage)
    dendro = dendrogram(Z, no_plot=True, truncate_mode="lastp", p=30)

    fig_dendro = go.Figure()
    icoord, dcoord = dendro["icoord"], dendro["dcoord"]
    for xs, ys in zip(icoord, dcoord):
        fig_dendro.add_trace(go.Scatter(x=xs, y=ys, mode="lines",
                                         line=dict(color="#1DB954", width=1.5), hoverinfo="skip",
                                         showlegend=False))
    fig_dendro.update_layout(xaxis_title="Track clusters (truncated)", yaxis_title="Distance",
                              xaxis=dict(showticklabels=False))
    style_fig(fig_dendro, f"Dendrogram — {linkage} linkage (sample of {len(clean_df)} tracks)", height=420)

    fx, fy = feature_cols[0], feature_cols[1] if len(feature_cols) > 1 else feature_cols[0]
    plot_df = clean_df.copy()
    plot_df["cluster"] = labels.astype(str)
    import plotly.express as px
    fig_scatter = px.scatter(plot_df, x=fx, y=fy, color="cluster",
                              color_discrete_sequence=px.colors.qualitative.Set2, opacity=0.7,
                              hover_data=["track_name", "artists"] if "track_name" in plot_df.columns else None)
    fig_scatter.update_layout(xaxis_title=fx, yaxis_title=fy)
    style_fig(fig_scatter, f"Hierarchical Clusters (K={k}, {linkage} linkage)")

    return {
        "silhouette": round(sil, 4) if sil is not None else None,
        "k": k, "linkage": linkage,
        "cluster_counts": {int(c): int(n) for c, n in counts.items()},
        "n_points": int(len(clean_df)), "sample_size": DENDROGRAM_SAMPLE_SIZE,
        "figs": {"dendrogram": fig_dendro.to_json(), "scatter": fig_scatter.to_json()},
        "feature_cols": feature_cols,
        "sample_note": f"Dendrogram and hierarchical cluster assignment use a representative sample of {len(clean_df)} tracks for performance (full pairwise distance matrices are impractical at ~90k rows).",
    }
