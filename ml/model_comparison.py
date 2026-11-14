"""
ml/model_comparison.py
=======================
Builds the comparison charts for the Model Comparison page from the
already-computed (cached) results of each individual model page. Does not
retrain anything itself.
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from ml.clustering_utils import style_fig


def classification_comparison(logreg: dict, xgb: dict, ada: dict) -> dict:
    rows = []
    for name, res in [("Logistic Regression", logreg), ("XGBoost", xgb), ("AdaBoost", ada)]:
        m = res["metrics"]
        rows.append({"model": name, **m})
    comp_df = pd.DataFrame(rows)

    fig = go.Figure()
    metrics = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    colors = ["#1DB954", "#1ED760", "#7F77DD", "#D85A30", "#66E39A"]
    for metric, color in zip(metrics, colors):
        fig.add_trace(go.Bar(x=comp_df["model"], y=comp_df[metric], name=metric.upper().replace("_", "-"),
                              marker_color=color))
    fig.update_layout(barmode="group", yaxis_title="Score")
    style_fig(fig, "Classification Model Comparison", height=440)

    return {"table": comp_df.to_dict(orient="records"), "fig": fig.to_json()}


def regression_comparison(linreg: dict) -> dict:
    rows = [{"model": k, **v} for k, v in linreg["results"].items()]
    for r in rows:
        r.pop("coefficients", None)
    comp_df = pd.DataFrame(rows)

    fig = go.Figure()
    fig.add_trace(go.Bar(x=comp_df["model"], y=comp_df["rmse"], name="RMSE", marker_color="#D85A30"))
    fig.add_trace(go.Bar(x=comp_df["model"], y=comp_df["mae"], name="MAE", marker_color="#7F77DD"))
    fig.update_layout(barmode="group", yaxis_title="Error (lower is better)")
    style_fig(fig, "Regression Model Comparison — Error", height=400)

    fig_r2 = px.bar(comp_df, x="model", y="r2", color="r2", color_continuous_scale="Greens")
    style_fig(fig_r2, "Regression Model Comparison — R²", height=380)

    return {"table": comp_df.to_dict(orient="records"), "fig": fig.to_json(), "fig_r2": fig_r2.to_json()}


def clustering_comparison(kmeans: dict, kmeans_pp: dict, hierarchical: dict, dbscan: dict) -> dict:
    rows = [
        {"algorithm": "K-Means", "silhouette": kmeans["silhouette"], "clusters": kmeans["k"], "noise_points": 0},
        {"algorithm": "K-Means++", "silhouette": kmeans_pp["silhouette"], "clusters": kmeans_pp["k"], "noise_points": 0},
        {"algorithm": "Hierarchical", "silhouette": hierarchical["silhouette"], "clusters": hierarchical["k"], "noise_points": 0},
        {"algorithm": "DBSCAN", "silhouette": dbscan["silhouette"], "clusters": dbscan["n_clusters"], "noise_points": dbscan["n_noise"]},
    ]
    comp_df = pd.DataFrame(rows)

    fig = px.bar(comp_df, x="algorithm", y="silhouette", color="algorithm",
                 color_discrete_sequence=px.colors.qualitative.Set2)
    fig.update_layout(showlegend=False, yaxis_title="Silhouette Score")
    style_fig(fig, "Clustering Comparison — Silhouette Score", height=380)

    return {"table": comp_df.to_dict(orient="records"), "fig": fig.to_json()}
