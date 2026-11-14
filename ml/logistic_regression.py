"""
ml/logistic_regression.py
==========================
Binary classification: predict High_Popularity (see classification_target.py
for the exact, transparent threshold) from numeric audio features.

Logistic Regression passes a linear combination of features through the
sigmoid function  s(z) = 1 / (1 + e^-z)  to squash the output into a
probability between 0 and 1, then thresholds it at 0.5 to get a class.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                              roc_auc_score, roc_curve, confusion_matrix)

from ml.classification_target import build_target


def _style(fig, title=None, height=400):
    fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                       plot_bgcolor="rgba(0,0,0,0)",
                       font=dict(color="#E6EDF3", family="Inter, sans-serif"),
                       title=title, height=height, margin=dict(l=40, r=20, t=50, b=40))
    return fig


def run_logistic_regression(df: pd.DataFrame) -> dict:
    tgt = build_target(df)
    X, y = tgt["X"], tgt["y"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train_s, y_train)

    y_pred = model.predict(X_test_s)
    y_proba = model.predict_proba(X_test_s)[:, 1]

    metrics = {
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "precision": round(float(precision_score(y_test, y_pred)), 4),
        "recall": round(float(recall_score(y_test, y_pred)), 4),
        "f1": round(float(f1_score(y_test, y_pred)), 4),
        "roc_auc": round(float(roc_auc_score(y_test, y_proba)), 4),
    }

    cm = confusion_matrix(y_test, y_pred)
    fig_cm = go.Figure(data=go.Heatmap(
        z=cm, x=["Predicted 0 (Low)", "Predicted 1 (High)"], y=["Actual 0 (Low)", "Actual 1 (High)"],
        colorscale="Greens", text=cm, texttemplate="%{text}", textfont=dict(size=16),
    ))
    _style(fig_cm, "Confusion Matrix", height=380)

    fpr, tpr, _ = roc_curve(y_test, y_proba)
    fig_roc = go.Figure()
    fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"ROC (AUC={metrics['roc_auc']:.3f})",
                                  line=dict(color="#1DB954", width=3)))
    fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random Classifier",
                                  line=dict(color="#666", dash="dash")))
    fig_roc.update_layout(xaxis_title="False Positive Rate", yaxis_title="True Positive Rate")
    _style(fig_roc, "ROC Curve")

    fig_proba = go.Figure()
    fig_proba.add_trace(go.Histogram(x=y_proba[y_test == 0], name="Actual: Low Popularity",
                                      marker_color="#7F77DD", opacity=0.7, nbinsx=40))
    fig_proba.add_trace(go.Histogram(x=y_proba[y_test == 1], name="Actual: High Popularity",
                                      marker_color="#1DB954", opacity=0.7, nbinsx=40))
    fig_proba.update_layout(barmode="overlay", xaxis_title="Predicted Probability of High Popularity",
                             yaxis_title="Count")
    _style(fig_proba, "Predicted Probability Distribution")

    z = np.linspace(-10, 10, 200)
    sigmoid = 1 / (1 + np.exp(-z))
    fig_sigmoid = go.Figure()
    fig_sigmoid.add_trace(go.Scatter(x=z, y=sigmoid, mode="lines", line=dict(color="#1DB954", width=3)))
    fig_sigmoid.add_hline(y=0.5, line_dash="dash", line_color="#666")
    fig_sigmoid.update_layout(xaxis_title="z (linear combination of features)", yaxis_title="σ(z) — probability")
    _style(fig_sigmoid, "The Sigmoid Function", height=350)

    coefs = dict(zip(tgt["features"], [round(float(c), 4) for c in model.coef_[0]]))

    return {
        "metrics": metrics,
        "threshold": tgt["threshold"],
        "percentile": tgt["percentile"],
        "positive_rate": tgt["positive_rate"],
        "coefficients": coefs,
        "figs": {
            "confusion_matrix": fig_cm.to_json(),
            "roc_curve": fig_roc.to_json(),
            "probability_distribution": fig_proba.to_json(),
            "sigmoid": fig_sigmoid.to_json(),
        },
        "features": tgt["features"],
        "n_train": int(len(X_train)), "n_test": int(len(X_test)),
    }
