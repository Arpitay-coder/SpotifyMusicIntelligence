"""
ml/linear_regression.py
========================
Regression task: predict a track's `popularity` (0-100) from its numeric
audio features. Trains four linear models — Linear, Ridge, Lasso, Elastic
Net — on the SAME train/test split so their metrics are directly comparable.

What is Linear Regression?
    A model that fits a straight-line (hyperplane) relationship between the
    input features and a continuous target: y = w1*x1 + w2*x2 + ... + b.
Why used here?
    Popularity is a continuous 0-100 score, so this is a regression problem;
    linear regression gives an interpretable, fast baseline.
Ridge / Lasso / Elastic Net
    Same linear equation, but with an added penalty on the size of the
    coefficients (L2 / L1 / both) to reduce overfitting and, for Lasso,
    shrink unhelpful features' weights to exactly zero.
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

FEATURES = [
    "danceability", "energy", "loudness", "speechiness", "acousticness",
    "instrumentalness", "liveness", "valence", "tempo", "duration_ms",
]
TARGET = "popularity"


def _style(fig, title=None, height=400):
    fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                       plot_bgcolor="rgba(0,0,0,0)",
                       font=dict(color="#E6EDF3", family="Inter, sans-serif"),
                       title=title, height=height, margin=dict(l=40, r=20, t=50, b=40))
    return fig


def run_linear_regression(df: pd.DataFrame) -> dict:
    X = df[FEATURES].copy()
    y = df[TARGET].copy()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    models = {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(alpha=1.0, random_state=42),
        "Lasso Regression": Lasso(alpha=0.1, random_state=42, max_iter=5000),
        "Elastic Net": ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42, max_iter=5000),
    }

    results = {}
    predictions = {}
    for name, model in models.items():
        model.fit(X_train_s, y_train)
        y_pred = model.predict(X_test_s)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
        r2 = r2_score(y_test, y_pred)
        coefs = dict(zip(FEATURES, [round(float(c), 4) for c in model.coef_]))
        results[name] = {
            "mae": round(float(mae), 3), "rmse": round(rmse, 3), "r2": round(float(r2), 4),
            "coefficients": coefs,
        }
        predictions[name] = y_pred

    # Actual vs predicted for the best model by R2
    best_name = max(results, key=lambda k: results[k]["r2"])
    best_pred = predictions[best_name]

    sample_idx = np.random.RandomState(42).choice(len(y_test), size=min(1500, len(y_test)), replace=False)
    y_test_arr = y_test.values
    actual_sample = y_test_arr[sample_idx]
    pred_sample = best_pred[sample_idx]

    fig_actual_pred = go.Figure()
    fig_actual_pred.add_trace(go.Scatter(x=actual_sample, y=pred_sample, mode="markers",
                                          marker=dict(color="#1DB954", size=5, opacity=0.55),
                                          name="Predictions"))
    lims = [0, 100]
    fig_actual_pred.add_trace(go.Scatter(x=lims, y=lims, mode="lines",
                                          line=dict(color="#E6EDF3", dash="dash"), name="Perfect Prediction"))
    fig_actual_pred.update_layout(xaxis_title="Actual Popularity", yaxis_title="Predicted Popularity")
    _style(fig_actual_pred, f"Actual vs Predicted — {best_name} (test sample of {len(sample_idx)})")

    residuals = actual_sample - pred_sample
    fig_residual = go.Figure()
    fig_residual.add_trace(go.Scatter(x=pred_sample, y=residuals, mode="markers",
                                       marker=dict(color="#7F77DD", size=5, opacity=0.55)))
    fig_residual.add_hline(y=0, line_dash="dash", line_color="#E6EDF3")
    fig_residual.update_layout(xaxis_title="Predicted Popularity", yaxis_title="Residual (Actual − Predicted)")
    _style(fig_residual, f"Residual Plot — {best_name}")

    comp_df = pd.DataFrame(results).T.reset_index().rename(columns={"index": "model"})
    fig_comparison = go.Figure()
    for metric, color in [("mae", "#7F77DD"), ("rmse", "#D85A30")]:
        fig_comparison.add_trace(go.Bar(x=comp_df["model"], y=comp_df[metric], name=metric.upper(), marker_color=color))
    _style(fig_comparison, "Model Comparison — MAE & RMSE (lower is better)")
    fig_comparison.update_layout(barmode="group")

    fig_r2 = px.bar(comp_df, x="model", y="r2", color="r2", color_continuous_scale="Greens")
    _style(fig_r2, "Model Comparison — R² (higher is better)")

    return {
        "results": results,
        "best_model": best_name,
        "threshold_note": None,
        "figs": {
            "actual_vs_predicted": fig_actual_pred.to_json(),
            "residual_plot": fig_residual.to_json(),
            "comparison_error": fig_comparison.to_json(),
            "comparison_r2": fig_r2.to_json(),
        },
        "features": FEATURES,
        "target": TARGET,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
    }
