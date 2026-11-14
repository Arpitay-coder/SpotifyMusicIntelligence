"""
ml/xgboost_model.py
====================
XGBoost gradient-boosted trees on the same High_Popularity target used by
Logistic Regression (see classification_target.py), for a fair comparison.

XGBoost builds an ensemble of decision trees sequentially, where each new
tree focuses on correcting the errors (residual gradient) of the trees
built so far. Parameters are kept modest (100 trees, depth 5) so training
stays fast on ~90k rows without an expensive hyperparameter search.
"""

import time
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                              roc_auc_score, confusion_matrix)
from xgboost import XGBClassifier

from ml.classification_target import build_target


def _style(fig, title=None, height=400):
    fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                       plot_bgcolor="rgba(0,0,0,0)",
                       font=dict(color="#E6EDF3", family="Inter, sans-serif"),
                       title=title, height=height, margin=dict(l=40, r=20, t=50, b=40))
    return fig


def run_xgboost(df: pd.DataFrame) -> dict:
    tgt = build_target(df)
    X, y = tgt["X"], tgt["y"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    model = XGBClassifier(
        n_estimators=100, max_depth=5, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8,
        random_state=42, eval_metric="logloss", n_jobs=-1,
    )
    start = time.time()
    model.fit(X_train, y_train)
    train_seconds = round(time.time() - start, 2)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

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

    imp_df = pd.DataFrame({"feature": tgt["features"], "importance": model.feature_importances_})
    imp_df = imp_df.sort_values("importance", ascending=False)
    fig_imp = px.bar(imp_df, x="importance", y="feature", orientation="h",
                      color="importance", color_continuous_scale="Greens")
    fig_imp.update_layout(yaxis=dict(autorange="reversed"), yaxis_title="", xaxis_title="Importance")
    _style(fig_imp, "Feature Importance", height=420)

    return {
        "metrics": metrics,
        "threshold": tgt["threshold"], "percentile": tgt["percentile"],
        "train_seconds": train_seconds,
        "params": {"n_estimators": 100, "max_depth": 5, "learning_rate": 0.1,
                   "subsample": 0.8, "colsample_bytree": 0.8},
        "figs": {"confusion_matrix": fig_cm.to_json(), "feature_importance": fig_imp.to_json()},
        "features": tgt["features"],
        "n_train": int(len(X_train)), "n_test": int(len(X_test)),
    }
