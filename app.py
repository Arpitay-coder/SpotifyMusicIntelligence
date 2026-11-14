"""
app.py
======
Flask application factory + routes for the Spotify Music Intelligence
dashboard. Heavy work (loading the CSV, cleaning it, training models) is
done ONCE per process and cached in memory (see `CACHE`), not on every
request. Clustering pages, which take interactive parameters (K, eps,
min_samples, feature selection) from the query string, are cheap enough
(a few hundred ms on the clustering sample — see ml/clustering_utils.py)
to recompute per request, which is what lets the K/eps controls feel live.
"""

import json
from flask import Flask, render_template, request, jsonify

from ml.data_loader import load_raw_data, raw_data_summary, headline_stats
from ml.preprocessing import preprocess, NUMERIC_AUDIO_FEATURES
from ml.eda import run_eda
from ml.linear_regression import run_linear_regression
from ml.logistic_regression import run_logistic_regression
from ml.xgboost_model import run_xgboost
from ml.adaboost_model import run_adaboost
from ml.kmeans import run_kmeans
from ml.kmeans_plusplus import run_kmeans_plusplus
from ml.hierarchical import run_hierarchical
from ml.dbscan import run_dbscan
from ml.dbscan_animation import build_dbscan_animation
from ml.model_comparison import classification_comparison, regression_comparison, clustering_comparison
from ml.insights import generate_insights
from ml.clustering_utils import ALL_CLUSTER_FEATURES, DEFAULT_FEATURES

app = Flask(__name__)

# ---------------------------------------------------------------------------
# In-memory cache. Populated lazily via CACHE getters below so the app boots
# fast; the first request to a given page pays the training cost once, every
# request after that (across all users, since this is a single dev-server
# process) is served from memory.
# ---------------------------------------------------------------------------
CACHE = {}


def get_clean_df():
    if "clean_df" not in CACHE:
        raw = load_raw_data()
        CACHE["raw_summary"] = raw_data_summary(raw)
        result = preprocess(raw)
        CACHE["clean_df"] = result["data"]
        CACHE["preprocess_report"] = result["report"]
    return CACHE["clean_df"]


def get_preprocess_report():
    get_clean_df()
    return CACHE["preprocess_report"]


def get_raw_summary():
    get_clean_df()
    return CACHE["raw_summary"]


def get_eda():
    if "eda" not in CACHE:
        CACHE["eda"] = run_eda(get_clean_df())
    return CACHE["eda"]


def get_linear_regression():
    if "linreg" not in CACHE:
        CACHE["linreg"] = run_linear_regression(get_clean_df())
    return CACHE["linreg"]


def get_logistic_regression():
    if "logreg" not in CACHE:
        CACHE["logreg"] = run_logistic_regression(get_clean_df())
    return CACHE["logreg"]


def get_xgboost():
    if "xgb" not in CACHE:
        CACHE["xgb"] = run_xgboost(get_clean_df())
    return CACHE["xgb"]


def get_adaboost():
    if "ada" not in CACHE:
        CACHE["ada"] = run_adaboost(get_clean_df())
    return CACHE["ada"]


def get_dbscan_animation():
    if "dbscan_anim" not in CACHE:
        CACHE["dbscan_anim"] = build_dbscan_animation(get_clean_df())
    return CACHE["dbscan_anim"]


def parse_features(request):
    feats = request.args.getlist("features")
    feats = [f for f in feats if f in ALL_CLUSTER_FEATURES]
    return feats if feats else DEFAULT_FEATURES


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    df = get_clean_df()
    stats = headline_stats(df)
    return render_template("index.html", stats=stats, active="dashboard")


@app.route("/dataset")
def dataset():
    df = get_clean_df()
    stats = headline_stats(df)
    sample_rows = df.head(12).to_dict(orient="records")
    columns = list(df.columns)
    return render_template("dataset.html", stats=stats, sample_rows=sample_rows,
                            columns=columns, raw_summary=get_raw_summary(), active="dataset")


@app.route("/preprocessing")
def preprocessing_page():
    report = get_preprocess_report()
    return render_template("preprocessing.html", report=report, active="preprocessing")


@app.route("/eda")
def eda_page():
    eda_data = get_eda()
    return render_template("eda.html", eda=eda_data, active="eda")


@app.route("/linear-regression")
def linear_regression_page():
    res = get_linear_regression()
    return render_template("linear_regression.html", res=res, active="linear_regression")


@app.route("/logistic-regression")
def logistic_regression_page():
    res = get_logistic_regression()
    return render_template("logistic_regression.html", res=res, active="logistic_regression")


@app.route("/xgboost")
def xgboost_page():
    res = get_xgboost()
    return render_template("xgboost.html", res=res, active="xgboost")


@app.route("/adaboost")
def adaboost_page():
    res = get_adaboost()
    return render_template("adaboost.html", res=res, active="adaboost")


@app.route("/clustering")
def clustering_overview():
    return render_template("clustering.html", active="clustering",
                            all_features=ALL_CLUSTER_FEATURES, default_features=DEFAULT_FEATURES)


@app.route("/clustering/kmeans")
def clustering_kmeans():
    df = get_clean_df()
    k = int(request.args.get("k", 4))
    k = max(2, min(k, 10))
    feats = parse_features(request)
    res = run_kmeans(df, feats, k=k)
    return render_template("kmeans.html", res=res, k=k, active="kmeans",
                            all_features=ALL_CLUSTER_FEATURES, selected_features=feats)


@app.route("/clustering/kmeans-plus-plus")
def clustering_kmeans_pp():
    df = get_clean_df()
    k = int(request.args.get("k", 4))
    k = max(2, min(k, 10))
    feats = parse_features(request)
    res = run_kmeans_plusplus(df, feats, k=k)
    return render_template("kmeans_plus_plus.html", res=res, k=k, active="kmeans_pp",
                            all_features=ALL_CLUSTER_FEATURES, selected_features=feats)


@app.route("/clustering/hierarchical")
def clustering_hierarchical():
    df = get_clean_df()
    k = int(request.args.get("k", 4))
    k = max(2, min(k, 8))
    linkage = request.args.get("linkage", "ward")
    if linkage not in ("ward", "complete", "average"):
        linkage = "ward"
    feats = parse_features(request)
    res = run_hierarchical(df, feats, k=k, linkage=linkage)
    return render_template("hierarchical.html", res=res, k=k, linkage=linkage, active="hierarchical",
                            all_features=ALL_CLUSTER_FEATURES, selected_features=feats)


@app.route("/clustering/dbscan")
def clustering_dbscan():
    df = get_clean_df()
    eps = float(request.args.get("eps", 0.3))
    min_samples = int(request.args.get("min_samples", 5))
    feats = parse_features(request)
    res = run_dbscan(df, feats, eps=eps, min_samples=min_samples)
    return render_template("dbscan.html", res=res, eps=eps, min_samples=min_samples, active="dbscan",
                            all_features=ALL_CLUSTER_FEATURES, selected_features=feats)


@app.route("/clustering/animation")
def clustering_animation():
    anim = get_dbscan_animation()
    return render_template("dbscan_animation.html", anim_json=json.dumps(anim), anim=anim, active="animation")


@app.route("/model-comparison")
def model_comparison_page():
    lin = get_linear_regression()
    log = get_logistic_regression()
    xgb = get_xgboost()
    ada = get_adaboost()
    df = get_clean_df()
    km = run_kmeans(df, DEFAULT_FEATURES, k=4)
    kmpp = run_kmeans_plusplus(df, DEFAULT_FEATURES, k=4)
    hier = run_hierarchical(df, DEFAULT_FEATURES, k=4)
    db = run_dbscan(df, DEFAULT_FEATURES, eps=0.3, min_samples=5)

    cls_comp = classification_comparison(log, xgb, ada)
    reg_comp = regression_comparison(lin)
    clust_comp = clustering_comparison(km, kmpp, hier, db)

    return render_template("model_comparison.html", cls_comp=cls_comp, reg_comp=reg_comp,
                            clust_comp=clust_comp, active="model_comparison")


@app.route("/music-insights")
def music_insights_page():
    df = get_clean_df()
    log = get_logistic_regression()
    km = run_kmeans(df, DEFAULT_FEATURES, k=4)
    insights = generate_insights(df, log, km)
    return render_template("music_insights.html", insights=insights, active="insights")


@app.errorhandler(404)
def not_found(e):
    return render_template("404.html"), 404


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
