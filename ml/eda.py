"""
ml/eda.py
=========
Builds the interactive Plotly figures and aggregate tables for the EDA page.
Every function returns a plotly Figure; app.py converts these to JSON with
fig.to_json() for the template to render client-side via Plotly.js.
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

DARK_TEMPLATE = "plotly_dark"
ACCENT = "#1DB954"
PALETTE = px.colors.sequential.Greens[::-1]


def _style(fig, title=None, height=380):
    fig.update_layout(
        template=DARK_TEMPLATE,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#E6EDF3", family="Inter, -apple-system, sans-serif"),
        title=title,
        height=height,
        margin=dict(l=40, r=20, t=50, b=40),
        hoverlabel=dict(bgcolor="#181818", font_size=12),
    )
    return fig


def distribution_fig(df, col, title, color=ACCENT, nbins=50):
    """Pre-binned histogram (numpy) so the page ships bin counts, not all
    90k raw values, per-figure."""
    counts, edges = np.histogram(df[col].dropna(), bins=nbins)
    centers = (edges[:-1] + edges[1:]) / 2
    widths = edges[1:] - edges[:-1]
    fig = go.Figure(go.Bar(x=centers, y=counts, width=widths, marker_color=color,
                            marker_line_width=0, opacity=0.9,
                            hovertemplate=f"{col}: %{{x:.3f}}<br>count: %{{y}}<extra></extra>"))
    fig.update_layout(bargap=0.02)
    return _style(fig, title)


def duration_distribution_fig(df, nbins=50):
    d = df["duration_ms"] / 60000
    counts, edges = np.histogram(d.dropna(), bins=nbins)
    centers = (edges[:-1] + edges[1:]) / 2
    widths = edges[1:] - edges[:-1]
    fig = go.Figure(go.Bar(x=centers, y=counts, width=widths, marker_color=ACCENT,
                            marker_line_width=0, opacity=0.9,
                            hovertemplate="duration: %{x:.2f} min<br>count: %{y}<extra></extra>"))
    fig.update_layout(bargap=0.02, xaxis_title="Duration (minutes)")
    return _style(fig, "Track Duration Distribution")


def top_artists_fig(df, n=15):
    g = (df.groupby("artists")["popularity"].agg(["mean", "count"])
         .query("count >= 3").sort_values("mean", ascending=False).head(n).reset_index())
    fig = px.bar(g, x="mean", y="artists", orientation="h", color="mean",
                 color_continuous_scale="Greens", labels={"mean": "Avg Popularity", "artists": ""})
    fig.update_layout(yaxis=dict(autorange="reversed"))
    return _style(fig, f"Top {n} Artists by Average Popularity (min. 3 tracks)", height=450)


def top_tracks_fig(df, n=15):
    g = df.nlargest(n, "popularity")[["track_name", "artists", "popularity"]]
    fig = px.bar(g, x="popularity", y="track_name", orientation="h", color="popularity",
                 color_continuous_scale="Greens", hover_data=["artists"], labels={"track_name": ""})
    fig.update_layout(yaxis=dict(autorange="reversed"))
    return _style(fig, f"Top {n} Tracks by Popularity", height=450)


def genre_counts_fig(df, n=15):
    g = df["track_genre"].value_counts().head(n).reset_index()
    g.columns = ["track_genre", "count"]
    fig = px.bar(g, x="count", y="track_genre", orientation="h", color="count",
                 color_continuous_scale="Greens", labels={"track_genre": ""})
    fig.update_layout(yaxis=dict(autorange="reversed"))
    return _style(fig, f"Top {n} Most Common Genres", height=450)


def genre_vs_metric_fig(df, metric, n=15, title=None):
    top_genres = df["track_genre"].value_counts().head(n).index
    g = df[df["track_genre"].isin(top_genres)].groupby("track_genre")[metric].mean().sort_values(ascending=False).reset_index()
    fig = px.bar(g, x=metric, y="track_genre", orientation="h", color=metric,
                 color_continuous_scale="Greens", labels={"track_genre": ""})
    fig.update_layout(yaxis=dict(autorange="reversed"))
    return _style(fig, title or f"Genre vs Average {metric.title()}", height=450)


def correlation_heatmap_fig(df, cols):
    corr = df[cols].corr()
    fig = go.Figure(data=go.Heatmap(
        z=corr.values, x=corr.columns, y=corr.columns,
        colorscale="Greens", zmin=-1, zmax=1,
        text=corr.round(2).values, texttemplate="%{text}", textfont=dict(size=9),
    ))
    return _style(fig, "Correlation Heatmap — Audio Features", height=520)


def explicit_comparison_fig(df):
    g = df.groupby("explicit")[["popularity", "energy", "danceability", "valence"]].mean().reset_index()
    g["explicit"] = g["explicit"].map({0: "Not Explicit", 1: "Explicit"})
    g_melt = g.melt(id_vars="explicit", var_name="metric", value_name="value")
    fig = px.bar(g_melt, x="metric", y="value", color="explicit", barmode="group",
                 color_discrete_sequence=[ACCENT, "#7F77DD"])
    return _style(fig, "Explicit vs Non-Explicit Tracks")


def audio_feature_box_fig(df, cols, sample_n=4000):
    """Boxplot on a random sample (quartiles/outlier shape are stable at this
    size, and it keeps the page payload small compared to embedding all
    ~90k rows per feature)."""
    d = df[cols].sample(n=min(sample_n, len(df)), random_state=42).melt(var_name="feature", value_name="value")
    fig = px.box(d, x="feature", y="value", color="feature",
                 color_discrete_sequence=px.colors.qualitative.Set2)
    fig.update_layout(showlegend=False)
    return _style(fig, f"Audio Feature Distributions (Boxplot, sample of {min(sample_n, len(df))})", height=450)


def run_eda(df: pd.DataFrame) -> dict:
    """Compute everything the EDA page needs in one pass."""
    audio_cols = ["danceability", "energy", "loudness", "speechiness", "acousticness",
                  "instrumentalness", "liveness", "valence", "tempo"]

    overview = {
        "n_tracks": int(len(df)),
        "n_artists": int(df["artists"].nunique()),
        "n_albums": int(df["album_name"].nunique()),
        "n_genres": int(df["track_genre"].nunique()),
        "explicit_pct": round(float(100 * df["explicit"].mean()), 2),
        "avg_duration_min": round(float(df["duration_ms"].mean() / 60000), 2),
    }

    figs = {
        "popularity_dist": distribution_fig(df, "popularity", "Popularity Distribution"),
        "danceability_dist": distribution_fig(df, "danceability", "Danceability Distribution", "#1ED760"),
        "energy_dist": distribution_fig(df, "energy", "Energy Distribution", "#17A34A"),
        "valence_dist": distribution_fig(df, "valence", "Valence (Mood) Distribution", "#0F9D58"),
        "tempo_dist": distribution_fig(df, "tempo", "Tempo (BPM) Distribution", "#66E39A"),
        "duration_dist": duration_distribution_fig(df),
        "top_artists": top_artists_fig(df),
        "top_tracks": top_tracks_fig(df),
        "genre_counts": genre_counts_fig(df),
        "genre_vs_popularity": genre_vs_metric_fig(df, "popularity"),
        "genre_vs_energy": genre_vs_metric_fig(df, "energy"),
        "genre_vs_danceability": genre_vs_metric_fig(df, "danceability"),
        "correlation_heatmap": correlation_heatmap_fig(df, audio_cols + ["popularity"]),
        "explicit_comparison": explicit_comparison_fig(df),
        "audio_feature_box": audio_feature_box_fig(df, audio_cols),
    }
    figs_json = {k: v.to_json() for k, v in figs.items()}

    return {"overview": overview, "figs": figs_json}
