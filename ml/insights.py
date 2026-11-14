"""
ml/insights.py
===============
Generates plain-language, data-driven insight statements directly from the
cleaned dataset and the already-computed model/clustering results. Nothing
here is invented — every number quoted is pulled from a real aggregate.
"""

import pandas as pd


def generate_insights(df: pd.DataFrame, logreg: dict, kmeans: dict) -> list:
    insights = []

    # Genre with highest average danceability
    genre_dance = df.groupby("track_genre")["danceability"].mean()
    top_genre = genre_dance.idxmax()
    insights.append(
        f"'{top_genre}' tracks have the highest average danceability in the dataset "
        f"({genre_dance.max():.2f} vs. an overall average of {df['danceability'].mean():.2f})."
    )

    # Genre with highest average energy
    genre_energy = df.groupby("track_genre")["energy"].mean()
    top_energy_genre = genre_energy.idxmax()
    insights.append(
        f"'{top_energy_genre}' is the highest-energy genre on average ({genre_energy.max():.2f}), "
        f"compared to the dataset average of {df['energy'].mean():.2f}."
    )

    # High popularity characteristics vs overall
    threshold = logreg["threshold"]
    high_pop = df[df["popularity"] >= threshold]
    low_pop = df[df["popularity"] < threshold]
    insights.append(
        f"Tracks at or above the high-popularity threshold ({threshold:.0f}, the {logreg['percentile']}th percentile) "
        f"average {high_pop['danceability'].mean():.2f} danceability and {high_pop['energy'].mean():.2f} energy, "
        f"versus {low_pop['danceability'].mean():.2f} and {low_pop['energy'].mean():.2f} for lower-popularity tracks."
    )

    # Explicit tracks
    exp = df.groupby("explicit")["popularity"].mean()
    if 1 in exp.index and 0 in exp.index:
        direction = "higher" if exp[1] > exp[0] else "lower"
        insights.append(
            f"Explicit tracks average {direction} popularity ({exp[1]:.1f}) than non-explicit tracks ({exp[0]:.1f})."
        )

    # Correlation-based insight
    corr = df[["popularity", "danceability", "energy", "valence", "acousticness", "loudness"]].corr()["popularity"].drop("popularity")
    strongest = corr.abs().idxmax()
    direction = "positively" if corr[strongest] > 0 else "negatively"
    insights.append(
        f"Among audio features, '{strongest}' has the strongest linear relationship with popularity "
        f"(correlation = {corr[strongest]:.3f}, {direction} correlated) — though the relationship remains weak overall, "
        f"which is why the regression models on this dashboard show a modest R²."
    )

    # Clustering insight (describe the largest cluster from K-Means run)
    counts = kmeans["cluster_counts"]
    largest_cluster = max(counts, key=counts.get)
    insights.append(
        f"In the K-Means clustering (K={kmeans['k']}), cluster {largest_cluster} is the largest group with "
        f"{counts[largest_cluster]} of the {kmeans['n_points']} sampled tracks — silhouette score {kmeans['silhouette']}."
    )

    # Duration insight
    long_tracks = df[df["duration_ms"] > df["duration_ms"].quantile(0.9)]
    insights.append(
        f"The longest 10% of tracks (over {long_tracks['duration_ms'].min()/60000:.1f} minutes) "
        f"average {long_tracks['popularity'].mean():.1f} popularity, versus {df['popularity'].mean():.1f} overall."
    )

    return insights
