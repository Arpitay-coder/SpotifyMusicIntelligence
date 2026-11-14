"""
ml/dbscan_animation.py
=======================
Builds the step-by-step trace consumed by the "DBSCAN — Discovering Music
Communities" animation on the frontend (static/js/dbscan_animation.js).

Concept reference: the supplied dbscan_animation.html (play/pause/step/reset,
epsilon ring, core/border/noise coloring). This module re-implements the same
region-growing DBSCAN algorithm in Python, but runs it on REAL track data
(a sampled subset — 100 to 300 tracks, as specified) using two audio
features (danceability & energy) as 2D coordinates, and records track names
so the info panel can show real song titles instead of generic dot numbers.

Each step in the returned trace records: which track is "active", every
track's current label/core/visited state, and a human-readable message —
exactly the information static/js/dbscan_animation.js needs to redraw the
SVG frame by frame.
"""

import numpy as np
import pandas as pd


ANIMATION_SAMPLE_SIZE = 140
CANVAS_W, CANVAS_H = 640, 380
MARGIN = 40


def _project_to_canvas(x, y):
    """Map [0,1]x[0,1] audio-feature space onto SVG canvas coordinates."""
    px = MARGIN + x * (CANVAS_W - 2 * MARGIN)
    py = CANVAS_H - MARGIN - y * (CANVAS_H - 2 * MARGIN)
    return round(float(px), 1), round(float(py), 1)


def build_dbscan_animation(df: pd.DataFrame, eps_raw: float = 0.08, min_samples: int = 4,
                            random_state: int = 7) -> dict:
    feats = ["danceability", "energy"]
    sample = df.dropna(subset=feats + ["track_name", "artists"]).sample(
        n=min(ANIMATION_SAMPLE_SIZE, len(df)), random_state=random_state
    ).reset_index(drop=True)

    coords = sample[feats].values.astype(float)  # already 0-1 scale, no need to standardize for this illustrative view
    n = len(sample)

    def region_query(i):
        d = np.hypot(coords[:, 0] - coords[i, 0], coords[:, 1] - coords[i, 1])
        return list(np.where(d <= eps_raw)[0])

    visited = [False] * n
    labels = [None] * n
    core = [False] * n
    history = []

    def snap(active, msg):
        history.append({
            "active": int(active),
            "visited": visited.copy(),
            "labels": labels.copy(),
            "core": core.copy(),
            "msg": msg,
        })

    cluster_id = -1
    for i in range(n):
        if visited[i]:
            continue
        visited[i] = True
        nb = region_query(i)
        track = f"{sample.loc[i, 'track_name']} — {sample.loc[i, 'artists']}"
        snap(i, f"Visiting '{sample.loc[i, 'track_name']}' — {len(nb)} neighboring tracks within ε")

        if len(nb) < min_samples:
            labels[i] = -1
            snap(i, f"'{sample.loc[i, 'track_name']}' has too few musical neighbors → marked noise")
        else:
            cluster_id += 1
            core[i] = True
            labels[i] = cluster_id
            snap(i, f"'{sample.loc[i, 'track_name']}' is a core track → starts music community {cluster_id + 1}")
            queue = list(nb)
            qi = 0
            while qi < len(queue):
                j = queue[qi]
                qi += 1
                if not visited[j]:
                    visited[j] = True
                    jn = region_query(j)
                    snap(j, f"Expanding to '{sample.loc[j, 'track_name']}'")
                    if len(jn) >= min_samples:
                        core[j] = True
                        for x in jn:
                            if x not in queue:
                                queue.append(x)
                        snap(j, f"'{sample.loc[j, 'track_name']}' is also a core track → its neighbors join the queue")
                if labels[j] is None or labels[j] == -1:
                    labels[j] = cluster_id
                    role = "" if core[j] else " as a border track"
                    snap(j, f"'{sample.loc[j, 'track_name']}' joins community {cluster_id + 1}{role}")

    n_noise = sum(1 for l in labels if l == -1)
    snap(-1, f"Done — {cluster_id + 1} music communit{'y' if cluster_id == 0 else 'ies'} found, {n_noise} noise track(s)")

    points = []
    for i in range(n):
        px, py = _project_to_canvas(coords[i, 0], coords[i, 1])
        points.append({
            "x": px, "y": py,
            "track_name": str(sample.loc[i, "track_name"]),
            "artists": str(sample.loc[i, "artists"]),
            "danceability": round(float(sample.loc[i, "danceability"]), 3),
            "energy": round(float(sample.loc[i, "energy"]), 3),
        })

    # eps radius in canvas pixel units (x and y scaled the same since canvas area is square-ish per axis)
    eps_px = eps_raw * (CANVAS_W - 2 * MARGIN)

    return {
        "points": points,
        "history": history,
        "eps": eps_raw,
        "eps_px": round(float(eps_px), 1),
        "min_samples": min_samples,
        "n_points": n,
        "canvas": {"w": CANVAS_W, "h": CANVAS_H},
        "n_clusters_found": cluster_id + 1,
        "n_noise": n_noise,
    }
