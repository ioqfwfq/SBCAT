"""WaveMAP: data-driven cell classes from spike waveforms.

Faithful reimplementation of Lee et al. 2021 (eLife) / the STAR-Protocols recipe:
    1. preprocess   align each mean waveform to its trough, crop a common window,
                    mean-subtract, normalize each waveform to max |amplitude| = 1
    2. UMAP         build the fuzzy-simplicial graph (n_neighbors=15)
    3. Louvain      community-detect on that graph (resolution=2) -> cluster labels
    4. interpret    XGBoost waveform->cluster classifier + TreeSHAP feature importance

Notes
-----
* Inputs are the polarity-normalized mean waveforms from features.build_unit_features
  (return_waveforms=True) — already trough-down, so WaveMAP clusters on shape, not
  recording polarity.
* Cross-dataset caveat (from the protocol): waveforms from recordings with different
  high-pass filters / sampling produce artifactual "mirrored" structure. Crop+resample
  to a common window here, and prefer running per-dataset then comparing clusters, or
  verify no dataset-driven splitting before pooling. `dataset` labels can be passed to
  `check_dataset_confound` for that check.
* ~300 high-quality units are recommended for stable results.
"""

from __future__ import annotations

import numpy as np

from .features import OSORT_FS_HZ


# ── 1. preprocessing ─────────────────────────────────────────────────────────
def _crop_around_trough(w: np.ndarray, pre: int, post: int) -> np.ndarray:
    """Crop [trough-pre, trough+post]; edge-pad when the window runs off either end."""
    w = np.asarray(w, dtype=float).ravel()
    ti = int(np.argmin(w))
    start, end = ti - pre, ti + post
    left_pad = max(0, -start)
    right_pad = max(0, end - w.size)
    seg = w[max(0, start):min(w.size, end)]
    if left_pad or right_pad:
        seg = np.concatenate([np.full(left_pad, seg[0]), seg, np.full(right_pad, seg[-1])])
    return seg[: pre + post]


def preprocess_waveforms(waveforms, fs_hz: float = OSORT_FS_HZ,
                         pre_ms: float = 0.6, post_ms: float = 1.2, n_out: int | None = None):
    """List of polarity-normalized mean waveforms -> WaveMAP input matrix.

    Returns (X, valid) where X is (n_valid, L) and `valid` is a boolean mask aligned
    to the input list (so cluster labels can be mapped back to units).

    Each row: cropped to a common trough-aligned window, optionally resampled to
    `n_out` samples (use when pooling datasets with different fs), then mean-subtracted
    and scaled to max |amplitude| = 1.
    """
    pre = int(round(pre_ms / 1000.0 * fs_hz))
    post = int(round(post_ms / 1000.0 * fs_hz))
    rows, valid = [], []
    for w in waveforms:
        if w is None:
            valid.append(False)
            continue
        w = np.asarray(w, dtype=float).ravel()
        if w.size < 8 or not np.all(np.isfinite(w)):
            valid.append(False)
            continue
        rows.append(_crop_around_trough(w, pre, post))
        valid.append(True)
    valid = np.asarray(valid, dtype=bool)
    if not rows:
        return np.empty((0, pre + post)), valid
    X = np.vstack(rows)

    if n_out and n_out != X.shape[1]:
        from scipy.signal import resample
        X = resample(X, n_out, axis=1)

    X = X - X.mean(axis=1, keepdims=True)
    scale = np.max(np.abs(X), axis=1, keepdims=True)
    scale[scale == 0] = 1.0
    X = X / scale
    return X, valid


# ── 2-3. UMAP graph -> Louvain communities ───────────────────────────────────
def run_wavemap(X, n_neighbors: int = 15, min_dist: float = 0.1, metric: str = "euclidean",
                resolution: float = 2.0, random_state: int = 42) -> dict:
    """UMAP fuzzy graph + Louvain community detection (the WaveMAP core).

    Returns dict(labels, embedding, graph, reducer). `labels` are 0-based cluster ids
    aligned to X rows. `resolution` up -> fewer clusters (protocol default 2).
    """
    import umap
    import networkx as nx
    import community as community_louvain   # python-louvain

    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, metric=metric,
                        random_state=random_state)
    embedding = reducer.fit_transform(X)

    # reducer.graph_ is the weighted fuzzy-simplicial set (scipy sparse).
    G = nx.from_scipy_sparse_array(reducer.graph_)
    partition = community_louvain.best_partition(G, resolution=resolution,
                                                 random_state=random_state)
    labels = np.array([partition.get(i, -1) for i in range(X.shape[0])], dtype=int)
    return dict(labels=labels, embedding=np.asarray(embedding),
                graph=reducer.graph_, reducer=reducer)


# ── 4. interpretation: which waveform samples define the clusters ─────────────
def cluster_waveform_importance(X, labels, n_estimators: int = 300, max_depth: int = 3,
                                random_state: int = 42) -> dict:
    """XGBoost waveform->cluster classifier + TreeSHAP importance per time sample.

    Returns dict(model, shap_values, mean_abs_shap, accuracy). Optional step — needs
    xgboost + shap. mean_abs_shap[j] ranks how informative sample j is overall.
    """
    import xgboost as xgb
    import shap
    from sklearn.model_selection import cross_val_score

    _, y = np.unique(np.asarray(labels), return_inverse=True)   # contiguous 0..k-1 for xgboost
    clf = xgb.XGBClassifier(n_estimators=n_estimators, max_depth=max_depth,
                            random_state=random_state,
                            eval_metric="mlogloss", tree_method="hist")
    acc = float(np.mean(cross_val_score(clf, X, y, cv=5)))
    clf.fit(X, y)
    sv_arr = np.asarray(shap.TreeExplainer(clf).shap_values(X))
    # collapse |SHAP| over every axis except the L-length waveform-sample axis -> (L,)
    L = X.shape[1]
    ax = [i for i, s in enumerate(sv_arr.shape) if s == L][-1]
    other = tuple(i for i in range(sv_arr.ndim) if i != ax)
    mean_abs = np.abs(sv_arr).mean(axis=other)
    return dict(model=clf, shap_values=sv_arr, mean_abs_shap=mean_abs, accuracy=acc)


# ── cluster characterization + cross-dataset confound check ──────────────────
def characterize_clusters(feat_df, cluster_col: str = "wavemap_cluster",
                          feats=("trough_to_peak_ms", "half_width_ms", "mean_rate_hz",
                                 "isi_cv", "burst_index", "acg_refrac_ms")):
    """Per-cluster median of the physiology features (name clusters by their profile)."""
    import pandas as pd
    cols = [f for f in feats if f in feat_df.columns]
    g = feat_df.dropna(subset=[cluster_col]).groupby(cluster_col)
    out = g[cols].median()
    out.insert(0, "n", g.size())
    return out


def check_dataset_confound(labels, dataset):
    """Cluster x dataset contingency + chi-square: is any WaveMAP cluster dataset-specific?

    A near block-diagonal table (each cluster from one dataset) flags the artifactual
    mirroring the protocol warns about — run WaveMAP per dataset instead of pooled.
    """
    import pandas as pd
    from scipy.stats import chi2_contingency
    tab = pd.crosstab(pd.Series(labels, name="cluster"), pd.Series(dataset, name="dataset"))
    chi2, p, dof, _ = chi2_contingency(tab)
    return dict(table=tab, chi2=float(chi2), p=float(p), dof=int(dof))
