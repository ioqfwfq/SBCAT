"""WaveMAP: data-driven cell classes from spike waveforms.

Faithful reimplementation of Lee et al. 2021 (eLife) / the STAR-Protocols recipe:
    1. preprocess   align each mean waveform to its trough, crop a common window,
                    mean-subtract, normalize each waveform to max |amplitude| = 1
    2. UMAP         build the fuzzy-simplicial graph (n_neighbors=15)
    3. Louvain      community-detect on that graph -> cluster labels (resolution chosen
                    per-dataset by sweep; see the convention warning in run_wavemap)
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
* The Louvain `resolution` here is python-louvain's, NOT the protocol's cylouvain: higher
  -> MORE clusters, the reverse of what the paper's text says. The paper's value of 2 is
  therefore not portable; derive it with resolution_sweep(). See run_wavemap's docstring.
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


def align_crop_waveforms(waveforms, fs_hz: float = OSORT_FS_HZ,
                         pre_ms: float = 0.3, post_ms: float = 1.7, n_out: int | None = None):
    """Trough-align + crop each waveform to a common window (protocol steps 2-4; no normalization).

    Puts the trough `pre_ms` into the window (protocol: 0.3 ms) and takes `pre_ms+post_ms`
    (~2.0 ms) around it; optionally resamples to `n_out` samples (for pooling mixed fs).
    Returns (X_cropped, valid) with `valid` a boolean mask aligned to the input list.
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
    return X, valid


def normalize_waveforms(X):
    """Protocol steps 9-10: mean-subtract each waveform, then divide by its max |amplitude|."""
    X = np.asarray(X, dtype=float)
    if X.size == 0:
        return X
    X = X - X.mean(axis=1, keepdims=True)
    scale = np.max(np.abs(X), axis=1, keepdims=True)
    scale[scale == 0] = 1.0
    return X / scale


def preprocess_waveforms(waveforms, fs_hz: float = OSORT_FS_HZ,
                         pre_ms: float = 0.3, post_ms: float = 1.7, n_out: int | None = None):
    """Full WaveMAP input pipeline: align_crop_waveforms then normalize_waveforms.

    Returns (X, valid) where X is (n_valid, L) mean-subtracted, max-normalized waveforms
    and `valid` maps rows back to the input list. Use the two component functions directly
    if you want to inspect the aligned-but-unnormalized waveforms (e.g. for curation).
    """
    X, valid = align_crop_waveforms(waveforms, fs_hz=fs_hz, pre_ms=pre_ms,
                                    post_ms=post_ms, n_out=n_out)
    return (normalize_waveforms(X) if X.size else X), valid


# ── 2-3. UMAP graph -> Louvain communities ───────────────────────────────────
def run_wavemap(X, n_neighbors: int = 15, min_dist: float = 0.1, metric: str = "euclidean",
                resolution: float = 2.0, random_state: int = 42) -> dict:
    """UMAP fuzzy graph + Louvain community detection (the WaveMAP core).

    Returns dict(labels, embedding, graph, reducer). `labels` are 0-based cluster ids
    aligned to X rows.

    RESOLUTION CONVENTION -- read before tuning. We use python-louvain (`community`),
    where `resolution` multiplies the NULL-MODEL term in __one_level:
        incr = remove_cost + dnc - resolution * degrees[com] * degc_totw
    so higher resolution -> MORE, smaller clusters (standard Reichardt-Bornholdt gamma).
    Verified empirically: on a UMAP fuzzy graph, r=0.5->3, r=1->3, r=2->7, r=3->16, r=5->31.

    The STAR-Protocols paper uses a DIFFERENT package, `cylouvain`, and states the opposite
    direction ("larger resolution results in fewer" clusters). Its default of 2 was tuned
    under that convention and does NOT transfer here -- at resolution=2 this function
    over-splits badly (856 units -> 21 clusters). cylouvain does not build on Python 3.13,
    so the two cannot be compared directly. Do not inherit the number 2: choose `resolution`
    from resolution_sweep() / tune_wavemap() on your own data, as the protocol itself
    sanctions ("this can be changed by the user").
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
                graph=reducer.graph_, reducer=reducer, X=np.asarray(X))


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


# ── resolution selection (protocol Fig. 4: modularity vs number of clusters) ──
def resolution_sweep(X, resolutions=(0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0),
                     n_neighbors: int = 15, min_dist: float = 0.1,
                     metric: str = "euclidean", random_state: int = 42):
    """Sweep Louvain resolution on ONE UMAP graph -> DataFrame(resolution, n_clusters, modularity).

    The protocol chooses `resolution` by jointly maximizing modularity while minimizing
    the number of clusters (Lee et al. 2021, Fig. 4). UMAP is fit once; only Louvain re-runs,
    so this is cheap. Use it to justify the resolution passed to run_wavemap().

    Grid runs DOWN from 3.0 because higher resolution -> MORE clusters in python-louvain
    (see run_wavemap's convention warning); the few-cluster end of the sweep is below 1.0,
    not above it. `modularity` is plain Newman modularity at gamma=1, so it is comparable
    across rows.
    """
    import umap
    import networkx as nx
    import community as community_louvain
    import pandas as pd

    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, metric=metric,
                        random_state=random_state)
    reducer.fit(X)
    G = nx.from_scipy_sparse_array(reducer.graph_)
    rows = []
    for r in resolutions:
        part = community_louvain.best_partition(G, resolution=float(r), random_state=random_state)
        rows.append(dict(resolution=float(r),
                         n_clusters=len(set(part.values())),
                         modularity=float(community_louvain.modularity(part, G))))
    return pd.DataFrame(rows)


# ── batch-effect check: is a cluster dominated by one nuisance-factor level? ──
def cluster_composition(labels, factor, factor_name: str = "factor"):
    """Cluster x factor contingency + per-cluster dominance + chi-square.

    Protocol step 18 (CRITICAL) batch-effect check: does a WaveMAP cluster segregate by a
    nuisance variable (subject, session, region) rather than cell type? `max_frac` is, per
    cluster, the fraction of its units from the single most common factor level — values
    near 1 flag a cluster that is likely a batch artifact rather than a real waveform type.
    Returns dict(table, frac, max_frac, chi2, p, dof).
    """
    import pandas as pd
    from scipy.stats import chi2_contingency

    tab = pd.crosstab(pd.Series(list(labels), name="cluster"),
                      pd.Series(list(factor), name=factor_name))
    frac = tab.div(tab.sum(axis=1).replace(0, 1), axis=0)
    max_frac = frac.max(axis=1)
    try:
        chi2, p, dof, _ = chi2_contingency(tab)
    except ValueError:
        chi2, p, dof = float("nan"), float("nan"), 0
    return dict(table=tab, frac=frac, max_frac=max_frac,
                chi2=float(chi2), p=float(p), dof=int(dof))


# ── putative cell-type label per cluster (from median physiology profile) ──
def label_clusters(profile, split_ms: float, rate_fs_hz: float = 5.0, burst_hi: float = 0.15):
    """Assign a putative cell-type label to each WaveMAP cluster from its median profile.

    Transparent, tunable rules on the per-cluster medians (from characterize_clusters):
      * trough_to_peak_ms < split_ms  -> narrow (putative interneuron);
          mean_rate_hz >= rate_fs_hz  -> "fast-spiking ...", else "narrow-spiking ..."
      * otherwise                     -> broad (putative pyramidal);
          burst_index >= burst_hi     -> "broad bursting ...", else "broad regular-spiking ..."

    These are PUTATIVE summaries — the mean waveform, SHAP importances, and physiology
    profile are the evidence. Triphasic/axonal types are best assigned by eye from the
    per-cluster mean-waveform grid, then edited here. Returns dict{cluster -> label}.
    """
    import numpy as _np
    labels = {}
    for cid, row in profile.iterrows():
        ttp = row.get("trough_to_peak_ms", _np.nan)
        rate = row.get("mean_rate_hz", _np.nan)
        burst = row.get("burst_index", _np.nan)
        if _np.isfinite(ttp) and ttp < split_ms:
            labels[cid] = ("fast-spiking (putative interneuron)"
                           if _np.isfinite(rate) and rate >= rate_fs_hz
                           else "narrow-spiking (putative interneuron)")
        else:
            labels[cid] = ("broad bursting (putative pyramidal)"
                           if _np.isfinite(burst) and burst >= burst_hi
                           else "broad regular-spiking (putative pyramidal)")
    return labels


# ── parameter tuning: grid over (n_neighbors, resolution) scored on 3 axes ────
def tune_wavemap(X, n_neighbors_grid=(15, 20, 30, 50),
                 resolution_grid=(0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0), seeds=(1, 2, 3),
                 feat_df=None, factor=None,
                 physio_feats=("trough_to_peak_ms", "half_width_ms", "peak_trough_ratio",
                               "repol_slope", "mean_rate_hz", "isi_cv", "burst_index"),
                 min_dist: float = 0.1, metric: str = "euclidean", verbose: bool = True):
    """Score a (n_neighbors x resolution) grid for WaveMAP so params are chosen from evidence.

    WaveMAP is unsupervised, so there is no accuracy to maximize -- the goal is the FEWEST
    clusters that are stable, well-separated, physiologically distinct, and not batch
    artifacts. Per grid cell this returns:
        n_clusters       from the first seed
        modularity       Louvain modularity (graph-native separation), first seed
        silhouette       silhouette on the UMAP embedding, first seed (visual separation;
                         distorted by UMAP, use as a rough proxy)
        stability_ari    mean pairwise Adjusted Rand Index of labels ACROSS `seeds`
                         (the key test -- clusters that reshuffle per seed are not real)
        min_physio_dist  min pairwise distance between clusters' z-scored median-physiology
                         profiles (near 0 => two clusters are near-duplicates => over-split)
        max_factor_frac  max over clusters of the single-`factor`-level fraction
                         (near 1 => a cluster is dominated by one subject/session => batch)

    UMAP is fit once per (n_neighbors, seed); Louvain re-runs per resolution (cheap).
    Pass feat_df (rows aligned to X) for min_physio_dist and factor (e.g. subject) for
    max_factor_frac. Returns a DataFrame, one row per grid cell.

    Read it: prefer high stability_ari + decent silhouette/modularity + low max_factor_frac
    + high min_physio_dist, choosing the SMALLEST n_clusters among cells that tie.
    """
    import numpy as np
    import pandas as pd
    import umap
    import networkx as nx
    import community as community_louvain
    from sklearn.metrics import adjusted_rand_score, silhouette_score
    from scipy.spatial.distance import pdist

    X = np.asarray(X)
    seeds = list(seeds)
    labels_store, embed_store, mod_store = {}, {}, {}
    for nn in n_neighbors_grid:
        for seed in seeds:
            reducer = umap.UMAP(n_neighbors=int(nn), min_dist=min_dist, metric=metric,
                                random_state=int(seed))
            embed_store[(nn, seed)] = np.asarray(reducer.fit_transform(X))
            G = nx.from_scipy_sparse_array(reducer.graph_)
            for res in resolution_grid:
                part = community_louvain.best_partition(G, resolution=float(res),
                                                        random_state=int(seed))
                labels_store[(nn, res, seed)] = np.array(
                    [part.get(i, -1) for i in range(X.shape[0])], dtype=int)
                mod_store[(nn, res, seed)] = float(community_louvain.modularity(part, G))
        if verbose:
            print(f"  n_neighbors={nn}: fit {len(seeds)} seeds x {len(resolution_grid)} resolutions")

    base = seeds[0]
    feats = [f for f in physio_feats if feat_df is not None and f in feat_df.columns]
    rows = []
    for nn in n_neighbors_grid:
        emb0 = embed_store[(nn, base)]
        for res in resolution_grid:
            lab0 = labels_store[(nn, res, base)]
            k = int(len(set(lab0)))
            sil = float(silhouette_score(emb0, lab0)) if k > 1 else np.nan
            aris = [adjusted_rand_score(labels_store[(nn, res, seeds[i])],
                                        labels_store[(nn, res, seeds[j])])
                    for i in range(len(seeds)) for j in range(i + 1, len(seeds))]
            stab = float(np.mean(aris)) if aris else np.nan

            min_phys = np.nan
            if feats and k > 1:
                tmp = feat_df[feats].copy()
                tmp["_c"] = lab0
                prof = tmp.groupby("_c").median()
                z = (prof - prof.mean()) / prof.std(ddof=0).replace(0, 1.0)
                z = z.fillna(0.0)
                if len(z) > 1:
                    min_phys = float(pdist(z.to_numpy()).min())

            max_frac = np.nan
            if factor is not None:
                comp = cluster_composition(lab0, factor)
                max_frac = float(comp["max_frac"].max())

            rows.append(dict(n_neighbors=int(nn), resolution=float(res), n_clusters=k,
                             modularity=mod_store[(nn, res, base)], silhouette=sil,
                             stability_ari=stab, min_physio_dist=min_phys,
                             max_factor_frac=max_frac))
    return pd.DataFrame(rows)
