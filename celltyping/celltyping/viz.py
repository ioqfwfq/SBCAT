"""Label-QC figures: waveform split, interneuron verification, WaveMAP embedding.

Standard matplotlib; each function takes a features/label frame (and optional WaveMAP
extras) and returns a Figure. Colors follow the neural_01c convention
(narrow = red, broad = blue) for continuity with the WM-binding notebooks.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

_NB_COLORS = {"narrow": "#d62728", "broad": "#1f77b4"}
_VERIFY_FEATS = [("mean_rate_hz", "firing rate (Hz)"),
                 ("acg_refrac_ms", "ACG refractory (ms)"),
                 ("burst_index", "burst index"),
                 ("isi_cv", "ISI CV")]


def plot_waveform_split(feat_df, split_ms=None, ax=None):
    """Trough-to-peak histogram colored by narrow/broad, with the split boundary."""
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))
    v = feat_df[feat_df.get("wf_valid", True).astype(bool)] if "wf_valid" in feat_df else feat_df
    for g in ("broad", "narrow"):
        s = v.loc[v["wf_group"] == g, "trough_to_peak_ms"].dropna()
        if len(s):
            ax.hist(s, bins=40, color=_NB_COLORS[g], alpha=0.7, label=f"{g} (n={len(s)})")
    if split_ms is not None:
        ax.axvline(split_ms, color="k", ls="--", lw=1.5, label=f"split = {split_ms:.2f} ms")
    ax.set_xlabel("trough-to-peak (ms)")
    ax.set_ylabel("# units")
    ax.set_title("Waveform-width split")
    ax.legend(fontsize=8)
    return ax.figure


def plot_interneuron_verification(feat_df, group_col="wf_group"):
    """Boxplots of the FS-interneuron discriminating features, narrow vs broad."""
    fig, axes = plt.subplots(1, len(_VERIFY_FEATS), figsize=(3.2 * len(_VERIFY_FEATS), 3.8))
    for ax, (feat, label) in zip(np.atleast_1d(axes), _VERIFY_FEATS):
        if feat not in feat_df.columns:
            ax.set_visible(False)
            continue
        data, cols = [], []
        for g in ("broad", "narrow"):
            s = feat_df.loc[feat_df[group_col] == g, feat].dropna().astype(float)
            data.append(s.values)
            cols.append(_NB_COLORS[g])
        bp = ax.boxplot(data, labels=["broad", "narrow"], patch_artist=True, showfliers=False)
        for patch, c in zip(bp["boxes"], cols):
            patch.set_facecolor(c)
            patch.set_alpha(0.6)
        ax.set_title(label, fontsize=9)
    fig.suptitle("Narrow vs broad — putative-interneuron verification", fontsize=10)
    fig.tight_layout()
    return fig


def plot_mean_waveforms(X, groups, palette=None, ax=None):
    """Mean +/- SD normalized waveform per group/cluster (X rows aligned to `groups`)."""
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))
    X = np.asarray(X)
    groups = np.asarray(groups)
    t = np.arange(X.shape[1])
    for g in [x for x in np.unique(groups) if x == x]:      # skip NaN
        m = X[groups == g]
        col = (palette or {}).get(g)
        mu, sd = m.mean(0), m.std(0)
        ax.plot(t, mu, lw=2, color=col, label=f"{g} (n={len(m)})")
        ax.fill_between(t, mu - sd, mu + sd, color=col, alpha=0.2)
    ax.set_xlabel("sample")
    ax.set_ylabel("normalized amplitude")
    ax.set_title("Mean waveform per group")
    ax.legend(fontsize=8)
    return ax.figure


def plot_wavemap_embedding(embedding, feat_df=None, labels=None, color_by="wavemap_cluster"):
    """UMAP scatter colored three ways: WaveMAP cluster, wf_group, trough-to-peak."""
    embedding = np.asarray(embedding)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))

    # (1) by cluster
    if labels is not None:
        for c in np.unique(labels):
            m = labels == c
            axes[0].scatter(embedding[m, 0], embedding[m, 1], s=10, alpha=0.7, label=f"c{c}")
        axes[0].legend(fontsize=7, markerscale=1.5)
    axes[0].set_title("WaveMAP clusters")

    # (2) by narrow/broad and (3) by trough-to-peak require aligned feat_df rows
    if feat_df is not None and len(feat_df) == len(embedding):
        grp = feat_df["wf_group"].to_numpy()
        for g in ("broad", "narrow"):
            m = grp == g
            axes[1].scatter(embedding[m, 0], embedding[m, 1], s=10, alpha=0.7,
                            color=_NB_COLORS[g], label=g)
        axes[1].legend(fontsize=8)
        axes[1].set_title("narrow / broad")

        ttp = feat_df["trough_to_peak_ms"].to_numpy(dtype=float)
        sc = axes[2].scatter(embedding[:, 0], embedding[:, 1], s=10, c=ttp, cmap="viridis")
        fig.colorbar(sc, ax=axes[2], label="trough-to-peak (ms)")
        axes[2].set_title("trough-to-peak")

    for ax in axes:
        ax.set_xlabel("UMAP-1")
        ax.set_ylabel("UMAP-2")
    fig.tight_layout()
    return fig


def plot_waveform_grid(waveforms, unit_ids=None, order=None, ncols=10,
                       excluded=None, suspicion=None, title=None, max_plots=200):
    """Grid of individual mean waveforms for visual curation / inspection.

    waveforms : list of 1-D arrays (None entries skipped).
    order     : optional iterable of positions to draw first (e.g. worst-first).
    excluded  : optional set/list of positions or unit_ids to draw in red (to be culled).
    suspicion : optional per-position score printed under each waveform.
    """
    idx = list(order) if order is not None else list(range(len(waveforms)))
    idx = [i for i in idx if waveforms[i] is not None]
    if max_plots is not None:
        idx = idx[:max_plots]
    n = len(idx)
    if n == 0:
        fig, ax = plt.subplots()
        ax.set_title("no waveforms to show")
        return fig
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 1.3, nrows * 1.2), squeeze=False)
    exc = set(excluded) if excluded is not None else set()
    for k, i in enumerate(idx):
        ax = axes[k // ncols][k % ncols]
        w = np.asarray(waveforms[i], dtype=float).ravel()
        uid = unit_ids[i] if unit_ids is not None else i
        is_exc = (i in exc) or (uid in exc)
        ax.plot(w, lw=0.8, color="#d62728" if is_exc else "#333")
        lab = str(uid).split("_ses")[0].replace("sub-", "") if unit_ids is not None else str(i)
        if suspicion is not None:
            lab += f"\n{suspicion[i]:.2g}"
        ax.set_title(lab, fontsize=5)
        ax.set_xticks([])
        ax.set_yticks([])
    for k in range(n, nrows * ncols):
        axes[k // ncols][k % ncols].set_visible(False)
    if title:
        fig.suptitle(title, fontweight="bold", fontsize=10)
    fig.tight_layout()
    return fig


def plot_umap_colored(embedding, values, categorical=True, title=None, ax=None,
                      cmap="tab20", max_legend=15):
    """Recolor a UMAP embedding by an arbitrary per-unit value (batch-effect check).

    Protocol step 18: color by subject/session/region and look for segregation. With more
    than `max_legend` categorical levels the legend is suppressed (colors only). Continuous
    values (categorical=False) get a colorbar.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(5.2, 4.4))
    E = np.asarray(embedding)
    v = np.asarray(values)
    if categorical:
        levels = list(np.unique(v.astype(str)))
        cols = plt.get_cmap(cmap, max(len(levels), 1))
        for j, lev in enumerate(levels):
            m = v.astype(str) == lev
            ax.scatter(E[m, 0], E[m, 1], s=10, alpha=0.7, color=cols(j),
                       label=lev if len(levels) <= max_legend else None)
        if len(levels) <= max_legend:
            ax.legend(fontsize=6, markerscale=1.3, loc="best")
    else:
        sc = ax.scatter(E[:, 0], E[:, 1], s=10, c=v.astype(float), cmap="viridis")
        ax.figure.colorbar(sc, ax=ax)
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    if title:
        ax.set_title(title, fontsize=10)
    return ax.figure
