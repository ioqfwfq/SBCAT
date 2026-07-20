"""ACG-based temporal-dynamics cell typing — the third, spike-timing line.

An independent axis from the two waveform-shape lines (width split, WaveMAP): this one
reads only *spike timing*, so it also runs on datasets without stored waveforms. Follows
the CellExplorer/Buzs-lab autocorrelogram (ACG) protocol as used in the reference:

    1. Per unit, build two ACGs from all spike times (features.autocorrelogram, windowed):
         narrow  +/-50 ms  @ 0.5 ms bins  -> burst / refractory structure  (fit target)
         wide    +/-500 ms @ 1.0 ms bins  -> rhythmicity (theta)           (theta index)
    2. Divide each by (n_spikes * bin_width_s) -> spike-density (conditional-rate) ACG;
       divide by its max -> unit-normalized [0, 1] shape.
    3. Fit the narrow ACG (RAW spike-density, not the normalized one) with a bounded
       CellExplorer-style triple-exponential -> tau_rise, tau_decay, R^2.
    4. Drop poor fits (R^2 < 0.3).
    5. Spectral clustering (robust to noise, captures nonlinear structure) on
       [firing rate, mean ACG, tau_rise] -> two groups = putative pyramidal (PY) /
       interneuron (IN). UMAP is used ONLY to visualize the spectral labels.

Deliverable: a per-unit label table keyed by unit_id (acg_cluster / acg_type + the ACG
features), joinable to the width and WaveMAP tables for a three-way consensus.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .features import autocorrelogram

# CellExplorer ACG specs (match the reference exactly)
NARROW = dict(win_ms=50.0, bin_ms=0.5)      # 201 pts across +/-50 ms
WIDE = dict(win_ms=500.0, bin_ms=1.0)       # 1001 pts across +/-500 ms


# ── restrict spikes to in-trial windows (discard inter-trial interval) ───────
def restrict_to_trials(spike_times, trials_df, start_col: str = "start_time",
                       stop_col: str = "stop_time"):
    """Keep only spikes inside any trial's [start, stop]; drop the inter-trial interval.

    Returns (in_trial_spike_times, total_in_trial_seconds). Trial windows are assumed
    non-overlapping (true for this task). If no usable trials, returns the input unchanged
    with duration NaN. The ACG's windowed coincidence count is unaffected by the removed
    gaps (ITI gaps >> the ACG window), so this cleanly limits the ACG to task periods.
    """
    st = np.sort(np.asarray(spike_times, dtype=float))
    if trials_df is None or start_col not in trials_df or stop_col not in trials_df:
        return st, np.nan
    a = pd.to_numeric(trials_df[start_col], errors="coerce").to_numpy(dtype=float)
    b = pd.to_numeric(trials_df[stop_col], errors="coerce").to_numpy(dtype=float)
    ok = np.isfinite(a) & np.isfinite(b) & (b > a)
    a, b = a[ok], b[ok]
    if a.size == 0:
        return st, np.nan
    order = np.argsort(a)
    a, b = a[order], b[order]
    idx = np.searchsorted(a, st, side="right") - 1              # last trial starting at/<= spike
    keep = (idx >= 0) & (st <= b[np.clip(idx, 0, b.size - 1)])
    return st[keep], float((b - a).sum())


# ── build one ACG (spike-density + unit-normalized) ──────────────────────────
def compute_acg(spike_times_s, win_ms: float, bin_ms: float) -> dict:
    """Symmetric ACG for one unit -> counts / spike-density / unit-normalized.

    spike-density = counts / (n_spikes * bin_width_s)  (conditional rate; asymptote ~= rate)
    norm          = density / max(density)             (shape, firing-rate invariant, [0,1])
    """
    st = np.asarray(spike_times_s, dtype=float)
    n = int(np.sum(np.isfinite(st)))
    counts, centers = autocorrelogram(st, bin_ms=bin_ms, win_ms=win_ms)
    density = counts.astype(float)
    if n > 0:
        density = density / (n * (bin_ms / 1000.0))
    peak = float(density.max()) if density.size else 0.0
    norm = density / peak if peak > 0 else density
    return dict(counts=counts, centers=centers, density=density, norm=norm,
                n_spikes=n, bin_ms=bin_ms, win_ms=win_ms)


# ── CellExplorer-style triple-exponential fit of the narrow ACG ──────────────
def _acg_triple_exp(t, tau_decay, tau_rise, c, d, t0, tau_burst, h, rate):
    """Rise-then-decay toward an asymptotic rate, plus a Gaussian burst term (positive lags, ms)."""
    core = c * (np.exp(-(t - t0) / tau_decay) - d * np.exp(-(t - t0) / tau_rise))
    burst = h * np.exp(-((t - t0) ** 2) / (tau_burst ** 2))
    return np.maximum(core + burst + rate, 0.0)


def fit_acg(density, centers_ms, fit_max_ms: float = 50.0) -> dict:
    """Fit the positive-lag narrow ACG (raw spike-density). Returns tau_rise/tau_decay/R^2.

    Bounded non-linear least squares (scipy trf), CellExplorer-style init/bounds. On any
    failure returns NaNs with fit_ok=False. R^2 is on the fit region; gate with R^2>=0.3.
    """
    out = dict(tau_rise=np.nan, tau_decay=np.nan, tau_burst=np.nan, refrac_ms=np.nan,
               burst_amp=np.nan, asymptote_hz=np.nan, acg_r2=np.nan, fit_ok=False, fit=None,
               fit_t=None)
    lag = np.asarray(centers_ms, dtype=float)
    y_all = np.asarray(density, dtype=float)
    m = (lag > 0) & (lag <= fit_max_ms)
    t, y = lag[m], y_all[m]
    if t.size < 8 or not np.isfinite(y).all() or y.max() <= 0:
        return out

    from scipy.optimize import curve_fit
    ymax = float(y.max())
    tail_m = t >= 0.8 * fit_max_ms
    tail = float(np.mean(y[tail_m])) if tail_m.any() else float(np.median(y))
    #      tau_decay tau_rise  c      d    t0  tau_burst   h            rate
    p0 = [15.0,      1.0,   ymax,   1.0, 1.0,  3.0,     0.3 * ymax, max(tail, 1e-6)]
    lo = [1.0,       0.1,   0.0,    0.0, 0.0,  0.1,     0.0,        0.0]
    hi = [500.0,     50.0,  10 * ymax + 1e-9, 30.0, 15.0, 20.0, 10 * ymax + 1e-9, 5 * ymax + 1e-9]
    try:
        popt, _ = curve_fit(_acg_triple_exp, t, y, p0=p0, bounds=(lo, hi), maxfev=10000)
    except Exception:
        return out
    yhat = _acg_triple_exp(t, *popt)
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    out.update(tau_decay=float(popt[0]), tau_rise=float(popt[1]), refrac_ms=float(popt[4]),
               tau_burst=float(popt[5]), burst_amp=float(popt[6]), asymptote_hz=float(popt[7]),
               acg_r2=float(r2), fit_ok=bool(np.isfinite(r2)), fit=yhat, fit_t=t)
    return out


# ── rhythmicity: theta modulation from the wide ACG ──────────────────────────
def theta_index(acg_norm, bin_ms: float, band=(4.0, 11.0), base_hz: float = 50.0) -> float:
    """Theta-band power of the (mean-subtracted) wide ACG relative to its 0-50 Hz mean."""
    x = np.asarray(acg_norm, dtype=float)
    if x.size < 8:
        return np.nan
    x = x - x.mean()
    freqs = np.fft.rfftfreq(x.size, d=bin_ms / 1000.0)
    P = np.abs(np.fft.rfft(x)) ** 2
    tb = (freqs >= band[0]) & (freqs <= band[1])
    bb = (freqs > 0) & (freqs <= base_hz)
    if not tb.any() or not bb.any() or P[bb].mean() <= 0:
        return np.nan
    return float(P[tb].max() / P[bb].mean())


# ── per-unit ACG feature table ───────────────────────────────────────────────
def acg_feature_table(unit_ids, spike_times, narrow: dict = NARROW, wide: dict = WIDE,
                      fit_max_ms: float = 50.0, verbose: bool = True):
    """Build the per-unit ACG feature frame + the spike-density ACG matrices (for plots).

    Returns (feat_df, narrow_dens, wide_dens, lag_narrow, lag_wide) where *_dens are
    (n_units, n_bins) spike-density ACGs aligned to feat_df rows. Density (not unit-norm) is
    returned so callers can rate-normalize per group (density / firing rate; chance = 1) for
    a firing-rate-invariant mean-ACG comparison. feat_df columns:
        unit_id, n_spikes, tau_rise, tau_decay, acg_r2, refrac_ms, tau_burst, burst_amp,
        asymptote_hz, mean_acg_narrow, mean_acg_wide, theta_index
    (mean_acg_* = mean of the unit-normalized ACG over positive lags — a tonicity proxy:
    flat/tonic ACG -> high; sharply burst-peaked ACG -> low.)
    """
    rows, narrow_dens, wide_dens = [], [], []
    lagN = lagW = None
    n = len(unit_ids)
    for k, (uid, st) in enumerate(zip(unit_ids, spike_times)):
        aN = compute_acg(st, **narrow)
        aW = compute_acg(st, **wide)
        fit = fit_acg(aN["density"], aN["centers"], fit_max_ms=fit_max_ms)
        posN, posW = aN["centers"] > 0, aW["centers"] > 0
        rows.append(dict(
            unit_id=uid, n_spikes=aN["n_spikes"],
            tau_rise=fit["tau_rise"], tau_decay=fit["tau_decay"], acg_r2=fit["acg_r2"],
            refrac_ms=fit["refrac_ms"], tau_burst=fit["tau_burst"], burst_amp=fit["burst_amp"],
            asymptote_hz=fit["asymptote_hz"],
            mean_acg_narrow=float(np.mean(aN["norm"][posN])) if posN.any() else np.nan,
            mean_acg_wide=float(np.mean(aW["norm"][posW])) if posW.any() else np.nan,
            theta_index=theta_index(aW["norm"], wide["bin_ms"])))
        narrow_dens.append(aN["density"])
        wide_dens.append(aW["density"])
        lagN, lagW = aN["centers"], aW["centers"]
        if verbose and (k + 1) % 100 == 0:
            print(f"  ACG {k + 1}/{n}")
    feat_df = pd.DataFrame(rows)
    return feat_df, np.asarray(narrow_dens), np.asarray(wide_dens), lagN, lagW


# ── spectral clustering (the reference's clusterer) ──────────────────────────
def _standardize(F):
    from sklearn.preprocessing import StandardScaler
    return StandardScaler().fit_transform(np.asarray(F, dtype=float))


def spectral_cluster(F, n_clusters: int = 2, affinity: str = "nearest_neighbors",
                     n_neighbors: int = 15, gamma: float = 1.0, seed: int = 42,
                     standardize: bool = True):
    """Spectral clustering on the feature matrix F -> (labels, Z_used).

    n_clusters=2 reproduces the reference's PY/IN split. affinity 'nearest_neighbors'
    (k-NN graph, robust to noise/scale) or 'rbf' (Gaussian kernel, uses gamma). Features
    are z-scored by default (rate/mean-ACG/tau_rise live on very different scales).
    """
    from sklearn.cluster import SpectralClustering
    Z = _standardize(F) if standardize else np.asarray(F, dtype=float)
    kw = dict(n_clusters=n_clusters, affinity=affinity, assign_labels="kmeans", random_state=seed)
    if affinity == "nearest_neighbors":
        kw["n_neighbors"] = n_neighbors
    elif affinity == "rbf":
        kw["gamma"] = gamma
    labels = SpectralClustering(**kw).fit_predict(Z)
    return labels, Z


def spectral_stability(F, n_clusters: int = 2, seeds=(1, 2, 3, 4, 5), **kw) -> float:
    """Mean pairwise ARI of spectral labels across seeds (k-means final step is seed-dependent)."""
    from sklearn.metrics import adjusted_rand_score
    labs = [spectral_cluster(F, n_clusters=n_clusters, seed=s, **kw)[0] for s in seeds]
    aris = [adjusted_rand_score(labs[i], labs[j])
            for i in range(len(labs)) for j in range(i + 1, len(labs))]
    return float(np.mean(aris)) if aris else np.nan


def eigengap_spectrum(F, n_neighbors: int = 15, n_eig: int = 10, standardize: bool = True):
    """Smallest `n_eig` eigenvalues of the symmetric-normalized k-NN graph Laplacian.

    A large gap after the k-th eigenvalue supports k clusters — a sanity check on the
    reference's choice of two groups (plot these and look for the elbow after index 2).
    """
    from sklearn.neighbors import kneighbors_graph
    Z = _standardize(F) if standardize else np.asarray(F, dtype=float)
    A = kneighbors_graph(Z, n_neighbors=n_neighbors, include_self=False).toarray()
    A = np.maximum(A, A.T)
    deg = A.sum(1)
    dinv = np.zeros_like(deg)
    nz = deg > 0
    dinv[nz] = 1.0 / np.sqrt(deg[nz])
    L = np.eye(A.shape[0]) - (dinv[:, None] * A * dinv[None, :])
    w = np.sort(np.linalg.eigvalsh(L))
    return w[:n_eig]


# ── name the two clusters PY / IN by physiology ──────────────────────────────
def _z(x):
    x = np.asarray(x, dtype=float)
    s = np.nanstd(x)
    return (x - np.nanmean(x)) / s if s > 0 else np.zeros_like(x)


def assign_py_in(feat_df, labels, rate_col: str = "mean_rate_hz",
                 rise_col: str = "tau_rise") -> dict:
    """Map the two spectral clusters to PY/IN by profile.

    Interneurons fire faster and rise faster: IN-ness = z(log10 rate) - z(tau_rise).
    The cluster with the higher mean IN-ness is labeled the interneuron cluster. For
    n_clusters != 2 returns generic 'cluster_k' names (nothing to interpret as PY/IN).
    """
    lab = np.asarray(labels)
    uniq = [u for u in np.unique(lab) if u == u]
    rate = feat_df[rate_col].to_numpy(dtype=float)
    rise = feat_df[rise_col].to_numpy(dtype=float)
    inness = _z(np.log10(np.clip(rate, 1e-6, None))) - _z(rise)
    if len(uniq) != 2:
        return {u: f"cluster_{int(u)}" for u in uniq}
    score = {u: float(np.nanmean(inness[lab == u])) for u in uniq}
    in_c = max(score, key=score.get)
    return {u: ("IN (putative interneuron)" if u == in_c else "PY (putative pyramidal)")
            for u in uniq}


def assign_fast_slow_rise(feat_df, rise_col: str = "tau_rise", seed: int = 0):
    """Interpretable 1-D companion split (OURS, not the reference): antimode of log10(tau_rise).

    Fits a 2-component GMM to log10(tau_rise); the split is the density antimode between
    the two means. 'fast-rise' (small tau_rise) ~ putative interneuron; 'slow-rise' ~
    putative pyramidal. Returns (labels array of {'fast-rise','slow-rise'}, split_ms).
    Kept independent so it can be cross-checked against the spectral PY/IN clusters.
    """
    from sklearn.mixture import GaussianMixture
    x = feat_df[rise_col].to_numpy(dtype=float)
    ok = np.isfinite(x) & (x > 0)
    out = np.full(x.shape, None, dtype=object)
    if ok.sum() < 10:
        return out, np.nan
    lx = np.log10(x[ok]).reshape(-1, 1)
    gm = GaussianMixture(n_components=2, random_state=seed).fit(lx)
    order = np.argsort(gm.means_.ravel())
    lo_m, hi_m = gm.means_.ravel()[order]
    grid = np.linspace(lo_m, hi_m, 512).reshape(-1, 1)
    dens = np.exp(gm.score_samples(grid))
    split = float(10 ** grid[np.argmin(dens), 0])
    lab_ok = np.where(x[ok] <= split, "fast-rise", "slow-rise")
    out[ok] = lab_ok
    return out, split


def cluster_agreement(cluster_labels, ref_labels) -> dict:
    """How well an unsupervised clustering matches a reference labeling.

    Returns dict(ari, nmi, best_accuracy, confusion). best_accuracy uses the optimal
    cluster->label assignment (Hungarian), so it is invariant to how the clusters are
    numbered — the right "recovery" score when comparing an ACG cluster to narrow/broad.
    """
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
    from scipy.optimize import linear_sum_assignment
    c = pd.Series(np.asarray(cluster_labels))
    r = pd.Series(np.asarray(ref_labels))
    m = c.notna() & r.notna()
    c, r = c[m], r[m]
    ari = float(adjusted_rand_score(r, c))
    nmi = float(normalized_mutual_info_score(r, c))
    ct = pd.crosstab(c, r)
    row_ind, col_ind = linear_sum_assignment(-ct.values)
    best = float(ct.values[row_ind, col_ind].sum() / ct.values.sum())
    return dict(ari=ari, nmi=nmi, best_accuracy=best, confusion=ct)


def characterize_acg_clusters(feat_df, labels,
                              feats=("mean_rate_hz", "tau_rise", "tau_decay",
                                     "mean_acg_narrow", "burst_index", "theta_index",
                                     "acg_r2", "n_spikes")):
    """Per-cluster median profile + size, for interpreting/naming the spectral groups."""
    lab = np.asarray(labels)
    prof = {}
    for u in [u for u in np.unique(lab) if u == u]:
        m = lab == u
        row = {"n_units": int(m.sum())}
        for f in feats:
            if f in feat_df.columns:
                row[f] = float(np.nanmedian(feat_df[f].to_numpy(dtype=float)[m]))
        prof[u] = row
    return pd.DataFrame(prof).T
