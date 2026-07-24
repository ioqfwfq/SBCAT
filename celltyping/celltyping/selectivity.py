"""stim1 category selectivity + stimulus-aligned PSTHs, to be split by cell type.

Loads ONLY spike times + the trials table (no waveform recompute); cell-type labels come
from the existing outputs/celltype/unit_labels_*.csv. For each unit it tests whether the
stim1 response (aligned to Encoding1 onset) differs across the picture categories, and builds
peri-stimulus firing-rate histograms.

000673 trials columns used:
    timestamps_Encoding1   stim1 onset (s)
    PicIDs_Encoding1       stim1 category id (1..5)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from pynwb import NWBHDF5IO

from .features import _unit_location

ENC1_ONSET = "timestamps_Encoding1"
ENC1_CAT = "PicIDs_Encoding1"


# ── load spikes + trials (no waveform compute) ───────────────────────────────
def load_spikes_trials(nwb_files, areas=None, verbose: bool = True):
    """Per-unit spike times + identity and per-session trials (no waveform features).

    Returns (units_df, spike_times, trials_by_session):
        units_df           : unit_id, session_id, subject, location (area-filtered)
        spike_times        : list of 1-D arrays (s), aligned to units_df rows
        trials_by_session  : {session_id: trials DataFrame or None}
    unit_id = f"{session}_u{idx:03d}" — matches the cell-type label tables, so wf_group can
    be joined straight on.
    """
    area_pats = None
    if areas:
        al = [areas] if isinstance(areas, str) else list(areas)
        area_pats = [str(a).lower() for a in al if str(a).strip()] or None

    rows, spikes, trials_by_session = [], [], {}
    for fp in map(Path, nwb_files):
        sid = fp.stem
        try:
            io = NWBHDF5IO(str(fp), mode="r", load_namespaces=True)
            nwb = io.read()
        except Exception as exc:
            if verbose:
                print(f"  [SKIP] {fp.name}: {exc}")
            continue
        if nwb.units is None:
            io.close()
            continue
        udf = nwb.units.to_dataframe()
        edf = nwb.electrodes.to_dataframe() if nwb.electrodes is not None else pd.DataFrame()
        trials_by_session[sid] = (nwb.trials.to_dataframe() if nwb.trials is not None else None)
        n_kept = 0
        for i in range(len(udf)):
            loc = _unit_location(i, udf, edf)
            if area_pats is not None and not any(p in (loc or "").lower() for p in area_pats):
                continue
            rows.append(dict(unit_id=f"{sid}_u{i:03d}", session_id=sid,
                             subject=sid.split("_")[0], location=loc))
            spikes.append(np.asarray(nwb.units["spike_times"][i][:], dtype=float))
            n_kept += 1
        io.close()
        if verbose:
            nt = 0 if trials_by_session[sid] is None else len(trials_by_session[sid])
            print(f"  {sid}: {n_kept} units, {nt} trials")
    return pd.DataFrame(rows), spikes, trials_by_session


def trial_table(trials_df) -> pd.DataFrame:
    """Per-trial epoch onsets (s) + categories (PicID // 100) for the full-trial PSTH.

    Columns: enc1_on/enc1_cat, enc2_on/enc2_cat, enc3_on/enc3_cat, maint_on, probe_on/probe_cat.
    Missing onsets/categories are NaN (e.g. enc2/enc3 in load-1 trials). Empty if no trials.
    """
    n = 0 if trials_df is None else len(trials_df)

    def num(col):
        if trials_df is None or col not in trials_df:
            return np.full(n, np.nan)
        return pd.to_numeric(trials_df[col], errors="coerce").to_numpy(dtype=float)

    def cat(col):
        v = num(col)
        return np.where(v >= 100, v // 100, v)     # 3-digit PicID -> hundreds-digit category

    return pd.DataFrame(dict(
        loads=num("loads"),
        enc1_on=num("timestamps_Encoding1"), enc1_cat=cat("PicIDs_Encoding1"),
        enc2_on=num("timestamps_Encoding2"), enc2_cat=cat("PicIDs_Encoding2"),
        enc3_on=num("timestamps_Encoding3"), enc3_cat=cat("PicIDs_Encoding3"),
        maint_on=num("timestamps_Maintenance"),
        probe_on=num("timestamps_Probe"), probe_cat=cat("PicIDs_Probe"),
    ))


def epoch_alignment(trial_tab, epoch: str, pref):
    """(onsets, valid_mask, in_mask) for one epoch, where in = preferred category present.

    epoch in {'enc1','enc2','enc3'} : the stimulus at that position is the preferred category.
    epoch == 'delay' (maintenance)  : the preferred category is anywhere in the encoded set.
    epoch == 'probe'                : the probe's category is the preferred category.
    Split trials into preferred-present (valid & in) vs absent (valid & ~in).
    """
    T = trial_tab
    if epoch in ("enc1", "enc2", "enc3"):
        on = T[f"{epoch}_on"].to_numpy(float)
        c = T[f"{epoch}_cat"].to_numpy(float)
        valid = np.isfinite(on) & (on > 0) & np.isfinite(c)
        return on, valid, valid & (c == pref)
    if epoch == "delay":
        on = T["maint_on"].to_numpy(float)
        valid = np.isfinite(on) & (on > 0)
        present = np.zeros(len(T), dtype=bool)
        for col in ("enc1_cat", "enc2_cat", "enc3_cat"):
            cc = T[col].to_numpy(float)
            present |= np.isfinite(cc) & (cc == pref)
        return on, valid, valid & present
    if epoch == "probe":
        on = T["probe_on"].to_numpy(float)
        c = T["probe_cat"].to_numpy(float)
        valid = np.isfinite(on) & (on > 0) & np.isfinite(c)
        return on, valid, valid & (c == pref)
    raise ValueError(f"unknown epoch: {epoch}")


def load_trials(nwb_files, verbose: bool = False) -> dict:
    """{session_id: trials DataFrame} per file (session_id = file stem); None if no trials."""
    out = {}
    for fp in map(Path, nwb_files):
        try:
            io = NWBHDF5IO(str(fp), mode="r", load_namespaces=True)
            nwb = io.read()
            out[fp.stem] = nwb.trials.to_dataframe() if nwb.trials is not None else None
            io.close()
        except Exception as exc:
            if verbose:
                print(f"  [SKIP] {fp.name}: {exc}")
    return out


def trial_onsets_categories(trials_df, keep=(1, 2, 3, 4, 5)):
    """(onsets_s, category_ids) for valid stim1 trials from one session's trials table.

    PicIDs_Encoding1 is a 3-digit picture id whose hundreds digit is the CATEGORY
    (201,202 -> 2 ; 503 -> 5 ; 104 -> 1), so category = id // 100 (matches the repo's
    group_trials_by_first_stim_category). Ids already in 1..5 are used as-is.
    """
    if trials_df is None or ENC1_ONSET not in trials_df or ENC1_CAT not in trials_df:
        return np.array([]), np.array([])
    on = pd.to_numeric(trials_df[ENC1_ONSET], errors="coerce").to_numpy(dtype=float)
    pic = pd.to_numeric(trials_df[ENC1_CAT], errors="coerce").to_numpy(dtype=float)
    cat = np.where(pic >= 100, pic // 100, pic)
    ok = np.isfinite(on) & (on > 0) & np.isfinite(cat) & np.isin(cat, keep)
    return on[ok], cat[ok].astype(int)


def pooled_presentations(trials_df, keep=(1, 2, 3, 4, 5), include_probe: bool = True):
    """All picture presentations in a session -> (onsets_s, category_ids), pooled.

    Stacks every encoding stimulus (Encoding1/2/3) and, if `include_probe`, the Probe, each
    labelled by its OWN picture category (PicID // 100). Absent presentations (e.g. Encoding2/3
    in load-1 trials) and out-of-`keep` categories are dropped. This is the observation set for
    the paper's category-cell selection (200-1000 ms after every stimulus onset).
    """
    cols = [("timestamps_Encoding1", "PicIDs_Encoding1"),
            ("timestamps_Encoding2", "PicIDs_Encoding2"),
            ("timestamps_Encoding3", "PicIDs_Encoding3")]
    if include_probe:
        cols.append(("timestamps_Probe", "PicIDs_Probe"))
    ons, cats = [], []
    for tcol, pcol in cols:
        if trials_df is None or tcol not in trials_df or pcol not in trials_df:
            continue
        on = pd.to_numeric(trials_df[tcol], errors="coerce").to_numpy(dtype=float)
        pic = pd.to_numeric(trials_df[pcol], errors="coerce").to_numpy(dtype=float)
        cat = np.where(pic >= 100, pic // 100, pic)
        ok = np.isfinite(on) & (on > 0) & np.isfinite(cat) & np.isin(cat, keep)
        ons.append(on[ok]); cats.append(cat[ok].astype(int))
    if not ons:
        return np.array([]), np.array([])
    return np.concatenate(ons), np.concatenate(cats)


# ── spike counting / selectivity ─────────────────────────────────────────────
def window_counts(spike_times, onsets, t0: float, t1: float) -> np.ndarray:
    """Spike count in [onset+t0, onset+t1] per trial (searchsorted, O(trials log n))."""
    st = np.sort(np.asarray(spike_times, dtype=float))
    on = np.asarray(onsets, dtype=float)
    lo = np.searchsorted(st, on + t0, side="left")
    hi = np.searchsorted(st, on + t1, side="right")
    return (hi - lo).astype(float)


def _perm_anova(groups, n_perm: int, rng) -> tuple[float, float]:
    """One-way ANOVA permutation p-value (vectorized; matches population/cell_selection...).

    Under label permutation the data values are fixed, so the ANOVA F is monotonic with the
    between-group SS = sum_g(group_sum^2 / n_g); ranking by it gives the identical permutation
    p-value as re-running f_oneway, but all N permutations are computed with one shuffled matrix.
    """
    from scipy.stats import f_oneway
    groups = [np.asarray(g, dtype=float) for g in groups]
    groups = [g[np.isfinite(g)] for g in groups]
    groups = [g for g in groups if g.size > 0]
    if len(groups) < 2:
        return 1.0, np.nan
    f_obs, _ = f_oneway(*groups)
    if not np.isfinite(f_obs):
        return 1.0, float(f_obs)
    sizes = np.array([g.size for g in groups], dtype=float)
    v = np.concatenate(groups)
    bounds = np.concatenate(([0], np.cumsum(sizes[:-1]).astype(int)))     # group start indices
    obs_stat = float((np.array([g.sum() for g in groups]) ** 2 / sizes).sum())
    M = rng.permuted(np.tile(v, (n_perm, 1)), axis=1)                     # N independent shuffles
    gs = np.add.reduceat(M, bounds, axis=1)                              # (N, k) group sums
    perm_stat = (gs ** 2 / sizes).sum(axis=1)
    p = (int(np.count_nonzero(perm_stat >= obs_stat)) + 1) / (n_perm + 1)
    return p, float(f_obs)


def _perm_ttest_gt(g1, g2, n_perm: int, rng) -> float:
    """One-tailed permutation t-test p-value for mean(g1) > mean(g2) (vectorized)."""
    g1 = np.asarray(g1, dtype=float); g1 = g1[np.isfinite(g1)]
    g2 = np.asarray(g2, dtype=float); g2 = g2[np.isfinite(g2)]
    if g1.size == 0 or g2.size == 0:
        return 1.0
    n1, n2 = g1.size, g2.size
    v = np.concatenate([g1, g2]); tot = v.sum()
    obs = g1.mean() - g2.mean()
    M = rng.permuted(np.tile(v, (n_perm, 1)), axis=1)
    s1 = M[:, :n1].sum(axis=1)
    diff = s1 / n1 - (tot - s1) / n2                                     # mean(perm g1) - mean(perm g2)
    return (int(np.count_nonzero(diff >= obs)) + 1) / (n_perm + 1)


def category_selectivity(spike_times, onsets, categories, win=(0.2, 1.0),
                         baseline=(-0.5, 0.0), min_trials: int = 5,
                         n_perm: int = 1000, alpha: float = 0.05, seed: int = 0) -> dict:
    """Permutation category-cell test — the project's concept-cell criteria, by category.

    Follows population/cell_selection_concept_maintenance_probe.ipynb (encoding window
    0.2-1.0 s), but groups stim1 trials by CATEGORY rather than image identity:
      (1) permutation one-way ANOVA of response rates across categories  (p < alpha), AND
      (2) preferred-category rate > all others (one-tailed permutation t-test, p < alpha).
    `selective` requires BOTH. Returns dict(anova_p, pref_p, selective, pref_cat, dos, n_cat,
    rate_by_cat, resp_rate, base_rate, tested). tested=False when <2 categories clear min_trials.
    """
    on = np.asarray(onsets, dtype=float)
    cat = np.asarray(categories)
    out = dict(anova_p=1.0, pref_p=1.0, selective=False, pref_cat=np.nan, dos=np.nan,
               n_cat=0, rate_by_cat={}, resp_rate=np.nan, base_rate=np.nan, tested=False)
    if on.size == 0:
        return out
    dur = win[1] - win[0]
    resp = window_counts(spike_times, on, *win) / dur              # per-trial rate (Hz)
    cats = [c for c in np.unique(cat) if (cat == c).sum() >= min_trials]
    if len(cats) < 2:
        return {**out, "n_cat": len(cats)}

    rng = np.random.default_rng(seed)
    anova_p, _ = _perm_anova([resp[cat == c] for c in cats], n_perm, rng)
    rate_by_cat = {int(c): float(resp[cat == c].mean()) for c in cats}
    means = np.array([rate_by_cat[int(c)] for c in cats], dtype=float)
    n = means.size; rmax = means.max()
    dos = float((n - means.sum() / rmax) / (n - 1)) if (rmax > 0 and n > 1) else np.nan
    pref = int(cats[int(np.argmax(means))])
    base = window_counts(spike_times, on, *baseline) / (baseline[1] - baseline[0])
    out.update(anova_p=float(anova_p), pref_cat=pref, dos=dos, n_cat=n, rate_by_cat=rate_by_cat,
               resp_rate=float(resp.mean()), base_rate=float(base.mean()), tested=True)
    if anova_p < alpha:
        non = resp[np.isin(cat, [c for c in cats if c != pref])]
        out["pref_p"] = float(_perm_ttest_gt(resp[cat == pref], non, n_perm, rng))
    out["selective"] = bool(out["anova_p"] < alpha and out["pref_p"] < alpha)
    return out


def _auc_gt(a, b) -> float:
    """AUC = P(a > b) via Mann-Whitney U / (na*nb); 0.5 = chance, rate-invariant effect size."""
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    if a.size == 0 or b.size == 0:
        return np.nan
    from scipy.stats import mannwhitneyu
    try:
        U = mannwhitneyu(a, b).statistic          # U for `a`; alternative doesn't change U
    except Exception:
        return np.nan
    return float(U / (a.size * b.size))


def category_selectivity_ovr(spike_times, onsets, categories, win=(0.2, 1.0), min_trials: int = 5,
                             n_perm: int = 1000, alpha: float = 0.05, seed: int = 0) -> dict:
    """One-vs-rest max-statistic category selectivity (unbiased; no double-dipping, no FDR needed).

    For each category c: effect(c) = mean_rate(c) - mean_rate(all OTHER categories pooled). The
    observed statistic S = max_c effect(c); the permutation null shuffles category labels and
    recomputes max_c effect(c) each time, so the "best category" selection is built into the null
    (this simultaneously corrects for having tried all categories). selective = perm p < alpha.
    Reports the selective category (argmax effect) and its rate-invariant AUC (best cat vs rest).
    Returns dict(p, selective, pref_cat, best_effect, best_auc, resp_rate, n_cat, tested).
    """
    on = np.asarray(onsets, dtype=float); cat = np.asarray(categories)
    out = dict(p=1.0, selective=False, pref_cat=np.nan, best_effect=np.nan, best_auc=np.nan,
               resp_rate=np.nan, n_cat=0, tested=False)
    if on.size == 0:
        return out
    dur = win[1] - win[0]
    resp_all = window_counts(spike_times, on, *win) / dur              # per-trial rate (Hz)
    cats = [c for c in np.unique(cat) if (cat == c).sum() >= min_trials]
    if len(cats) < 2:
        return {**out, "n_cat": len(cats)}
    m = np.isin(cat, cats)
    r = resp_all[m]; cc = cat[m]
    N = float(r.size); T = float(r.sum())
    sizes = np.array([(cc == c).sum() for c in cats], dtype=float)
    sums0 = np.array([r[cc == c].sum() for c in cats], dtype=float)

    def eff(sums):                                                     # one-vs-rest effect per cat
        return sums / sizes - (T - sums) / (N - sizes)

    obs = eff(sums0)
    S = float(obs.max())
    pref = int(cats[int(np.argmax(obs))])
    # permutation null: shuffle r into the fixed-size category blocks, take the max effect
    bounds = np.concatenate(([0], np.cumsum(sizes[:-1]).astype(int)))
    rng = np.random.default_rng(seed)
    M = rng.permuted(np.tile(r, (n_perm, 1)), axis=1)
    bs = np.add.reduceat(M, bounds, axis=1)                            # (n_perm, n_cats) block sums
    null_max = (bs / sizes - (T - bs) / (N - sizes)).max(axis=1)
    p = (int(np.count_nonzero(null_max >= S)) + 1) / (n_perm + 1)
    best = cc == pref
    out.update(p=float(p), selective=bool(p < alpha), pref_cat=pref, best_effect=S,
               best_auc=_auc_gt(r[best], r[~best]), resp_rate=float(r.mean()),
               n_cat=len(cats), tested=True)
    return out


def bh_fdr(pvals) -> np.ndarray:
    """Benjamini-Hochberg FDR-adjusted q-values; NaN p's pass through as NaN."""
    p = np.asarray(pvals, dtype=float)
    q = np.full(p.shape, np.nan)
    idx = np.where(np.isfinite(p))[0]
    if idx.size == 0:
        return q
    ps = p[idx]
    order = np.argsort(ps)
    n = ps.size
    ranked = ps[order] * n / np.arange(1, n + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    qs = np.empty(n)
    qs[order] = np.minimum(ranked, 1.0)
    q[idx] = qs
    return q


# ── PSTH ─────────────────────────────────────────────────────────────────────
def psth(spike_times, onsets, window=(-0.5, 1.5), bin_ms: float = 50.0, sigma_ms: float = 0.0):
    """Trial-averaged firing rate (Hz) aligned to onsets. Returns (rate, t_centers_s).

    Gaussian-smoothed across bins if sigma_ms > 0.
    """
    st = np.sort(np.asarray(spike_times, dtype=float))
    on = np.asarray(onsets, dtype=float)
    on = on[np.isfinite(on)]
    bin_s = bin_ms / 1000.0
    edges = np.arange(window[0], window[1] + bin_s, bin_s)
    counts = np.zeros(edges.size - 1, dtype=float)
    for o in on:
        lo = np.searchsorted(st, o + window[0], side="left")
        hi = np.searchsorted(st, o + window[1], side="right")
        counts += np.histogram(st[lo:hi] - o, bins=edges)[0]
    rate = counts / max(on.size, 1) / bin_s
    if sigma_ms > 0:
        from scipy.ndimage import gaussian_filter1d
        rate = gaussian_filter1d(rate, sigma_ms / bin_ms)
    return rate, (edges[:-1] + edges[1:]) / 2.0


def psth_by_category(spike_times, onsets, categories, window=(-0.5, 1.5), bin_ms: float = 50.0,
                     sigma_ms: float = 0.0):
    """{category_id: rate} + t_centers, one trial-averaged PSTH per category."""
    on = np.asarray(onsets, dtype=float)
    cat = np.asarray(categories)
    out, t = {}, None
    for c in np.unique(cat):
        r, t = psth(spike_times, on[cat == c], window=window, bin_ms=bin_ms, sigma_ms=sigma_ms)
        out[int(c)] = r
    return out, t
