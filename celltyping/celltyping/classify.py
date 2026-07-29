"""Narrow/broad waveform split + putative-interneuron verification.

split: trough-to-peak duration, boundary from the GMM density antimode (the same
rule as the WM-binding neural_01c pipeline), with 'kde'/'median'/'fixed' fallbacks.

'antimode' fits a 2-component GMM, which fails when the narrow cluster is a small
minority (<~10%) and the broad mass is wide: the mixture then splits the BROAD
population in two, finds no valley between its component means, and silently falls
back to their midpoint -- a boundary that cuts through the middle of the main mode.
'kde' is the robust alternative: a kernel density estimate searched for its deepest
local minimum inside a physiologically plausible window. Use kde_antimode_split()
directly to get the diagnostics (valley depth, fallback flag) needed to tell a real
valley from a fabricated one.

verify: the physiology check the project calls for — narrow-spiking cells, if they
are putative fast-spiking inhibitory interneurons, should (vs broad) fire faster,
have a shorter ACG refractory, burst less, and fire more regularly. Reported as
per-group medians + Mann-Whitney tests on the extractor's ISI/ACG features.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def kde_antimode_split(x, lo: float = 0.25, hi: float = 0.75, bw: float = 0.15) -> dict:
    """Deepest KDE local minimum of `x` inside [lo, hi] -- the narrow/broad valley.

    The search window brackets where the FS/pyramidal boundary can physiologically sit,
    so a small narrow cluster cannot be outvoted by structure inside the broad mass.

    Returns dict(split_ms, rel_depth, found). `rel_depth` is the valley density over the
    window's peak density: ~0.2-0.3 is a well-separated valley, >0.5 is barely a dip and
    means the split is weakly supported. found=False (split_ms = midpoint of the window)
    when the density is monotonic inside the window -- treat that as "no split found",
    not as a boundary.
    """
    from scipy.stats import gaussian_kde
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 20:
        return dict(split_ms=float(0.5 * (lo + hi)), rel_depth=np.nan, found=False)
    grid = np.linspace(lo, hi, 2001)
    d = gaussian_kde(x, bw_method=bw)(grid)
    interior = np.where((d[1:-1] < d[:-2]) & (d[1:-1] < d[2:]))[0] + 1
    if interior.size == 0:
        return dict(split_ms=float(0.5 * (lo + hi)), rel_depth=np.nan, found=False)
    j = int(interior[np.argmin(d[interior])])
    return dict(split_ms=float(grid[j]), rel_depth=float(d[j] / d.max()), found=True)


def assign_narrow_broad(feat_df: pd.DataFrame, col: str = "trough_to_peak_ms",
                        method: str = "antimode", fixed_ms: float = 0.5,
                        valid_col: str = "wf_valid", random_state: int = 0,
                        kde_window: tuple = (0.25, 0.75), kde_bw: float = 0.15):
    """Return (group Series aligned to feat_df.index, split_ms).

    narrow = col < split_ms, broad = col >= split_ms; pd.NA where invalid.

    method: 'antimode' (2-component GMM valley) | 'kde' (KDE valley in kde_window,
    robust to a small narrow minority) | 'median' | 'fixed'.
    """
    valid = feat_df.get(valid_col, pd.Series(True, index=feat_df.index)).fillna(False).astype(bool)
    x_all = pd.to_numeric(feat_df[col], errors="coerce")
    valid &= x_all.notna()
    x = x_all[valid].to_numpy(float)
    if x.size == 0:
        raise ValueError(f"no valid '{col}' values to split on")

    if method == "fixed":
        split_ms = float(fixed_ms)
    elif method == "median":
        split_ms = float(np.median(x))
    elif method == "kde":
        res = kde_antimode_split(x, lo=kde_window[0], hi=kde_window[1], bw=kde_bw)
        split_ms = res["split_ms"] if res["found"] else float(fixed_ms)
    elif method == "antimode":
        from sklearn.mixture import GaussianMixture
        gm = GaussianMixture(2, random_state=random_state).fit(x.reshape(-1, 1))
        m_lo, m_hi = np.sort(gm.means_.ravel())
        if not (np.isfinite(m_lo) and np.isfinite(m_hi)) or m_hi <= m_lo:
            split_ms = float(np.median(x))
        else:
            grid = np.linspace(m_lo, m_hi, 1001)
            pdf = np.exp(gm.score_samples(grid.reshape(-1, 1)))
            j = int(np.argmin(pdf))
            split_ms = float(0.5 * (m_lo + m_hi)) if j in (0, grid.size - 1) else float(grid[j])
    else:
        raise ValueError(f"unknown method {method!r}")

    group = pd.Series(pd.NA, index=feat_df.index, dtype="object")
    group[valid & (x_all < split_ms)] = "narrow"
    group[valid & (x_all >= split_ms)] = "broad"
    return group, split_ms


# Features on which narrow (putative FS interneuron) should differ from broad, and
# the direction that supports the interneuron interpretation.
_VERIFY = {
    "mean_rate_hz":   ("higher in narrow", "greater"),   # interneurons fire faster
    "acg_refrac_ms":  ("shorter in narrow", "less"),     # shorter refractory
    "burst_index":    ("lower in narrow",  "less"),      # FS cells are non-bursting
    "acg_burst_ratio":("lower in narrow",  "less"),
    "isi_cv":         ("lower in narrow",  "less"),       # more regular firing
    "half_width_ms":  ("narrower in narrow", "less"),
}


def interneuron_verification(feat_df: pd.DataFrame, group_col: str = "wf_group") -> pd.DataFrame:
    """Per-feature narrow-vs-broad medians + Mann-Whitney U, with an expectation flag."""
    from scipy.stats import mannwhitneyu
    nar = feat_df[feat_df[group_col] == "narrow"]
    brd = feat_df[feat_df[group_col] == "broad"]
    rows = []
    for feat, (expect, alt) in _VERIFY.items():
        if feat not in feat_df.columns:
            continue
        a = pd.to_numeric(nar[feat], errors="coerce").dropna()
        b = pd.to_numeric(brd[feat], errors="coerce").dropna()
        if a.size < 3 or b.size < 3:
            continue
        # one-sided test in the predicted direction
        try:
            _, p = mannwhitneyu(a, b, alternative=alt)
        except ValueError:
            p = np.nan
        rows.append(dict(feature=feat, expectation=expect,
                         narrow_med=float(a.median()), broad_med=float(b.median()),
                         n_narrow=int(a.size), n_broad=int(b.size),
                         p_onesided=float(p),
                         supports=bool(np.isfinite(p) and p < 0.05)))
    return pd.DataFrame(rows)
