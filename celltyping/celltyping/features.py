"""Per-unit waveform-shape and spike-train features for cell typing.

One row per unit, joinable across NWB files. Two feature families:

Waveform shape (from the mean spike waveform)
    trough_to_peak_ms   trough -> following-peak duration (narrow/broad axis)
    half_width_ms       full width at half-depth of the main trough
    repol_slope         normalized slope from trough to peak (per ms)
    peak_trough_ratio   |peak| / |trough| amplitude ratio
    amplitude           peak-to-peak of the mean waveform (native units)
    trough_idx/peak_idx sample indices (QC)
    n_spikes_wf         spikes averaged into the mean (QC)
    wf_valid            shape/finiteness guard passed

Spike train / autocorrelogram (from the full-session spike times)
    mean_rate_hz        n_spikes / recording span
    isi_cv              SD/mean of ISIs
    isi_cv2             mean 2|d|/(sum) of adjacent ISIs (rate-drift robust)
    local_variation     Shinomoto Lv
    burst_index         fraction of ISIs < 5 ms
    prop_isi_viol       fraction of ISIs < 2 ms (contamination QC)
    median_isi_ms       median ISI
    acg_refrac_ms       ACG refractory: first lag reaching 50% of the 20-50 ms plateau
    acg_peak_ms         lag of the ACG mode (short for bursting cells)
    acg_burst_ratio     ACG mass 0-8 ms / mass 8-40 ms (bursting > 1)

The Rutishauser/OSort waveform convention (100 kHz, 256 samples, 2.56 ms) is the
default; pass fs_hz to override for a differently-sampled dataset. Physiological
trough_to_peak values (~0.2-1.0 ms) confirm fs_hz is correct.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from pynwb import NWBHDF5IO

OSORT_FS_HZ = 100_000.0          # OSort upsampled waveform rate (Rutishauser lab)
_BASELINE_N = 30                 # leading samples used to estimate waveform baseline


# ── waveform: raw NWB array -> clean 1-D mean waveform ───────────────────────
def mean_waveform(wf) -> tuple[np.ndarray | None, int, int]:
    """Reduce an NWB waveform entry to a 1-D mean waveform on the peak channel.

    Handles the layouts seen across datasets:
      * 1-D (n_samples,)                       already a mean waveform
      * 2-D (rows, n_samples)                  rows = channels (<=8) -> pick peak
                                               channel; else = spikes -> average
      * 3-D (n_ch, n_spikes, n_samples)        average over spikes, pick peak channel

    Returns (mean_wf_1d | None, n_spikes_used, n_channels).
    """
    if wf is None:
        return None, 0, 0
    a = np.asarray(wf, dtype=float)
    if a.size == 0 or not np.isfinite(a).any():
        return None, 0, 0

    if a.ndim == 1:
        return a, 1, 1
    if a.ndim == 2:
        rows, ncols = a.shape
        if rows <= 8:                       # rows are channels
            ch = int(np.nanargmax(np.ptp(a, axis=1)))
            return a[ch], 1, rows
        return np.nanmean(a, axis=0), rows, 1   # rows are spikes
    if a.ndim == 3:                          # (n_ch, n_spikes, n_samples)
        n_ch, n_spk, _ = a.shape
        chan_means = np.nanmean(a, axis=1)   # (n_ch, n_samples)
        ch = int(np.nanargmax(np.ptp(chan_means, axis=1)))
        return chan_means[ch], n_spk, n_ch
    # unexpected: flatten trailing dims
    a2 = a.reshape(-1, a.shape[-1])
    return np.nanmean(a2, axis=0), a2.shape[0], 1


# ── polarity normalization (shared by shape features + WaveMAP collection) ───
def polarity_normalized_waveform(mean_wf):
    """Baseline-subtracted mean waveform with the main deflection as a downward trough.

    ~90% of these human units are recorded as positive ("m-shaped") spikes; flipping
    them so every waveform is trough-down is required both for comparable widths and
    so WaveMAP does not cluster on polarity artifact. Returns (w, inverted) or (None, False).
    """
    if mean_wf is None:
        return None, False
    w = np.asarray(mean_wf, dtype=float).ravel()
    if w.size < 8 or not np.all(np.isfinite(w)):
        return None, False
    w = w - np.median(w[:_BASELINE_N])
    inverted = abs(np.nanmax(w)) > abs(np.nanmin(w))
    if inverted:
        w = -w
    return w - np.median(w[:_BASELINE_N]), bool(inverted)


# ── waveform-shape features ──────────────────────────────────────────────────
def waveform_shape_features(mean_wf, fs_hz: float = OSORT_FS_HZ) -> dict:
    """Width/shape metrics for one mean waveform (polarity-normalized to a trough)."""
    nan = dict(trough_to_peak_ms=np.nan, half_width_ms=np.nan, repol_slope=np.nan,
               peak_trough_ratio=np.nan, amplitude=np.nan, trough_idx=-1, peak_idx=-1,
               inverted=False, wf_valid=False)
    w, inverted = polarity_normalized_waveform(mean_wf)
    if w is None:
        return nan
    dt_ms = 1000.0 / fs_hz

    trough_idx = int(np.argmin(w))
    if trough_idx >= w.size - 1:            # trough at the very end -> no peak to measure
        return {**nan, "inverted": inverted}
    peak_rel = int(np.argmax(w[trough_idx:]))
    peak_idx = trough_idx + peak_rel

    trough_to_peak_ms = peak_rel * dt_ms

    # FWHM of the main trough (half of trough depth).
    half = w[trough_idx] / 2.0
    left = trough_idx
    while left > 0 and w[left] <= half:
        left -= 1
    right = trough_idx
    while right < w.size - 1 and w[right] <= half:
        right += 1
    half_width_ms = (right - left) * dt_ms

    trough_amp = float(-w[trough_idx])                 # positive depth
    peak_amp = float(w[peak_idx])
    amplitude = float(np.ptp(w))
    repol_slope = ((peak_amp - w[trough_idx]) / max(trough_to_peak_ms, dt_ms)) / max(amplitude, 1e-12)
    peak_trough_ratio = peak_amp / max(trough_amp, 1e-12)

    return dict(trough_to_peak_ms=float(trough_to_peak_ms), half_width_ms=float(half_width_ms),
                repol_slope=float(repol_slope), peak_trough_ratio=float(peak_trough_ratio),
                amplitude=amplitude, trough_idx=trough_idx, peak_idx=peak_idx,
                inverted=bool(inverted), wf_valid=True)


# ── autocorrelogram ──────────────────────────────────────────────────────────
def autocorrelogram(spike_times_s, bin_ms: float = 1.0, win_ms: float = 50.0) -> tuple[np.ndarray, np.ndarray]:
    """Symmetric ACG (counts) over +/-win_ms, excluding the zero-lag self-count.

    Returns (counts, bin_centers_ms). O(n * neighbors) via searchsorted.
    """
    t = np.sort(np.asarray(spike_times_s, dtype=float)) * 1000.0   # ms
    n = t.size
    edges = np.arange(-win_ms, win_ms + bin_ms, bin_ms)
    counts = np.zeros(edges.size - 1, dtype=np.int64)
    if n < 2:
        return counts, (edges[:-1] + edges[1:]) / 2.0
    lo = np.searchsorted(t, t - win_ms, side="left")
    hi = np.searchsorted(t, t + win_ms, side="right")
    for i in range(n):
        j0, j1 = lo[i], hi[i]
        if j1 - j0 <= 1:
            continue
        d = t[j0:j1] - t[i]
        d = d[d != 0.0]
        counts += np.histogram(d, bins=edges)[0]
    return counts, (edges[:-1] + edges[1:]) / 2.0


# ── spike-train / ACG features ───────────────────────────────────────────────
def spike_train_features(spike_times_s, bin_ms: float = 1.0, win_ms: float = 50.0) -> dict:
    """Firing-statistics + ACG-shape features from one unit's full spike train."""
    nan = dict(mean_rate_hz=np.nan, isi_cv=np.nan, isi_cv2=np.nan, local_variation=np.nan,
               burst_index=np.nan, prop_isi_viol=np.nan, median_isi_ms=np.nan,
               acg_refrac_ms=np.nan, acg_peak_ms=np.nan, acg_burst_ratio=np.nan,
               n_spikes=0, st_valid=False)
    t = np.sort(np.asarray(spike_times_s, dtype=float))
    t = t[np.isfinite(t)]
    n = t.size
    if n < 3:
        return {**nan, "n_spikes": int(n)}
    span = t[-1] - t[0]
    mean_rate = n / span if span > 0 else np.nan

    isi = np.diff(t) * 1000.0                       # ms
    isi = isi[isi > 0]
    if isi.size < 2:
        return {**nan, "n_spikes": int(n), "mean_rate_hz": mean_rate}
    isi_cv = float(np.std(isi) / np.mean(isi))
    a, b = isi[:-1], isi[1:]
    isi_cv2 = float(np.mean(2.0 * np.abs(b - a) / (a + b)))
    local_variation = float(np.mean(3.0 * ((a - b) / (a + b)) ** 2))
    burst_index = float(np.mean(isi < 5.0))
    prop_isi_viol = float(np.mean(isi < 2.0))
    median_isi_ms = float(np.median(isi))

    counts, centers = autocorrelogram(t, bin_ms=bin_ms, win_ms=win_ms)
    pos = centers > 0
    c_pos, lag_pos = counts[pos].astype(float), centers[pos]
    acg_refrac_ms = acg_peak_ms = acg_burst_ratio = np.nan
    if c_pos.sum() > 0:
        plateau = c_pos[(lag_pos >= 20) & (lag_pos <= 50)].mean() if (lag_pos <= 50).any() else c_pos.mean()
        if plateau > 0:
            reach = np.where(c_pos >= 0.5 * plateau)[0]
            acg_refrac_ms = float(lag_pos[reach[0]]) if reach.size else float(lag_pos[-1])
        acg_peak_ms = float(lag_pos[int(np.argmax(c_pos))])
        near = c_pos[lag_pos <= 8].sum()
        far = c_pos[(lag_pos > 8) & (lag_pos <= 40)].sum()
        acg_burst_ratio = float(near / far) if far > 0 else np.nan

    return dict(mean_rate_hz=float(mean_rate), isi_cv=isi_cv, isi_cv2=isi_cv2,
                local_variation=local_variation, burst_index=burst_index,
                prop_isi_viol=prop_isi_viol, median_isi_ms=median_isi_ms,
                acg_refrac_ms=acg_refrac_ms, acg_peak_ms=acg_peak_ms,
                acg_burst_ratio=acg_burst_ratio, n_spikes=int(n), st_valid=True)


# ── NWB unit iteration -> per-unit feature table ─────────────────────────────
def _unit_location(unit_idx, units_df, electrodes_df):
    if "electrodes" not in units_df.columns or electrodes_df is None or electrodes_df.empty:
        return None
    try:
        info = units_df.iloc[unit_idx]["electrodes"]
        if isinstance(info, pd.DataFrame):
            eidx = int(info.index[0]) if len(info.index) else None
        elif hasattr(info, "index") and len(info.index):
            eidx = int(info.index[0])
        elif pd.notna(info):
            eidx = int(info)
        else:
            eidx = None
        if eidx is None or eidx not in electrodes_df.index:
            return None
        return str(electrodes_df.loc[eidx, "location"])
    except Exception:
        return None


def build_unit_features(nwb_files, fs_hz: float = OSORT_FS_HZ,
                        wf_col_priority=("waveform_mean", "waveforms"),
                        return_waveforms: bool = False,
                        verbose: bool = True):
    """One row per unit across all NWB files: identity + waveform + spike features.

    unit_id = f"{file.stem}_u{unit_idx:03d}" (matches the WM-binding loader scheme).
    Also carries any per-unit QC columns present (waveforms_mean_snr, isolation_distance)
    for downstream WaveMAP quality gating.

    If return_waveforms=True, returns (df, waveforms) where `waveforms` is a list of
    polarity-normalized 1-D mean waveforms (native length) in the same row order as df
    (None for units without a valid waveform) — feed these straight into WaveMAP.
    """
    rows = []
    waveforms_out = []
    for fp in nwb_files:
        fp = Path(fp)
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
        wf_col = next((c for c in wf_col_priority if c in udf.columns), None)
        qc_cols = [c for c in ("waveforms_mean_snr", "waveforms_peak_snr",
                               "waveforms_isolation_distance", "clusterID_orig") if c in udf.columns]
        n_valid = 0
        for i in range(len(udf)):
            loc = _unit_location(i, udf, edf)
            m_wf, n_spk_wf, n_ch = mean_waveform(udf.iloc[i][wf_col]) if wf_col else (None, 0, 0)
            wf_feat = waveform_shape_features(m_wf, fs_hz=fs_hz)
            st = np.asarray(nwb.units["spike_times"][i][:], dtype=float)
            st_feat = spike_train_features(st)
            n_valid += int(wf_feat["wf_valid"])
            row = dict(unit_id=f"{sid}_u{i:03d}", session_id=sid, subject=sid.split("_")[0],
                       location=loc, n_spikes_wf=n_spk_wf, n_channels=n_ch,
                       **wf_feat, **st_feat)
            for c in qc_cols:
                row[c] = udf.iloc[i][c]
            rows.append(row)
            if return_waveforms:
                norm_wf, _ = polarity_normalized_waveform(m_wf)
                waveforms_out.append(norm_wf)
        io.close()
        if verbose:
            print(f"  {sid}: {len(udf)} units ({n_valid} valid waveforms), wf_col={wf_col}")
    df = pd.DataFrame(rows)
    if return_waveforms:
        return df, waveforms_out
    return df
