"""Cross-frequency coupling helpers for SFC and ir-PAC analyses."""

from __future__ import annotations

import json
import pickle
import warnings
from pathlib import Path
from typing import Dict, Mapping, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pynwb import NWBHDF5IO
from pynwb.core import DynamicTable
from scipy.signal import butter, filtfilt, hilbert
from scipy.stats import norm, wilcoxon
from statsmodels.stats.multitest import fdrcorrection

from .data_loading import load_lfp_safe, load_nwb_file


def _to_numpy(data: Sequence[float] | np.ndarray | None) -> np.ndarray:
    """Convert arbitrary sequences (or None) to a 1D float array."""
    if data is None:
        return np.empty(0, dtype=float)
    if isinstance(data, np.ndarray):
        arr = data
    else:
        arr = np.asarray(list(data), dtype=float)
    return arr.astype(float, copy=False)


def _get_rng(seed=None):
    """Return a numpy Generator from None/int/Generator inputs."""
    if seed is None:
        return np.random.default_rng()
    if isinstance(seed, np.random.Generator):
        return seed
    return np.random.default_rng(seed)


def _infer_area(location: str | None) -> str:
    """Infer gross anatomical area label from electrode location string."""
    if location is None or (isinstance(location, float) and np.isnan(location)):
        return "unknown"
    loc = str(location).lower()
    if "hipp" in loc:
        return "hipp"
    if ("pfc" in loc and ("vm" in loc or "ventral" in loc)) or ("prefrontal" in loc and "ventral" in loc):
        return "vmPFC"
    if "acc" in loc:
        return "dACC"
    if "amyg" in loc:
        return "amyg"
    if "sma" in loc:
        return "preSMA"
    return loc.replace("_", " ")


def _infer_hemisphere(location: str | None) -> str:
    """Infer hemisphere from electrode or unit location text."""
    if location is None or (isinstance(location, float) and np.isnan(location)):
        return "unknown"
    loc = str(location).lower()
    if "left" in loc or loc.endswith("_l"):
        return "L"
    if "right" in loc or loc.endswith("_r"):
        return "R"
    if loc.endswith("l") and not loc.endswith("al"):
        return "L"
    if loc.endswith("r"):
        return "R"
    return "unknown"


def _as_dataframe(table: Union[pd.DataFrame, Sequence[dict]]) -> pd.DataFrame:
    if isinstance(table, pd.DataFrame):
        return table.copy()
    return pd.DataFrame(list(table))


def _validate_cache_payload(payload: dict) -> bool:
    required = {
        "trial_table",
        "spike_times_by_unit",
        "unit_meta",
        "lfp_by_channel",
        "lfp_fs",
        "channel_meta",
        "session_meta",
        "lfp_start_time",
    }
    return required.issubset(payload.keys())


def _save_session_cache(cache_path: Path, payload: dict) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("wb") as f:
        pickle.dump(payload, f)


def _load_session_cache(cache_path: Path) -> dict | None:
    if not cache_path.exists():
        return None
    with cache_path.open("rb") as f:
        payload = pickle.load(f)
    if not isinstance(payload, dict) or not _validate_cache_payload(payload):
        raise ValueError(f"Invalid cache structure at {cache_path}")
    return payload


def _pad_signal(signal: np.ndarray, pad_samples: int) -> tuple[np.ndarray, slice]:
    """Reflect-pad a 1D signal and return padded copy plus slicing indices."""
    if pad_samples <= 0:
        return signal, slice(None)
    padded = np.pad(signal, pad_samples, mode="reflect")
    valid_slice = slice(pad_samples, -pad_samples)
    return padded, valid_slice


def _bca_ci(
    data: np.ndarray,
    statistic,
    alpha: float = 0.05,
    n_boot: int = 2000,
    rng_seed: int | None = 42,
) -> Tuple[float, float]:
    """Bias-corrected accelerated CI for a univariate statistic."""
    data = np.asarray(data)
    data = data[~np.isnan(data)]
    if data.size == 0:
        return (np.nan, np.nan)

    obs = statistic(data)
    rng = _get_rng(rng_seed)
    boot_stats = []
    for _ in range(n_boot):
        sample = rng.choice(data, size=data.size, replace=True)
        boot_stats.append(statistic(sample))
    boot_stats = np.array(boot_stats)

    # Bias correction
    z0 = norm.ppf((boot_stats < obs).mean() or 1e-9)

    # Acceleration via jackknife
    jackknife_stats = []
    for idx in range(data.size):
        jackknife_sample = np.delete(data, idx)
        jackknife_stats.append(statistic(jackknife_sample))
    jackknife_stats = np.array(jackknife_stats)
    jackknife_mean = jackknife_stats.mean()
    numer = np.sum((jackknife_mean - jackknife_stats) ** 3)
    denom = 6.0 * (np.sum((jackknife_mean - jackknife_stats) ** 2) ** 1.5)
    accel = numer / denom if denom != 0 else 0.0

    def percentile(p):
        z = norm.ppf(p)
        adj = norm.cdf(z0 + (z0 + z) / (1 - accel * (z0 + z)))
        adj = np.clip(adj, 0, 1)
        return np.percentile(boot_stats, adj * 100)

    lower = percentile(alpha / 2)
    upper = percentile(1 - alpha / 2)
    return float(lower), float(upper)


def _build_trial_table(raw_trials: pd.DataFrame, *, conditions: Sequence[str], use_only_correct: bool) -> pd.DataFrame:
    """Return standardized trial metadata table."""
    trials = raw_trials.copy()
    trials = trials.reset_index().rename(columns={"index": "trial_id"})

    # Condition label derived from load or fallback to string representation.
    if "loads" in trials.columns:
        trials["condition_label"] = trials["loads"].apply(lambda x: f"L{int(x)}" if not pd.isna(x) else "Unknown")
    else:
        trials["condition_label"] = "Unknown"

    # Reaction time using probe -> response timestamps if available.
    if {"timestamps_Response", "timestamps_Probe"}.issubset(trials.columns):
        trials["rt"] = trials["timestamps_Response"] - trials["timestamps_Probe"]
    else:
        trials["rt"] = trials.get("stop_time", np.nan) - trials.get("start_time", np.nan)

    # Maintenance onset
    trials["maint_onset_time"] = trials.get("timestamps_Maintenance", np.nan)

    # Accuracy flag
    if "response_accuracy" in trials.columns:
        trials["is_correct"] = trials["response_accuracy"].astype(int) == 1
    else:
        trials["is_correct"] = True

    if use_only_correct:
        trials = trials[trials["is_correct"]]

    # Keep only requested conditions
    if conditions:
        trials = trials[trials["condition_label"].isin(conditions)]

    cols = ["trial_id", "condition_label", "rt", "maint_onset_time", "is_correct"]
    keep_cols = [col for col in cols if col in trials.columns]
    trials = trials[keep_cols + [c for c in trials.columns if c not in keep_cols]]
    return trials.reset_index(drop=True)


def _extract_spike_structures(nwbfile) -> tuple[dict, pd.DataFrame]:
    """Extract spike times and associated metadata."""
    spike_times_by_unit = {}
    unit_rows = []
    if nwbfile.units is None:
        return spike_times_by_unit, pd.DataFrame(columns=["unit_id", "channel_id", "area", "hemisphere"])

    electrodes_df = nwbfile.electrodes.to_dataframe() if nwbfile.electrodes is not None else pd.DataFrame()
    unit_ids = nwbfile.units.id[:]

    for idx, unit_id in enumerate(unit_ids):
        spikes = np.array(nwbfile.units["spike_times"][idx])
        spike_times_by_unit[int(unit_id)] = spikes

        electrode_region = None
        electrode_idx = None
        if "electrodes" in nwbfile.units.colnames:
            elec_ref = nwbfile.units["electrodes"][idx]
            if isinstance(elec_ref, (list, np.ndarray)) and len(elec_ref) > 0:
                electrode_idx = int(elec_ref[0])
            elif np.isscalar(elec_ref):
                electrode_idx = int(elec_ref)
        if electrode_idx is not None and not electrodes_df.empty and electrode_idx in electrodes_df.index:
            electrode_region = electrodes_df.loc[electrode_idx, "location"]

        area = _infer_area(electrode_region)
        hemisphere = _infer_hemisphere(electrode_region)
        unit_rows.append(
            {
                "unit_id": int(unit_id),
                "channel_id": electrode_idx,
                "area": area,
                "hemisphere": hemisphere,
            }
        )

    unit_meta = pd.DataFrame(unit_rows)
    return spike_times_by_unit, unit_meta


def _extract_lfp_structures(nwb_data: dict) -> tuple[dict, pd.DataFrame, float, float]:
    """Return {chan_id: signal}, channel metadata, sampling rate, and start time."""
    if "lfp" not in nwb_data:
        raise ValueError("NWB session missing LFP acquisition (expected 'LFPs').")

    lfp_series = nwb_data["lfp"]["series"]
    electrodes_df = nwb_data["nwbfile"].electrodes.to_dataframe()
    lfp_data = load_lfp_safe(lfp_series)
    if lfp_data.ndim == 1:
        lfp_data = lfp_data[:, np.newaxis]

    if hasattr(lfp_series, "electrodes") and lfp_series.electrodes is not None:
        electrode_indices = np.array(lfp_series.electrodes.data[:]).astype(int)
    else:
        electrode_indices = np.arange(lfp_data.shape[1])

    lfp_by_channel = {}
    channel_rows = []
    for col_idx, electrode_idx in enumerate(electrode_indices):
        signal = lfp_data[:, col_idx].astype(np.float32, copy=False)
        lfp_by_channel[int(electrode_idx)] = signal
        row = electrodes_df.loc[electrode_idx] if electrode_idx in electrodes_df.index else {}
        location = row.get("location") if isinstance(row, pd.Series) else None
        channel_rows.append(
            {
                "chan_id": int(electrode_idx),
                "lfp_col": int(col_idx),
                "area": _infer_area(location),
                "hemisphere": _infer_hemisphere(location),
                "bad": bool(row.get("bad", False)) if isinstance(row, pd.Series) and "bad" in row else False,
                "location": location,
            }
        )

    channel_meta = pd.DataFrame(channel_rows)
    lfp_fs = float(getattr(lfp_series, "rate", nwb_data["lfp"].get("sampling_rate", np.nan)))
    lfp_start_time = float(getattr(lfp_series, "starting_time", 0.0))
    return lfp_by_channel, channel_meta, lfp_fs, lfp_start_time


def _session_metadata(nwbfile, session_id: str | None) -> dict:
    subject = nwbfile.subject.subject_id if nwbfile.subject else None
    session_start = str(nwbfile.session_start_time) if hasattr(nwbfile, "session_start_time") else None
    reference = None
    if nwbfile.electrodes is not None and "filtering" in nwbfile.electrodes.colnames:
        ref_vals = nwbfile.electrodes["filtering"][:]
        if len(ref_vals):
            reference = ref_vals[0]
    return {
        "session_id": session_id or nwbfile.identifier,
        "subject_id": subject,
        "session_start_time": session_start,
        "reference_scheme": reference,
        "session_description": getattr(nwbfile, "session_description", ""),
    }


def prepare_session_structures(
    *,
    session_id: str,
    session_path: str | Path | None = None,
    cache_path: str | Path | None = None,
    use_cache: bool = True,
    save_cache: bool = False,
    conditions: Sequence[str] = ("L1", "L3"),
    use_only_correct: bool = True,
) -> dict:
    """
    Adapter that returns canonical data objects expected by downstream analyses.

    Returns:
        dict with keys:
            trial_table (DataFrame), spike_times_by_unit (dict), unit_meta (DataFrame),
            lfp_by_channel (dict), lfp_fs (float), lfp_start_time (float),
            channel_meta (DataFrame), session_meta (dict)
    """

    cache_path = Path(cache_path) if cache_path else None
    if use_cache and cache_path:
        payload = _load_session_cache(cache_path)
        if payload is not None:
            return payload

    if session_path is None:
        raise ValueError("session_path is required when cache is unavailable.")

    session_path = Path(session_path)
    nwb_data = load_nwb_file(session_path)
    nwbfile = nwb_data["nwbfile"]

    raw_trials = nwb_data.get("trials")
    if raw_trials is None:
        raise ValueError("Trial table not found in NWB session.")
    trial_table = _build_trial_table(raw_trials, conditions=conditions, use_only_correct=use_only_correct)

    spike_times_by_unit, unit_meta = _extract_spike_structures(nwbfile)
    lfp_by_channel, channel_meta, lfp_fs, lfp_start_time = _extract_lfp_structures(nwb_data)
    session_meta = _session_metadata(nwbfile, session_id=session_id)

    payload = {
        "trial_table": trial_table,
        "spike_times_by_unit": spike_times_by_unit,
        "unit_meta": unit_meta,
        "lfp_by_channel": lfp_by_channel,
        "lfp_fs": lfp_fs,
        "lfp_start_time": lfp_start_time,
        "channel_meta": channel_meta,
        "session_meta": session_meta,
    }

    if save_cache and cache_path:
        _save_session_cache(cache_path, payload)

    nwb_data["io"].close()
    return payload


def bandpass_filtfilt(x, fs, band, order=4):
    """Zero-phase band-pass filter."""
    x = _to_numpy(x)
    nyq = fs / 2.0
    low, high = band
    if low <= 0 or high >= nyq:
        raise ValueError("Band edges must satisfy 0 < low < high < fs/2.")
    b, a = butter(order, [low / nyq, high / nyq], btype="band")
    return filtfilt(b, a, x)


def theta_phase(lfp, fs, theta_band=(3, 7)):
    """Return instantaneous phase for theta-filtered LFP."""
    x = bandpass_filtfilt(lfp, fs, theta_band)
    return np.angle(hilbert(x))


def gamma_amplitude(lfp, fs, gamma_band=(70, 140)):
    """Return gamma-band amplitude envelope."""
    x = bandpass_filtfilt(lfp, fs, gamma_band)
    return np.abs(hilbert(x))


def sample_phase_at_spikes(phase_ts, fs, t0, spike_times):
    """Sample theta phase time-series at spike timestamps (with interpolation)."""
    phase_ts = _to_numpy(phase_ts)
    spikes = _to_numpy(spike_times)
    if phase_ts.size == 0 or spikes.size == 0:
        return np.empty(0, dtype=float)

    window_end = t0 + (phase_ts.size - 1) / fs
    valid_mask = (spikes >= t0) & (spikes <= window_end)
    if not np.any(valid_mask):
        return np.empty(0, dtype=float)

    spikes = spikes[valid_mask]
    time_axis = t0 + np.arange(phase_ts.size) / fs
    unwrapped = np.unwrap(phase_ts)
    interp_vals = np.interp(spikes, time_axis, unwrapped, left=np.nan, right=np.nan)
    interp_vals = interp_vals[~np.isnan(interp_vals)]
    if interp_vals.size == 0:
        return np.empty(0, dtype=float)
    wrapped = (interp_vals + np.pi) % (2 * np.pi) - np.pi
    return wrapped


def mvl(phases):
    """Mean vector length for circular data."""
    phases = _to_numpy(phases)
    if phases.size == 0:
        return np.nan
    return np.abs(np.mean(np.exp(1j * phases)))


def sfc_zscore(phases, jitter_phases_list):
    """Z-score the observed MVL against jitter-based surrogates."""
    obs = mvl(phases)
    if np.isnan(obs):
        return np.nan
    if not jitter_phases_list:
        return np.nan
    surrogate_vals = np.array([mvl(p) for p in jitter_phases_list], dtype=float)
    surrogate_vals = surrogate_vals[~np.isnan(surrogate_vals)]
    if surrogate_vals.size == 0:
        return np.nan
    mu = surrogate_vals.mean()
    sigma = surrogate_vals.std(ddof=0)
    if sigma == 0:
        return np.nan
    return (obs - mu) / sigma


def jitter_spikes_within_trial(spike_times, win_start, win_end, jitter_width=0.25, rng=None):
    """Uniformly jitter spikes within a trial while respecting window bounds."""
    spikes = _to_numpy(spike_times)
    if spikes.size == 0:
        return np.empty(0, dtype=float)
    rng = _get_rng(rng)
    jittered = np.empty_like(spikes)
    for idx, spike in enumerate(spikes):
        for _ in range(100):
            candidate = spike + rng.uniform(-jitter_width, jitter_width)
            if win_start <= candidate <= win_end:
                jittered[idx] = candidate
                break
        else:
            jittered[idx] = np.clip(spike, win_start, win_end)
    return jittered


def equalize_spike_counts(phase_lists_by_cond, repeats=200, rng=None):
    """Downsample spike-phase vectors to the minimum count across conditions."""
    if not phase_lists_by_cond:
        return {}, 0
    counts = {cond: len(_to_numpy(phases)) for cond, phases in phase_lists_by_cond.items()}
    if any(count == 0 for count in counts.values()):
        return {cond: np.nan for cond in counts}, 0
    min_count = min(counts.values())
    rng = _get_rng(rng)
    mvls_by_cond = {cond: [] for cond in phase_lists_by_cond}
    for _ in range(repeats):
        for cond, phases in phase_lists_by_cond.items():
            phases = _to_numpy(phases)
            idx = rng.choice(phases.size, size=min_count, replace=False)
            sample = phases[idx]
            mvls_by_cond[cond].append(mvl(sample))
    averaged = {
        cond: float(np.nanmean(vals)) if len(vals) else np.nan
        for cond, vals in mvls_by_cond.items()
    }
    return averaged, min_count


def equalize_trial_counts(values_by_cond, repeats=200, rng=None):
    """Downsample trial-level metrics to the minimum trial count across conditions."""
    if not values_by_cond:
        return {}, 0
    counts = {}
    cleaned = {}
    for cond, values in values_by_cond.items():
        arr = _to_numpy(values)
        arr = arr[~np.isnan(arr)]
        cleaned[cond] = arr
        counts[cond] = arr.size
    if any(count == 0 for count in counts.values()):
        return {cond: np.nan for cond in values_by_cond}, 0
    min_trials = min(counts.values())
    rng = _get_rng(rng)
    agg = {cond: [] for cond in values_by_cond}
    for _ in range(repeats):
        for cond, clean_vals in cleaned.items():
            if clean_vals.size < min_trials:
                continue
            idx = rng.choice(clean_vals.size, size=min_trials, replace=False)
            agg[cond].append(float(np.nanmean(clean_vals[idx])))
    averaged = {cond: float(np.nanmean(vals)) if len(vals) else np.nan for cond, vals in agg.items()}
    return averaged, min_trials


def _pac_from_segment_lists(phase_segments, amp_segments):
    """Concatenate phase/amp trial segments (with length matching) and compute PAC MVL."""
    phase_concat = []
    amp_concat = []
    for phase_seg, amp_seg in zip(phase_segments, amp_segments):
        phase_arr = _to_numpy(phase_seg)
        amp_arr = _to_numpy(amp_seg)
        length = min(phase_arr.size, amp_arr.size)
        if length == 0:
            continue
        phase_concat.append(phase_arr[:length])
        amp_concat.append(amp_arr[:length])
    if not phase_concat:
        return np.nan
    phase_full = np.concatenate(phase_concat)
    amp_full = np.concatenate(amp_concat)
    return pac_mvl(phase_full, amp_full)


def _random_derangement(length: int, rng: np.random.Generator) -> np.ndarray | None:
    """Return a permutation with no fixed points (derangement)."""
    if length < 2:
        return None
    perm = rng.permutation(length)
    indices = np.arange(length)
    for idx in range(length):
        if perm[idx] == idx:
            swap_idx = (idx + 1) % length
            perm[idx], perm[swap_idx] = perm[swap_idx], perm[idx]
    if np.any(perm == indices):
        # Fallback to a simple cyclic shift to guarantee a derangement.
        perm = (indices + 1) % length
    return perm


def pac_mvl(phase, amp):
    """Amplitude-weighted MVL (amp normalized to mean=1)."""
    phase = _to_numpy(phase)
    amp = _to_numpy(amp)
    if phase.size == 0 or amp.size == 0 or phase.size != amp.size:
        return np.nan
    mask = ~np.isnan(phase) & ~np.isnan(amp)
    if not np.any(mask):
        return np.nan
    phase = phase[mask]
    amp = amp[mask]
    mean_amp = np.mean(amp)
    if mean_amp == 0:
        return np.nan
    norm_amp = amp / mean_amp
    return np.abs(np.sum(norm_amp * np.exp(1j * phase)) / np.sum(norm_amp))


def pac_trialshuffle_zscore(values_obs, values_surr):
    """Z-score an observed PAC value relative to trial-shuffled surrogates."""
    if values_obs is None:
        return np.nan
    obs = float(values_obs)
    if np.isnan(obs):
        return np.nan
    surr = _to_numpy(values_surr)
    if surr.size == 0:
        return np.nan
    surr = surr[~np.isnan(surr)]
    if surr.size == 0:
        return np.nan
    mu = surr.mean()
    sigma = surr.std(ddof=0)
    if sigma == 0:
        return np.nan
    return (obs - mu) / sigma


def pac_surrogate_pvalue(values_obs, values_surr, tail: str = "greater") -> float:
    """Empirical one- or two-tailed p-value from surrogate PAC distribution."""
    if values_obs is None:
        return np.nan
    obs = float(values_obs)
    if np.isnan(obs):
        return np.nan
    surr = _to_numpy(values_surr)
    if surr.size == 0:
        return np.nan
    surr = surr[~np.isnan(surr)]
    if surr.size == 0:
        return np.nan
    if tail == "less":
        extreme = np.sum(surr <= obs)
    elif tail == "two-sided":
        higher = np.sum(surr >= obs)
        lower = np.sum(surr <= obs)
        extreme = min(higher, lower)
    else:
        extreme = np.sum(surr >= obs)
    return (extreme + 1.0) / (surr.size + 1.0)


def build_spike_field_pairs(
    *,
    spike_times_by_unit: Mapping[int, np.ndarray],
    unit_meta: pd.DataFrame,
    theta_phase: Mapping[int, np.ndarray],
    channel_meta: pd.DataFrame,
    hemisphere_only: bool = True,
) -> pd.DataFrame:
    """Enumerate hippocampal unit × vmPFC channel pairs."""
    if not spike_times_by_unit or not theta_phase:
        return pd.DataFrame(columns=["pair_id", "unit_id", "chan_id", "hemisphere"])

    units = _as_dataframe(unit_meta)
    channels = _as_dataframe(channel_meta)
    units = units[units["area"].str.lower().str.contains("hipp")]
    channels = channels[channels["chan_id"].isin(theta_phase.keys())]
    if hemisphere_only and "hemisphere" not in channels.columns:
        channels["hemisphere"] = "unknown"
    if units.empty or channels.empty:
        return pd.DataFrame(columns=["pair_id", "unit_id", "chan_id", "hemisphere"])

    pairs = []
    for unit in units.itertuples():
        unit_hemi = getattr(unit, "hemisphere", "unknown")
        for ch in channels.itertuples():
            chan_hemi = getattr(ch, "hemisphere", "unknown")
            if hemisphere_only and unit_hemi != "unknown" and chan_hemi != "unknown" and unit_hemi != chan_hemi:
                continue
            pair_id = f"unit{unit.unit_id}_chan{ch.chan_id}"
            pairs.append(
                {
                    "pair_id": pair_id,
                    "unit_id": unit.unit_id,
                    "unit_channel": getattr(unit, "channel_id", None),
                    "unit_area": unit.area,
                    "unit_hemisphere": unit_hemi,
                    "chan_id": ch.chan_id,
                    "chan_area": ch.area,
                    "chan_hemisphere": chan_hemi,
                }
            )

    return pd.DataFrame(pairs)


def compute_spike_field_coherence(
    *,
    pair_table: pd.DataFrame,
    trial_table: pd.DataFrame,
    spike_times_by_unit: Mapping[int, np.ndarray],
    theta_phase: Mapping[int, np.ndarray],
    lfp_fs: float,
    lfp_start_time: float,
    epoch_extract: Tuple[float, float],
    epoch_analyze: Tuple[float, float],
    conditions: Sequence[str],
    min_spikes: int,
    jitter_ms: float,
    n_surrogates: int,
    repeats_eq: int = 200,
    rng_seed: int | None = None,
) -> dict:
    """Compute vmPFC-theta spike-field coherence per pair and condition."""
    if pair_table is None or len(pair_table) == 0:
        return {"summary": pd.DataFrame()}

    pairs_df = _as_dataframe(pair_table).copy()
    if "hipp_chan_id" not in pairs_df.columns:
        if "unit_channel" in pairs_df.columns:
            pairs_df["hipp_chan_id"] = pairs_df["unit_channel"]
        elif "unit_id" in pairs_df.columns:
            pairs_df["hipp_chan_id"] = pairs_df["unit_id"]
        else:
            raise ValueError("pair_table must include 'hipp_chan_id' (or 'unit_channel' / 'unit_id').")
    if "chan_id" not in pairs_df.columns:
        if "vm_chan_id" in pairs_df.columns:
            pairs_df["chan_id"] = pairs_df["vm_chan_id"]
        else:
            raise ValueError("pair_table must include 'chan_id' (vmPFC reference channel).")
    if "pair_id" not in pairs_df.columns:
        pairs_df["pair_id"] = [
            f"hipp{hip}_vm{vm}" for hip, vm in zip(pairs_df["hipp_chan_id"], pairs_df["chan_id"])
        ]
    trials_df = _as_dataframe(trial_table)
    trials_df = trials_df[trials_df["condition_label"].isin(conditions)]
    theta_channels = theta_phase.keys()

    rng = _get_rng(rng_seed)
    summary_rows = []
    per_pair = {}
    jitter_width = jitter_ms / 1000.0

    for pair in pairs_df.itertuples():
        if pair.chan_id not in theta_channels or pair.unit_id not in spike_times_by_unit:
            continue

        phase_by_cond = {cond: [] for cond in conditions}
        trial_records = {cond: [] for cond in conditions}

        for cond in conditions:
            cond_trials = trials_df[trials_df["condition_label"] == cond]
            if cond_trials.empty:
                continue

            for trial in cond_trials.itertuples():
                maint_onset = getattr(trial, "maint_onset_time", np.nan)
                if np.isnan(maint_onset):
                    continue

                analyze_start = maint_onset + epoch_analyze[0]
                analyze_end = maint_onset + epoch_analyze[1]
                spikes = _to_numpy(spike_times_by_unit[pair.unit_id])
                spike_mask = (spikes >= analyze_start) & (spikes <= analyze_end)
                spikes_in_window = spikes[spike_mask]
                if spikes_in_window.size == 0:
                    continue

                extract_start = maint_onset + epoch_extract[0]
                extract_end = maint_onset + epoch_extract[1]
                start_idx = int(max(0, np.floor((extract_start - lfp_start_time) * lfp_fs)))
                end_idx = int(min(len(theta_phase[pair.chan_id]), np.ceil((extract_end - lfp_start_time) * lfp_fs)))
                if end_idx <= start_idx:
                    continue

                phase_segment = theta_phase[pair.chan_id][start_idx:end_idx]
                seg_t0 = lfp_start_time + start_idx / lfp_fs
                phases = sample_phase_at_spikes(phase_segment, lfp_fs, seg_t0, spikes_in_window)
                if phases.size == 0:
                    continue

                phase_by_cond[cond].append(phases)
                trial_records[cond].append(
                    {
                        "spikes": spikes_in_window,
                        "win_start": analyze_start,
                        "win_end": analyze_end,
                        "phase_segment": phase_segment,
                        "segment_t0": seg_t0,
                    }
                )

        phase_lists = {cond: np.concatenate(vals) if len(vals) else np.array([]) for cond, vals in phase_by_cond.items()}
        phase_counts = {cond: arr.size for cond, arr in phase_lists.items()}
        if any(count == 0 for count in phase_counts.values()):
            continue

        equalized_mvls, min_count = equalize_spike_counts(phase_lists, repeats=repeats_eq, rng=rng)
        if min_count < min_spikes:
            continue

        per_pair[pair.pair_id] = {
            "phase_lists": phase_lists,
            "jitter": {},
            "meta": {"unit_id": pair.unit_id, "chan_id": pair.chan_id},
        }

        for cond in conditions:
            phases_flat = phase_lists[cond]
            if phases_flat.size == 0:
                continue

            jitter_sets = []
            if n_surrogates > 0:
                cond_records = trial_records[cond]
                if cond_records:
                    for _ in range(n_surrogates):
                        surrogate_phases = []
                        for record in cond_records:
                            jittered = jitter_spikes_within_trial(
                                record["spikes"],
                                record["win_start"],
                                record["win_end"],
                                jitter_width=jitter_width,
                                rng=rng,
                            )
                            if jittered.size == 0:
                                continue
                            surrogate = sample_phase_at_spikes(
                                record["phase_segment"],
                                lfp_fs,
                                record["segment_t0"],
                                jittered,
                            )
                            if surrogate.size:
                                surrogate_phases.append(surrogate)
                        if surrogate_phases:
                            jitter_sets.append(np.concatenate(surrogate_phases))

            per_pair[pair.pair_id]["jitter"][cond] = jitter_sets
            z_value = sfc_zscore(phases_flat, jitter_sets)

            summary_rows.append(
                {
                    "pair_id": pair.pair_id,
                    "unit_id": pair.unit_id,
                    "chan_id": pair.chan_id,
                    "condition": cond,
                    "n_spikes": phases_flat.size,
                    "min_spike_count": min_count,
                    "equalized_mvl": equalized_mvls.get(cond, np.nan),
                    "z_sfc_theta": z_value,
                }
            )

    summary_df = pd.DataFrame(summary_rows)
    return {"summary": summary_df, "per_pair": per_pair}


def compute_irpac(
    *,
    pair_table: pd.DataFrame,
    trial_table: pd.DataFrame,
    theta_phase: Mapping[int, np.ndarray],
    gamma_envelope: Mapping[int, np.ndarray],
    lfp_fs: float,
    lfp_start_time: float,
    epoch_analyze: Tuple[float, float],
    conditions: Sequence[str],
    min_trials: int,
    n_surrogates: int,
    repeats_eq: int = 200,
    lag_grid_s: Sequence[float] | None = None,
    exclude_lag_s: Tuple[float, float] | None = None,
    significance_alpha: float = 0.05,
    rng_seed: int | None = None,
) -> dict:
    """Compute hippocampal gamma ↔ vmPFC theta ir-PAC per pair/condition."""
    if pair_table is None or len(pair_table) == 0:
        return {"summary": pd.DataFrame()}

    pairs_df = _as_dataframe(pair_table)
    trials_df = _as_dataframe(trial_table)
    trials_df = trials_df[trials_df["condition_label"].isin(conditions)]

    rng = _get_rng(rng_seed)
    summary_rows = []
    per_pair = {}

    lag_samples = None
    if lag_grid_s is not None:
        lag_samples = np.array([int(round(sec * lfp_fs)) for sec in lag_grid_s], dtype=int)

    for pair in pairs_df.itertuples():
        vm_chan = getattr(pair, "chan_id", None)
        hipp_chan = getattr(pair, "hipp_chan_id", None)
        unit_id = getattr(pair, "unit_id", hipp_chan)
        if vm_chan not in theta_phase or hipp_chan not in gamma_envelope:
            continue

        phase_signal = theta_phase[vm_chan]
        amp_signal = gamma_envelope[hipp_chan]
        if len(phase_signal) != len(amp_signal):
            n = min(len(phase_signal), len(amp_signal))
            phase_signal = phase_signal[:n]
            amp_signal = amp_signal[:n]

        trial_segments = {cond: [] for cond in conditions}
        trial_values = {cond: [] for cond in conditions}

        for cond in conditions:
            cond_trials = trials_df[trials_df["condition_label"] == cond]
            if cond_trials.empty:
                continue
            for trial in cond_trials.itertuples():
                maint_onset = getattr(trial, "maint_onset_time", np.nan)
                if np.isnan(maint_onset):
                    continue
                analyze_start = maint_onset + epoch_analyze[0]
                analyze_end = maint_onset + epoch_analyze[1]
                start_idx = int(max(0, np.floor((analyze_start - lfp_start_time) * lfp_fs)))
                end_idx = int(min(len(phase_signal), np.ceil((analyze_end - lfp_start_time) * lfp_fs)))
                if end_idx <= start_idx:
                    continue
                phase_segment = phase_signal[start_idx:end_idx]
                amp_segment = amp_signal[start_idx:end_idx]
                length = min(len(phase_segment), len(amp_segment))
                if length == 0:
                    continue
                phase_segment = phase_segment[:length]
                amp_segment = amp_segment[:length]
                trial_segments[cond].append((phase_segment, amp_segment))
                trial_values[cond].append(pac_mvl(phase_segment, amp_segment))

        # Require data for all conditions
        counts = {cond: len([v for v in vals if not np.isnan(v)]) for cond, vals in trial_values.items()}
        if any(count == 0 for count in counts.values()):
            continue

        equalized_vals, min_count = equalize_trial_counts(trial_values, repeats=repeats_eq, rng=rng)
        if min_count < min_trials:
            continue

        per_pair[pair.pair_id] = {
            "trial_segments": trial_segments,
            "surrogates": {},
            "meta": {
                "unit_id": unit_id,
                "vm_chan": vm_chan,
                "hipp_chan": hipp_chan,
            },
        }

        for idx, cond in enumerate(conditions):
            segments = trial_segments[cond]
            if not segments:
                continue
            phase_concat = np.concatenate([seg[0] for seg in segments])
            amp_concat = np.concatenate([seg[1] for seg in segments])
            obs = pac_mvl(phase_concat, amp_concat)

            # Trial-shuffle surrogates
            surrogate_values = []
            if n_surrogates > 0 and len(segments) > 1:
                phase_segments = [seg[0] for seg in segments]
                amp_segments = [seg[1] for seg in segments]
                for _ in range(n_surrogates):
                    permuted = rng.permutation(len(amp_segments))
                    shuffled_amp_segments = [amp_segments[i] for i in permuted]
                    shuffled_amp = []
                    shuffled_phase = []
                    for phase_seg, amp_seg in zip(phase_segments, shuffled_amp_segments):
                        length = min(len(phase_seg), len(amp_seg))
                        if length == 0:
                            continue
                        shuffled_phase.append(phase_seg[:length])
                        shuffled_amp.append(amp_seg[:length])
                    if shuffled_phase and shuffled_amp:
                        surrogate_values.append(
                            pac_mvl(np.concatenate(shuffled_phase), np.concatenate(shuffled_amp))
                        )

            z_value = pac_trialshuffle_zscore(obs, surrogate_values)
            p_value = pac_surrogate_pvalue(obs, surrogate_values, tail="greater")
            is_significant = bool(not np.isnan(p_value) and p_value <= significance_alpha)

            lag_curve = None
            if lag_samples is not None:
                lag_curve = []
                for lag_idx, lag in enumerate(lag_samples):
                    if lag == 0:
                        phase_shifted = phase_concat
                        amp_shifted = amp_concat
                    elif lag > 0:
                        phase_shifted = phase_concat[lag:]
                        amp_shifted = amp_concat[:-lag]
                    else:
                        phase_shifted = phase_concat[:lag]
                        amp_shifted = amp_concat[-lag:]
                    if len(phase_shifted) == 0 or len(amp_shifted) == 0:
                        lag_curve.append(np.nan)
                    else:
                        lag_curve.append(pac_mvl(phase_shifted, amp_shifted))
                lag_curve = np.array(lag_curve)
                if exclude_lag_s and lag_grid_s is not None:
                    excl_low, excl_high = sorted(exclude_lag_s)
                    mask = (np.abs(np.array(lag_grid_s)) >= excl_low) & (np.abs(np.array(lag_grid_s)) <= excl_high)
                    lag_curve = np.where(mask, np.nan, lag_curve)

            per_pair[pair.pair_id]["surrogates"][cond] = np.array(surrogate_values)
            summary_rows.append(
                {
                    "pair_id": pair.pair_id,
                    "unit_id": unit_id,
                    "hipp_chan_id": hipp_chan,
                    "vm_chan_id": vm_chan,
                    "condition": cond,
                    "n_trials": len(segments),
                    "min_trial_count": min_count,
                    "equalized_pac": equalized_vals.get(cond, np.nan),
                    "z_pac_theta_gamma": z_value,
                    "lag_curve": lag_curve,
                    "p_pac_theta_gamma": p_value,
                    "is_pac_significant": is_significant,
                    "alpha": significance_alpha,
                }
            )

    summary_df = pd.DataFrame(summary_rows)
    return {"summary": summary_df, "per_pair": per_pair, "lag_grid_s": lag_grid_s}


def compute_pair_pac_presence(
    *,
    pac_results: dict | None,
    conditions: Sequence[str],
    min_trials: int,
    n_surrogates: int,
    direction_label: str = "hipp_to_vm",
    freq_label: str = "gamma",
    z_threshold: float = 1.64,
    percentile: float = 95.0,
    fdr_alpha: float = 0.05,
    rng_seed: int | None = None,
) -> pd.DataFrame:
    """
    Determine per-pair PAC presence by pooling all available trials across conditions.

    Trials from each condition are concatenated (after enforcing a minimum count), and
    the pooled MVL is compared against deranged trial-shuffle surrogates to obtain
    both z-scores and empirical percentile thresholds. FDR correction is optionally
    applied within each (direction, freq) group using the BH procedure.
    """
    if pac_results is None:
        return pd.DataFrame()
    per_pair = pac_results.get("per_pair")
    if not per_pair:
        return pd.DataFrame()
    if len(conditions) != 2:
        raise ValueError("compute_pair_pac_presence currently expects exactly two conditions.")

    cond_a, cond_b = conditions
    rng = _get_rng(rng_seed)
    rows = []

    for pair_id, payload in per_pair.items():
        trial_segments = payload.get("trial_segments", {})
        segments_a = list(trial_segments.get(cond_a, []))
        segments_b = list(trial_segments.get(cond_b, []))
        counts = {cond_a: len(segments_a), cond_b: len(segments_b)}
        if any(count < min_trials for count in counts.values()):
            continue
        n_star = min(counts.values())
        if n_star == 0:
            continue

        pooled_segments = []
        used_counts = {cond_a: 0, cond_b: 0}
        for cond, cond_segments in ((cond_a, segments_a), (cond_b, segments_b)):
            for segment in cond_segments:
                pooled_segments.append((cond, segment))
                used_counts[cond] += 1
        if len(pooled_segments) < 2:
            continue

        order = rng.permutation(len(pooled_segments))
        pooled_segments = [pooled_segments[i] for i in order]
        phase_segments = [seg[0] for _, seg in pooled_segments]
        amp_segments = [seg[1] for _, seg in pooled_segments]
        pac_value = _pac_from_segment_lists(phase_segments, amp_segments)
        if np.isnan(pac_value):
            continue

        surrogate_vals = []
        if n_surrogates > 0 and len(phase_segments) > 1:
            for _ in range(n_surrogates):
                perm = _random_derangement(len(phase_segments), rng)
                if perm is None:
                    break
                shuffled_amp = [amp_segments[i] for i in perm]
                surrogate = _pac_from_segment_lists(phase_segments, shuffled_amp)
                if not np.isnan(surrogate):
                    surrogate_vals.append(surrogate)

        surrogate_vals = np.array(surrogate_vals, dtype=float)
        surrogate_vals = surrogate_vals[~np.isnan(surrogate_vals)]
        mu = float(np.mean(surrogate_vals)) if surrogate_vals.size else np.nan
        sigma = float(np.std(surrogate_vals, ddof=0)) if surrogate_vals.size else np.nan
        z_value = float((pac_value - mu) / sigma) if surrogate_vals.size and sigma > 0 else np.nan
        perc_threshold = (
            float(np.percentile(surrogate_vals, percentile)) if surrogate_vals.size else np.nan
        )
        p_empirical = (
            float(1.0 + np.sum(surrogate_vals >= pac_value)) / float(surrogate_vals.size + 1.0)
            if surrogate_vals.size
            else np.nan
        )

        rows.append(
            {
                "pair_id": pair_id,
                "direction": direction_label,
                "freq_label": freq_label,
                "cond_a_label": cond_a,
                "cond_b_label": cond_b,
                "n_trials_cond_a": counts[cond_a],
                "n_trials_cond_b": counts[cond_b],
                "n_star": int(n_star),
                "n_used_cond_a": int(used_counts[cond_a]),
                "n_used_cond_b": int(used_counts[cond_b]),
                "n_pooled_trials": int(len(pooled_segments)),
                "pac_value": float(pac_value),
                "surrogate_mean": mu,
                "surrogate_std": sigma,
                "surrogate_percentile": perc_threshold,
                "z_pooled": z_value,
                "p_empirical": p_empirical,
                "is_pac_positive_z": bool(not np.isnan(z_value) and z_value >= z_threshold),
                "is_pac_positive_percentile": bool(
                    not np.isnan(perc_threshold) and pac_value > perc_threshold
                ),
                "n_surrogates": int(surrogate_vals.size),
                "z_threshold": float(z_threshold),
                "percentile": float(percentile),
                "fdr_alpha": float(fdr_alpha),
                "meta_unit_id": payload.get("meta", {}).get("unit_id"),
                "meta_vm_chan": payload.get("meta", {}).get("vm_chan"),
                "meta_hipp_chan": payload.get("meta", {}).get("hipp_chan"),
            }
        )
        continue

    presence_df = pd.DataFrame(rows)
    if presence_df.empty:
        return presence_df

    presence_df["p_fdr"] = np.nan
    presence_df["fdr_significant"] = False
    if fdr_alpha is not None and fdr_alpha > 0:
        grouped = presence_df.groupby(["direction", "freq_label"], dropna=False)
        for _, idx in grouped.groups.items():
            idx = list(idx)
            if not idx:
                continue
            mask = presence_df.loc[idx, "p_empirical"].notna()
            if not mask.any():
                continue
            valid_idx = np.array(idx)[mask.to_numpy(dtype=bool)]
            pvals = presence_df.loc[valid_idx, "p_empirical"].to_numpy(dtype=float)
            reject, corrected = fdrcorrection(pvals, alpha=fdr_alpha)
            presence_df.loc[valid_idx, "p_fdr"] = corrected
            presence_df.loc[valid_idx, "fdr_significant"] = reject

    return presence_df


def run_session_stats(
    *,
    sfc_results: dict,
    pac_results: dict,
    conditions: Sequence[str],
) -> dict:
    """Summarize pair-level metrics into session-level tests."""
    if len(conditions) != 2:
        raise ValueError("run_session_stats currently expects exactly two conditions.")
    cond_a, cond_b = conditions

    stats = {}
    for label, results_key, value_col in [
        ("sfc_theta", sfc_results, "z_sfc_theta"),
        ("pac_theta_gamma", pac_results, "z_pac_theta_gamma"),
    ]:
        summary_df = results_key.get("summary") if isinstance(results_key, dict) else None
        if summary_df is None or summary_df.empty:
            stats[label] = None
            continue
        pivot = summary_df.pivot_table(index="pair_id", columns="condition", values=value_col)
        if cond_a not in pivot.columns or cond_b not in pivot.columns:
            stats[label] = None
            continue
        pivot = pivot.dropna(subset=[cond_a, cond_b])
        if len(pivot) == 0:
            stats[label] = None
            continue
        delta = pivot[cond_b] - pivot[cond_a]
        n_pairs = len(delta)
        if np.allclose(delta, 0):
            statistic, p_value = 0.0, 1.0
        else:
            try:
                statistic, p_value = wilcoxon(delta)
            except ValueError:
                statistic, p_value = 0.0, 1.0
        expected = n_pairs * (n_pairs + 1) / 4.0
        variance = n_pairs * (n_pairs + 1) * (2 * n_pairs + 1) / 24.0
        std = np.sqrt(variance) if variance > 0 else 1.0
        z_value = (statistic - expected) / std if std > 0 else 0.0
        effect_size = z_value / np.sqrt(n_pairs) if n_pairs > 0 else np.nan
        ci_low, ci_high = _bca_ci(delta.values, statistic=np.median)
        stats[label] = {
            "n_pairs": n_pairs,
            "median_a": float(np.median(pivot[cond_a])),
            "median_b": float(np.median(pivot[cond_b])),
            "median_delta": float(np.median(delta)),
            "iqr_a": (float(np.percentile(pivot[cond_a], 25)), float(np.percentile(pivot[cond_a], 75))),
            "iqr_b": (float(np.percentile(pivot[cond_b], 25)), float(np.percentile(pivot[cond_b], 75))),
            "p_value": float(p_value),
            "effect_size_r": float(effect_size),
            "delta_ci95": (ci_low, ci_high),
            "wilcoxon_stat": float(statistic),
            "wilcoxon_z": float(z_value),
        }

    return stats


def plot_pair_level_qc(
    *,
    pair_id: str,
    sfc_results: dict,
    pac_results: dict | None,
    theta_phase: Mapping[int, np.ndarray],
    gamma_envelope: Mapping[int, np.ndarray] | None,
    trial_table: pd.DataFrame,
    lfp_fs: float,
    epoch_analyze: Tuple[float, float],
    lfp_start_time: float,
    conditions: Sequence[str],
    output_dir: str | Path,
    dpi: int = 150,
) -> Path | None:
    """Plot per-pair QC including phase histograms and example traces."""
    if pair_id not in sfc_results.get("per_pair", {}):
        return None

    pair_info = sfc_results["per_pair"][pair_id]
    meta = pair_info.get("meta", {})
    chan_id = meta.get("chan_id")
    if chan_id is None or chan_id not in theta_phase:
        return None

    theta_signal = theta_phase[chan_id]
    hipp_chan = None
    if pac_results and "per_pair" in pac_results and pair_id in pac_results["per_pair"]:
        hipp_chan = pac_results["per_pair"][pair_id]["meta"].get("hipp_chan")
    gamma_signal = gamma_envelope.get(hipp_chan) if gamma_envelope and hipp_chan in gamma_envelope else None

    fig = plt.figure(figsize=(12, 5))
    ax_polar = fig.add_subplot(1, 2, 1, projection="polar")
    colors = sns.color_palette("Set2", len(conditions))

    for color, cond in zip(colors, conditions):
        phases = pair_info["phase_lists"].get(cond)
        if phases is None or len(phases) == 0:
            continue
        ax_polar.hist(phases, bins=30, alpha=0.5, color=color, label=cond)
    ax_polar.set_title(f"{pair_id} – Phase histograms")
    ax_polar.legend(loc="upper right", fontsize=8)

    ax_ts = fig.add_subplot(1, 2, 2)
    example_trial = trial_table[trial_table["condition_label"].isin(conditions)].head(1)
    if not example_trial.empty:
        trial = example_trial.iloc[0]
        start_time = trial["maint_onset_time"] + epoch_analyze[0]
        end_time = start_time + 1.5  # 1.5 s snippet
        start_idx = int(max(0, np.floor((start_time - lfp_start_time) * lfp_fs)))
        end_idx = int(min(len(theta_signal), np.ceil((end_time - lfp_start_time) * lfp_fs)))
        times = np.arange(start_idx, end_idx) / lfp_fs + lfp_start_time
        ax_ts.plot(times, np.sin(theta_signal[start_idx:end_idx]), label="vmPFC θ (sin)", color="#1f77b4")
        if gamma_signal is not None:
            ax_ts.plot(times, gamma_signal[start_idx:end_idx], label="Hipp γ amp", color="#ff7f0e", alpha=0.7)
        ax_ts.set_xlabel("Time (s)")
        ax_ts.set_ylabel("Normalized amplitude")
        ax_ts.set_title(f"{pair_id} – Example trial snippet")
        ax_ts.legend()
    else:
        ax_ts.axis("off")
        ax_ts.set_title("No example trial available")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{pair_id}_qc.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return out_path


def plot_session_summary(
    *,
    sfc_results: dict,
    pac_results: dict,
    stats_summary: dict,
    conditions: Sequence[str],
    output_dir: str | Path,
    dpi: int = 150,
) -> list[Path]:
    """Generate session-level summary plots for SFC and PAC metrics."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    plots = []
    config = [
        ("sfc", sfc_results, "z_sfc_theta", "SFC θ z-score"),
        ("pac", pac_results, "z_pac_theta_gamma", "ir-PAC θ×γ z-score"),
    ]

    for key, result_dict, value_col, title in config:
        df = result_dict.get("summary") if result_dict else None
        if df is None or df.empty:
            continue
        pivot = df.pivot_table(index="pair_id", columns="condition", values=value_col)
        if any(cond not in pivot.columns for cond in conditions):
            continue
        pivot = pivot.dropna(subset=conditions)
        if pivot.empty:
            continue
        cond_a, cond_b = conditions
        delta = pivot[cond_b] - pivot[cond_a]

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        for _, row in pivot.iterrows():
            axes[0].plot(conditions, [row[cond_a], row[cond_b]], color="0.7", alpha=0.6)
        axes[0].scatter(
            [cond_a] * len(pivot),
            pivot[cond_a],
            color="#1f77b4",
            label=cond_a,
        )
        axes[0].scatter(
            [cond_b] * len(pivot),
            pivot[cond_b],
            color="#ff7f0e",
            label=cond_b,
        )
        axes[0].set_title(f"{title}: paired values")
        axes[0].set_ylabel(value_col)
        axes[0].legend()

        sns.violinplot(y=delta, ax=axes[1], color="#c5c9e5", inner="box")
        axes[1].axhline(0, color="k", linestyle="--", linewidth=1)
        axes[1].set_title(f"{title}: Δ({cond_b}−{cond_a})")
        axes[1].set_ylabel("Delta")

        stats = stats_summary.get("sfc_theta" if key == "sfc" else "pac_theta_gamma") if stats_summary else None
        if stats:
            axes[1].text(
                0.05,
                0.95,
                f"n={stats['n_pairs']}\np={stats['p_value']:.3g}\nmedian Δ={stats['median_delta']:.2f}",
                transform=axes[1].transAxes,
                va="top",
            )

        fig.suptitle(title)
        fig.tight_layout()
        out_path = output_dir / f"{key}_session_summary.png"
        fig.savefig(out_path, dpi=dpi)
        plt.close(fig)
        plots.append(out_path)

    return plots


def write_pac_presence_to_nwb(
    *,
    session_path: str | Path,
    pac_presence: pd.DataFrame,
    module_name: str = "irpac",
    table_name: str = "pac_presence",
) -> Path | None:
    """Store per-pair PAC presence metrics inside the NWB processing module."""
    if pac_presence is None or pac_presence.empty:
        return None
    session_path = Path(session_path)
    if not session_path.exists():
        raise FileNotFoundError(session_path)

    io = NWBHDF5IO(str(session_path), "r+")
    nwbfile = io.read()
    try:
        if module_name in nwbfile.processing:
            module = nwbfile.processing[module_name]
        else:
            module = nwbfile.create_processing_module(
                name=module_name,
                description="Cross-frequency coupling analysis outputs (ir-PAC).",
            )
        if table_name in module.data_interfaces:
            del module.data_interfaces[table_name]

        table = DynamicTable(
            name=table_name,
            description="Per-pair hippocampus↔vmPFC ir-PAC presence metrics (all trials).",
        )
        for column in pac_presence.columns:
            table.add_column(name=column, description=f"{column} (ir-PAC metric)")

        for _, row in pac_presence.iterrows():
            clean_row = {}
            for key, value in row.items():
                if isinstance(value, np.generic):
                    clean_row[key] = value.item()
                elif pd.isna(value):
                    clean_row[key] = np.nan
                else:
                    clean_row[key] = value
            table.add_row(**clean_row)

        module.add_data_interface(table)
        io.write(nwbfile)
    finally:
        io.close()
    return session_path


def save_session_outputs(
    *,
    session_id: str,
    output_dir: str | Path,
    session_path: str | Path | None = None,
    pair_table: pd.DataFrame | None,
    sfc_results: dict | None,
    pac_results: dict | None,
    pac_presence: pd.DataFrame | None = None,
    stats_summary: dict | None,
    analysis_params: dict | None = None,
) -> dict:
    """Persist CSV/JSON artifacts for downstream reports."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts = {}

    if pair_table is not None and not pair_table.empty:
        pairs_path = output_dir / f"{session_id}_pairs.csv"
        pair_table.to_csv(pairs_path, index=False)
        artifacts["pair_table"] = pairs_path

    if sfc_results and isinstance(sfc_results.get("summary"), pd.DataFrame) and not sfc_results["summary"].empty:
        sfc_path = output_dir / f"{session_id}_sfc_summary.csv"
        sfc_results["summary"].to_csv(sfc_path, index=False)
        artifacts["sfc_summary"] = sfc_path

    if pac_results and isinstance(pac_results.get("summary"), pd.DataFrame) and not pac_results["summary"].empty:
        pac_path = output_dir / f"{session_id}_pac_summary.csv"
        pac_results["summary"].to_csv(pac_path, index=False)
        artifacts["pac_summary"] = pac_path

    if (
        session_path is not None
        and pac_presence is not None
        and isinstance(pac_presence, pd.DataFrame)
        and not pac_presence.empty
    ):
        try:
            write_pac_presence_to_nwb(session_path=session_path, pac_presence=pac_presence)
            artifacts["pac_presence_nwb"] = str(Path(session_path))
        except Exception as exc:  # noqa: BLE001
            warnings.warn(f"Failed to store PAC presence in NWB: {exc}")

    summary_payload = {
        "session_id": session_id,
        "stats": stats_summary,
        "analysis_params": analysis_params,
    }
    summary_path = output_dir / f"{session_id}_summary.json"
    summary_path.write_text(json.dumps(summary_payload, indent=2))
    artifacts["summary_json"] = summary_path

    return artifacts


def compute_vmPFC_theta_phase(
    *,
    lfp_by_channel: Mapping[int, np.ndarray],
    channel_meta: pd.DataFrame,
    sampling_rate: float,
    theta_band: Tuple[float, float],
    car_mode: str | None = "hemisphere",
    hilbert_pad_sec: float = 1.0,
    filter_order: int = 4,
    exclude_bad: bool = True,
    bad_flag_column: str = "bad",
    max_peak_to_peak: float | None = None,
) -> Dict[int, np.ndarray]:
    """
    Compute theta-band phase traces for vmPFC channels with optional CAR.
    """
    df = _as_dataframe(channel_meta)
    areas = df["area"].fillna("") if "area" in df.columns else pd.Series("", index=df.index)
    mask = areas.str.lower().str.contains("vmpfc")
    vm_df = df[mask].copy()
    if vm_df.empty:
        return {}

    signals = {}
    valid_chan_ids = []
    for row in vm_df.itertuples():
        if exclude_bad and bad_flag_column in vm_df.columns:
            if bool(getattr(row, bad_flag_column, False)):
                continue
        signal = _to_numpy(lfp_by_channel.get(row.chan_id))
        if signal.size == 0:
            continue
        if max_peak_to_peak is not None and np.ptp(signal) > max_peak_to_peak:
            continue
        signals[row.chan_id] = signal
        valid_chan_ids.append(row.chan_id)

    vm_df = vm_df[vm_df["chan_id"].isin(valid_chan_ids)]
    if vm_df.empty:
        return {}

    if car_mode in {"region", "hemisphere"}:
        if car_mode == "region":
            group_key = vm_df["area"].fillna("vmPFC")
        else:
            group_key = vm_df["hemisphere"].fillna("unknown") if "hemisphere" in vm_df.columns else pd.Series("unknown", index=vm_df.index)
        for group_val in group_key.dropna().unique():
            group_ids = vm_df[group_key == group_val]["chan_id"].tolist()
            if len(group_ids) < 2:
                continue
            stack = np.vstack([signals[cid] for cid in group_ids])
            group_mean = np.mean(stack, axis=0)
            for cid in group_ids:
                signals[cid] = signals[cid] - group_mean

    pad_samples = int(hilbert_pad_sec * sampling_rate)
    phase_dict = {}
    for chan_id, signal in signals.items():
        padded, valid_slice = _pad_signal(signal, pad_samples)
        filtered = bandpass_filtfilt(padded, sampling_rate, theta_band, order=filter_order)
        analytic = hilbert(filtered)
        phase = np.angle(analytic[valid_slice]) if pad_samples > 0 else np.angle(analytic)
        phase_dict[chan_id] = phase
    return phase_dict


def compute_hipp_gamma_envelope(
    *,
    lfp_by_channel: Mapping[int, np.ndarray],
    channel_meta: pd.DataFrame,
    sampling_rate: float,
    gamma_band: Tuple[float, float],
    hilbert_pad_sec: float = 1.0,
    filter_order: int = 4,
    exclude_bad: bool = True,
    bad_flag_column: str = "bad",
    max_peak_to_peak: float | None = None,
) -> Dict[int, np.ndarray]:
    """Compute hippocampal gamma-band amplitude envelopes."""
    df = _as_dataframe(channel_meta)
    areas = df["area"].fillna("")
    mask = areas.str.lower().str.contains("hipp")
    hipp_df = df[mask].copy()
    if hipp_df.empty:
        return {}

    pad_samples = int(hilbert_pad_sec * sampling_rate)
    amp_dict = {}
    for row in hipp_df.itertuples():
        if exclude_bad and bad_flag_column in hipp_df.columns:
            if bool(getattr(row, bad_flag_column, False)):
                continue
        signal = _to_numpy(lfp_by_channel.get(row.chan_id))
        if signal.size == 0:
            continue
        if max_peak_to_peak is not None and np.ptp(signal) > max_peak_to_peak:
            continue
        padded, valid_slice = _pad_signal(signal, pad_samples)
        filtered = bandpass_filtfilt(padded, sampling_rate, gamma_band, order=filter_order)
        analytic = hilbert(filtered)
        amp = np.abs(analytic[valid_slice]) if pad_samples > 0 else np.abs(analytic)
        amp_dict[row.chan_id] = amp
    return amp_dict
