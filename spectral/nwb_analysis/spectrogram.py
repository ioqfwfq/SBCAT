"""
Time-frequency spectrogram computation and plotting functions.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import spectrogram
from scipy.interpolate import interp1d
import pywt

from .config import EVENT_DEFINITIONS, EVENT_SEQUENCE


def compute_cwt_morlet(signal_data, sampling_rate, freq_range=(1, 150), n_freqs=100):
    """
    Compute Continuous Wavelet Transform using Morlet wavelets.

    Args:
        signal_data: 1D array of LFP data
        sampling_rate: Sampling frequency in Hz
        freq_range: Tuple of (min_freq, max_freq) in Hz
        n_freqs: Number of frequency bins (logarithmically spaced)

    Returns:
        frequencies: Array of frequency values (Hz)
        time_vector: Array of time values (seconds, same length as input)
        power: 2D array (n_freqs, n_times) of power values
    """
    # Create logarithmically-spaced frequencies for better representation
    # (more resolution at low frequencies, which is natural for wavelets)
    frequencies = np.logspace(
        np.log10(freq_range[0]),
        np.log10(freq_range[1]),
        n_freqs
    )

    # Convert frequencies to scales for PyWavelets
    # For Morlet wavelet: scale = center_frequency / (frequency * sampling_period)
    # PyWavelets uses: freq = center_freq / (scale * sampling_period)
    # Therefore: scale = center_freq / (freq * sampling_period)
    # With sampling_period = 1/sampling_rate:
    # scale = center_freq * sampling_rate / freq

    # Get the center frequency of the Morlet wavelet
    wavelet = pywt.ContinuousWavelet('morl')
    center_freq = pywt.central_frequency(wavelet)

    # Compute scales from frequencies
    scales = center_freq * sampling_rate / frequencies

    # Compute CWT using PyWavelets
    # Returns coefficients (complex values) and frequencies
    cwt_coeffs, _ = pywt.cwt(signal_data, scales, 'morl', sampling_period=1.0/sampling_rate)

    # Compute power (absolute value squared)
    power = np.abs(cwt_coeffs) ** 2

    # Time vector: one sample per time point in the original signal
    time_vector = np.arange(len(signal_data)) / sampling_rate

    return frequencies, time_vector, power


def extract_event_aligned_segments(lfp_series, trial, sampling_rate, channels,
                                   event_definitions=None, padding_sec=0.0):
    """
    Extract event-aligned LFP segments for configured task events.

    Args:
        lfp_series: ElectricalSeries object
        trial: Single row from trials DataFrame
        sampling_rate: LFP sampling rate
        channels: List of channel indices to extract
        event_definitions: Dict mapping event names to timestamp column and window.
                         If None, uses EVENT_DEFINITIONS from config.
        padding_sec: Padding duration in seconds to add before/after each segment
                    to reduce edge artifacts in wavelet analysis.

    Returns:
        dict with:
            - 'segments': {event_name: array (n_channels, n_samples) or None}
            - 'response_latency': Response time relative to probe (seconds, NaN if unavailable)
            - 'padding_samples': Number of padding samples added on each side
    """
    if event_definitions is None:
        event_definitions = EVENT_DEFINITIONS
    
    if len(channels) == 0:
        return None

    n_total_samples = lfp_series.data.shape[0]
    padding_samples = int(np.round(padding_sec * sampling_rate))

    segments = {event: None for event in event_definitions.keys()}

    # Determine response latency relative to probe
    probe_ts = trial.get('timestamps_Probe', np.nan)
    response_ts = trial.get('timestamps_Response', np.nan)
    response_latency = np.nan
    if pd.notna(probe_ts) and pd.notna(response_ts) and probe_ts > 0 and response_ts > probe_ts:
        response_latency = response_ts - probe_ts

    for event_name, config in event_definitions.items():
        timestamp_col = config['timestamp']
        window_start, window_end = config['window']
        timestamp = trial.get(timestamp_col, np.nan)

        if pd.isna(timestamp) or timestamp <= 0:
            continue

        expected_samples = int(np.round((window_end - window_start) * sampling_rate))
        if expected_samples <= 0:
            continue

        # Extract with padding to reduce edge artifacts
        start_time = timestamp + window_start
        start_idx = int(np.round(start_time * sampling_rate))
        end_idx = start_idx + expected_samples

        # Add padding on both sides
        start_idx_padded = max(0, start_idx - padding_samples)
        end_idx_padded = min(n_total_samples, end_idx + padding_samples)

        # Skip if padded window goes out of bounds
        if start_idx_padded < 0 or end_idx_padded > n_total_samples:
            continue

        # Check that we have enough data even with padding
        if end_idx_padded <= start_idx_padded:
            continue

        # Extract PADDED segment
        if lfp_series.data.ndim == 1:
            segment = lfp_series.data[start_idx_padded:end_idx_padded]
            segment = segment.reshape(-1, 1)
            if len(channels) > 1 or channels[0] != 0:
                continue
        else:
            segment = lfp_series.data[start_idx_padded:end_idx_padded, :]
            segment = segment[:, channels]

        # No trimming here - we keep the padding for CWT
        # Trimming will happen after CWT computation

        segments[event_name] = segment.T  # (n_channels, n_samples_padded)

    return {
        'segments': segments,
        'response_latency': response_latency,
        'padding_samples': padding_samples
    }


def compute_spectrogram_by_groups_stft(nwb_data, trials_df, region_channels, region_name,
                                       freq_range=(1, 50), n_freqs=100,
                                       nperseg_sec=0.4, noverlap_sec=0.35,
                                       groups=None, event_definitions=None, event_sequence=None):
    """
    Compute time-frequency maps using STFT (Short-Time Fourier Transform).

    Fast method with coarser time resolution and linearly-spaced frequencies.
    Uses padding to reduce edge artifacts and per-channel, per-frequency baseline z-scoring.

    Args:
        nwb_data: Dict with NWB file data
        trials_df: DataFrame with trial information
        region_channels: Dict mapping region names to channel lists
        region_name: Name of brain region
        freq_range: Tuple of (min_freq, max_freq) in Hz
        n_freqs: Number of linearly-spaced frequency bins
        nperseg_sec: STFT window size in seconds
        noverlap_sec: STFT overlap in seconds
        groups: Optional list/tuple of index arrays, one per group. Each entry is an
            array-like of trial indices (0-based) into trials_df. If None or empty,
            all trials are treated as a single group.
        event_definitions: Dict mapping event names to config. If None, uses EVENT_DEFINITIONS.
        event_sequence: List of event names in order. If None, uses EVENT_SEQUENCE.

    Returns:
        dict with:
            - 'groups': List of per-group result dicts
            - 'meta': Shared metadata across groups
    """
    if event_definitions is None:
        event_definitions = EVENT_DEFINITIONS
    if event_sequence is None:
        event_sequence = EVENT_SEQUENCE

    lfp_series = nwb_data['lfp']['series']
    sampling_rate = nwb_data['lfp']['sampling_rate']
    channels = region_channels.get(region_name, [])

    if len(channels) == 0:
        return None

    # Validate channel indices
    n_channels_available = lfp_series.data.shape[1] if lfp_series.data.ndim > 1 else 1
    valid_channels = [ch for ch in channels if ch < n_channels_available]

    if len(valid_channels) == 0:
        print(f"    WARNING: No valid channels for {region_name} (requested {len(channels)}, file has {n_channels_available})")
        return None

    if len(valid_channels) < len(channels):
        print(f"    WARNING: Some channels out of bounds for {region_name} (using {len(valid_channels)}/{len(channels)} channels)")

    channels = valid_channels

    # STFT uses linearly-spaced frequencies
    freq_bins = np.linspace(freq_range[0], freq_range[1], n_freqs)
    print(f"    Using STFT with {n_freqs} linearly-spaced frequencies")

    # Compute padding to reduce edge artifacts
    min_freq = freq_range[0]
    padding_sec = max(0.5, 1.5 / min_freq)  # At least 0.5 second or 1.5 cycles
    print(f"    Using {padding_sec:.2f}s padding to reduce edge artifacts (min freq: {min_freq} Hz)")

    # Pre-compute event offsets and STFT parameters
    event_offsets = {}
    cumulative_offset = 0.0
    event_params = {}
    event_time_grid = {}
    event_boundaries = {}
    time_vectors_available = []

    for event_name in event_sequence:
        window_start, window_end = event_definitions[event_name]['window']
        duration_sec = window_end - window_start
        expected_samples = int(np.round(duration_sec * sampling_rate))

        event_offsets[event_name] = cumulative_offset
        global_start = cumulative_offset + window_start
        global_end = cumulative_offset + window_end
        event_boundaries[event_name] = (global_start, global_end)
        cumulative_offset += max(duration_sec, 0)

        if expected_samples <= 1:
            event_params[event_name] = {'expected_samples': expected_samples}
            event_time_grid[event_name] = np.array([])
            continue

        # STFT: coarser resolution based on window size and overlap
        nperseg = int(np.round(nperseg_sec * sampling_rate))
        noverlap = int(np.round(noverlap_sec * sampling_rate))
        nstep = nperseg - noverlap

        if nperseg > expected_samples:
            event_params[event_name] = {'expected_samples': expected_samples}
            event_time_grid[event_name] = np.array([])
            continue

        # Compute time points spanning full event duration (0 to duration_sec)
        # With padding, we can get valid spectral estimates at the edges
        # Create evenly-spaced time points from 0 to duration
        t_tmp = np.arange(0, duration_sec, nstep / sampling_rate)

        # Ensure we include the endpoint if close enough
        if duration_sec - t_tmp[-1] > nstep / (2.0 * sampling_rate):
            t_tmp = np.append(t_tmp, duration_sec)

        event_time_grid[event_name] = global_start + t_tmp
        time_vectors_available.append(event_time_grid[event_name])

        event_params[event_name] = {
            'expected_samples': expected_samples,
            'nperseg': nperseg,
            'noverlap': noverlap,
            'nstep': nstep
        }

    if len(time_vectors_available) == 0:
        return None

    global_time_vector = np.concatenate(time_vectors_available)
    print(f"    Computing STFT time-frequency maps for all trials (event-aligned)...")

    trial_spectrograms = []
    trial_event_times = []
    trial_positions = []

    for trial_idx, (_, trial) in enumerate(trials_df.iterrows()):
        trial_segments = extract_event_aligned_segments(
            lfp_series, trial, sampling_rate, channels, event_definitions, padding_sec
        )

        if trial_segments is None:
            continue

        segments = trial_segments['segments']
        response_latency = trial_segments['response_latency']
        padding_samples = trial_segments['padding_samples']

        event_matrices = []
        event_presence = {event: False for event in event_sequence}

        for event_name in event_sequence:
            params = event_params[event_name]
            expected_samples = params['expected_samples']
            time_grid = event_time_grid[event_name]

            if expected_samples <= 1 or len(time_grid) == 0:
                event_matrices.append(np.full((len(channels), len(freq_bins), 0), np.nan))
                continue

            segment = segments.get(event_name)
            if segment is None:
                event_matrices.append(np.full((len(channels), len(freq_bins), len(time_grid)), np.nan))
                continue

            # Compute STFT for each channel
            nperseg = params['nperseg']
            noverlap = params['noverlap']
            nstep = params['nstep']
            n_times_expected = len(time_grid)

            channel_power = []
            for ch_idx in range(segment.shape[0]):
                lfp_channel = segment[ch_idx, :]

                # Compute STFT
                f_stft, t_stft, Zxx = spectrogram(
                    lfp_channel,
                    fs=sampling_rate,
                    nperseg=nperseg,
                    noverlap=noverlap,
                    window='hann',
                    detrend='constant',
                    scaling='density'
                )

                # Filter to requested frequency range
                freq_mask = (f_stft >= freq_range[0]) & (f_stft <= freq_range[1])
                f_filtered = f_stft[freq_mask]
                Zxx_filtered = Zxx[freq_mask, :]

                # Interpolate to match requested number of frequency bins
                if len(f_filtered) != len(freq_bins):
                    interp_func = interp1d(f_filtered, Zxx_filtered, axis=0,
                                         kind='linear', fill_value='extrapolate')
                    Zxx_resampled = interp_func(freq_bins)
                else:
                    Zxx_resampled = Zxx_filtered

                power = Zxx_resampled

                # Map STFT time points to event-relative time and select desired windows
                # t_stft is relative to padded signal start (starting at nperseg/(2*fs))
                # We want windows corresponding to event time [0, duration_sec]

                # STFT time relative to unpadded event start
                t_stft_relative = t_stft - padding_samples / sampling_rate

                # Find STFT windows that fall within [0, duration_sec] or close to target times
                # Target times from time_grid (event-relative, 0 to duration_sec)
                # time_grid contains global times, need to convert to event-relative
                event_start_global = event_boundaries[event_name][0]
                target_times = time_grid - event_start_global

                # For each target time, find closest STFT window
                if len(target_times) > 0 and len(t_stft_relative) > 0:
                    # Simple approach: interpolate to target time grid
                    # Check if we have enough coverage
                    if t_stft_relative[0] <= 0.1 and t_stft_relative[-1] >= duration_sec - 0.1:
                        # Good coverage, interpolate
                        interp_func = interp1d(t_stft_relative, power, axis=1,
                                             kind='linear', bounds_error=False,
                                             fill_value='extrapolate')
                        power = interp_func(target_times)
                    else:
                        # Insufficient coverage, use closest windows
                        indices = np.searchsorted(t_stft_relative, target_times)
                        indices = np.clip(indices, 0, len(t_stft_relative) - 1)
                        power = power[:, indices]

                # Ensure expected number of time points (backup)
                if power.shape[1] != n_times_expected:
                    if power.shape[1] > n_times_expected:
                        # Trim excess
                        power = power[:, :n_times_expected]
                    else:
                        # Pad with NaN
                        pad_width = n_times_expected - power.shape[1]
                        power = np.pad(power, ((0, 0), (0, pad_width)),
                                     constant_values=np.nan)

                channel_power.append(power)

            if len(channel_power) == 0:
                event_matrices.append(np.full((len(channels), len(freq_bins), len(time_grid)), np.nan))
                continue

            channel_power = np.array(channel_power)
            event_matrices.append(channel_power)
            event_presence[event_name] = True

        if len(event_matrices) == 0:
            continue

        trial_matrix = np.concatenate(event_matrices, axis=2)
        if np.all(np.isnan(trial_matrix)):
            continue

        trial_spectrograms.append(trial_matrix)
        trial_positions.append(trial_idx)

        # Record per-trial event times
        event_time_map = {event: (event_offsets[event] if event_presence[event] else np.nan)
                          for event in event_sequence}

        if not np.isnan(response_latency) and event_presence.get('probe', False):
            response_global = event_offsets['probe'] + response_latency
            probe_window_end = event_offsets['probe'] + event_definitions['probe']['window'][1]
            event_time_map['response'] = response_global if response_global <= probe_window_end else np.nan
        else:
            event_time_map['response'] = np.nan

        trial_event_times.append(event_time_map)

    if len(trial_spectrograms) == 0:
        return None

    spectrogram_array = np.stack(trial_spectrograms)

    # Baseline normalization
    baseline_mask = (global_time_vector >= -1.0) & (global_time_vector < 0.0)
    if not np.any(baseline_mask):
        raise ValueError("Baseline window [-1, 0) is empty")

    baseline_data = spectrogram_array[:, :, :, baseline_mask]

    with np.errstate(invalid='ignore'):
        pooled_baseline_mean = np.nanmean(baseline_data, axis=(0, 3))
        pooled_baseline_std = np.nanstd(baseline_data, axis=(0, 3))

    pooled_baseline_std[pooled_baseline_std < 1e-12] = 1.0

    baseline_mean_expanded = pooled_baseline_mean[np.newaxis, :, :, np.newaxis]
    baseline_std_expanded = pooled_baseline_std[np.newaxis, :, :, np.newaxis]

    spectrogram_z = (spectrogram_array - baseline_mean_expanded) / baseline_std_expanded

    n_trials_total = spectrogram_array.shape[0]
    trial_positions = np.array(trial_positions)

    # Map groups
    if groups is None or len(groups) == 0:
        group_masks = [np.ones(n_trials_total, dtype=bool)]
    else:
        group_masks = []
        for g in groups:
            group_trial_indices = np.asarray(g, dtype=int)
            mask = np.isin(trial_positions, group_trial_indices)
            group_masks.append(mask)

    results_groups = []
    event_keys = list(trial_event_times[0].keys()) if len(trial_event_times) > 0 else []

    for mask in group_masks:
        if not np.any(mask):
            continue

        group_power_linear = spectrogram_array[mask]
        group_power_zscored = spectrogram_z[mask]

        with np.errstate(invalid='ignore'):
            # Average trials for each channel separately (keep channel dimension)
            # Shape: (n_trials, n_channels, n_freqs, n_times) -> (n_channels, n_freqs, n_times)
            power_linear_per_channel = np.nanmean(group_power_linear, axis=0)
            power_zscored_per_channel = np.nanmean(group_power_zscored, axis=0)

            # For plotting, compute region-averaged versions
            power_linear_mean = np.nanmean(power_linear_per_channel, axis=0)  # (n_freqs, n_times)
            power_zscored = np.nanmean(power_zscored_per_channel, axis=0)

            power_mean = 10 * np.log10(power_linear_mean + 1e-12)

        avg_event_times = {}
        for key in event_keys:
            values = [trial_event_times[i][key] for i in range(len(trial_event_times))
                      if mask[i] and not np.isnan(trial_event_times[i][key])]
            avg_event_times[key] = np.mean(values) if len(values) > 0 else np.nan

        original_indices = trial_positions[mask]

        results_groups.append({
            'time_vector': global_time_vector,
            'frequencies': freq_bins,
            'power_mean': power_mean,
            'power_zscored': power_zscored,
            'power_zscored_per_channel': power_zscored_per_channel,  # (n_channels, n_freqs, n_times)
            'n_trials': int(np.sum(mask)),
            'n_channels': power_zscored_per_channel.shape[0],
            'event_times': avg_event_times,
            'event_boundaries': {k: tuple(v) for k, v in event_boundaries.items()},
            'event_sequence': list(event_sequence),
            'region': region_name,
            'pooled_baseline_mean': pooled_baseline_mean,
            'pooled_baseline_std': pooled_baseline_std,
            'trial_indices': original_indices
        })

    return {
        'groups': results_groups,
        'meta': {
            'event_boundaries': {k: tuple(v) for k, v in event_boundaries.items()},
            'event_sequence': list(event_sequence),
            'region': region_name,
            'pooled_baseline_mean': pooled_baseline_mean,
            'pooled_baseline_std': pooled_baseline_std
        }
    }


def compute_spectrogram_by_groups_cwt(nwb_data, trials_df, region_channels, region_name,
                                      freq_range=(1, 50), n_freqs=100,
                                      groups=None, event_definitions=None, event_sequence=None):
    """
    Compute time-frequency maps using CWT (Continuous Wavelet Transform) with Morlet wavelets.

    Slower method with sample-by-sample time resolution and logarithmically-spaced frequencies.
    Uses padding to reduce edge artifacts and per-channel, per-frequency baseline z-scoring.

    Args:
        nwb_data: Dict with NWB file data
        trials_df: DataFrame with trial information
        region_channels: Dict mapping region names to channel lists
        region_name: Name of brain region
        freq_range: Tuple of (min_freq, max_freq) in Hz
        n_freqs: Number of logarithmically-spaced frequency bins
        groups: Optional list/tuple of index arrays, one per group. Each entry is an
            array-like of trial indices (0-based) into trials_df. If None or empty,
            all trials are treated as a single group.
        event_definitions: Dict mapping event names to config. If None, uses EVENT_DEFINITIONS.
        event_sequence: List of event names in order. If None, uses EVENT_SEQUENCE.

    Returns:
        dict with:
            - 'groups': List of per-group result dicts
            - 'meta': Shared metadata across groups
    """
    if event_definitions is None:
        event_definitions = EVENT_DEFINITIONS
    if event_sequence is None:
        event_sequence = EVENT_SEQUENCE

    lfp_series = nwb_data['lfp']['series']
    sampling_rate = nwb_data['lfp']['sampling_rate']
    channels = region_channels.get(region_name, [])

    if len(channels) == 0:
        return None

    # Validate channel indices
    n_channels_available = lfp_series.data.shape[1] if lfp_series.data.ndim > 1 else 1
    valid_channels = [ch for ch in channels if ch < n_channels_available]

    if len(valid_channels) == 0:
        print(f"    WARNING: No valid channels for {region_name} (requested {len(channels)}, file has {n_channels_available})")
        return None

    if len(valid_channels) < len(channels):
        print(f"    WARNING: Some channels out of bounds for {region_name} (using {len(valid_channels)}/{len(channels)} channels)")

    channels = valid_channels

    # CWT uses logarithmically-spaced frequencies
    freq_bins = np.logspace(
        np.log10(freq_range[0]),
        np.log10(freq_range[1]),
        n_freqs
    )
    print(f"    Using CWT with {n_freqs} logarithmically-spaced frequencies")

    # Compute padding to reduce edge artifacts
    min_freq = freq_range[0]
    padding_sec = max(0.5, 1.5 / min_freq)  # At least 0.5 second or 1.5 cycles
    print(f"    Using {padding_sec:.2f}s padding to reduce edge artifacts (min freq: {min_freq} Hz)")

    # Pre-compute event offsets and CWT parameters
    event_offsets = {}
    cumulative_offset = 0.0
    event_params = {}
    event_time_grid = {}
    event_boundaries = {}
    time_vectors_available = []

    for event_name in event_sequence:
        window_start, window_end = event_definitions[event_name]['window']
        duration_sec = window_end - window_start
        expected_samples = int(np.round(duration_sec * sampling_rate))

        event_offsets[event_name] = cumulative_offset
        global_start = cumulative_offset + window_start
        global_end = cumulative_offset + window_end
        event_boundaries[event_name] = (global_start, global_end)
        cumulative_offset += max(duration_sec, 0)

        if expected_samples <= 1:
            event_params[event_name] = {'expected_samples': expected_samples}
            event_time_grid[event_name] = np.array([])
            continue

        # CWT: sample-by-sample resolution
        t_tmp = np.arange(expected_samples) / sampling_rate
        event_time_grid[event_name] = global_start + t_tmp
        time_vectors_available.append(event_time_grid[event_name])

        event_params[event_name] = {'expected_samples': expected_samples}

    if len(time_vectors_available) == 0:
        return None

    global_time_vector = np.concatenate(time_vectors_available)
    print(f"    Computing CWT time-frequency maps for all trials (event-aligned)...")

    trial_spectrograms = []
    trial_event_times = []
    trial_positions = []

    for trial_idx, (_, trial) in enumerate(trials_df.iterrows()):
        trial_segments = extract_event_aligned_segments(
            lfp_series, trial, sampling_rate, channels, event_definitions, padding_sec
        )

        if trial_segments is None:
            continue

        segments = trial_segments['segments']
        response_latency = trial_segments['response_latency']
        padding_samples = trial_segments['padding_samples']

        event_matrices = []
        event_presence = {event: False for event in event_sequence}

        for event_name in event_sequence:
            params = event_params[event_name]
            expected_samples = params['expected_samples']
            time_grid = event_time_grid[event_name]

            if expected_samples <= 1 or len(time_grid) == 0:
                event_matrices.append(np.full((len(channels), len(freq_bins), 0), np.nan))
                continue

            segment = segments.get(event_name)
            if segment is None:
                event_matrices.append(np.full((len(channels), len(freq_bins), len(time_grid)), np.nan))
                continue

            # Compute CWT for each channel
            channel_power = []
            n_times_expected = len(time_grid)

            for ch_idx in range(segment.shape[0]):
                lfp_channel = segment[ch_idx, :]
                _, _, power = compute_cwt_morlet(
                    lfp_channel,
                    sampling_rate,
                    freq_range=freq_range,
                    n_freqs=n_freqs
                )

                # Trim padding
                if padding_samples > 0:
                    power = power[:, padding_samples:-padding_samples]

                # Ensure expected number of time points
                if power.shape[1] != n_times_expected:
                    diff = n_times_expected - power.shape[1]
                    if diff > 0:
                        pad_left = diff // 2
                        pad_right = diff - pad_left
                        power = np.pad(power, ((0, 0), (pad_left, pad_right)),
                                     constant_values=np.nan)
                    else:
                        trim = abs(diff) // 2
                        power = power[:, trim:trim + n_times_expected]

                channel_power.append(power)

            if len(channel_power) == 0:
                event_matrices.append(np.full((len(channels), len(freq_bins), len(time_grid)), np.nan))
                continue

            channel_power = np.array(channel_power)
            event_matrices.append(channel_power)
            event_presence[event_name] = True

        if len(event_matrices) == 0:
            continue

        trial_matrix = np.concatenate(event_matrices, axis=2)
        if np.all(np.isnan(trial_matrix)):
            continue

        trial_spectrograms.append(trial_matrix)
        trial_positions.append(trial_idx)

        # Record per-trial event times
        event_time_map = {event: (event_offsets[event] if event_presence[event] else np.nan)
                          for event in event_sequence}

        if not np.isnan(response_latency) and event_presence.get('probe', False):
            response_global = event_offsets['probe'] + response_latency
            probe_window_end = event_offsets['probe'] + event_definitions['probe']['window'][1]
            event_time_map['response'] = response_global if response_global <= probe_window_end else np.nan
        else:
            event_time_map['response'] = np.nan

        trial_event_times.append(event_time_map)

    if len(trial_spectrograms) == 0:
        return None

    spectrogram_array = np.stack(trial_spectrograms)

    # Baseline normalization
    baseline_mask = (global_time_vector >= -1.0) & (global_time_vector < 0.0)
    if not np.any(baseline_mask):
        raise ValueError("Baseline window [-1, 0) is empty")

    baseline_data = spectrogram_array[:, :, :, baseline_mask]

    with np.errstate(invalid='ignore'):
        pooled_baseline_mean = np.nanmean(baseline_data, axis=(0, 3))
        pooled_baseline_std = np.nanstd(baseline_data, axis=(0, 3))

    pooled_baseline_std[pooled_baseline_std < 1e-12] = 1.0

    baseline_mean_expanded = pooled_baseline_mean[np.newaxis, :, :, np.newaxis]
    baseline_std_expanded = pooled_baseline_std[np.newaxis, :, :, np.newaxis]

    spectrogram_z = (spectrogram_array - baseline_mean_expanded) / baseline_std_expanded

    n_trials_total = spectrogram_array.shape[0]
    trial_positions = np.array(trial_positions)

    # Map groups
    if groups is None or len(groups) == 0:
        group_masks = [np.ones(n_trials_total, dtype=bool)]
    else:
        group_masks = []
        for g in groups:
            group_trial_indices = np.asarray(g, dtype=int)
            mask = np.isin(trial_positions, group_trial_indices)
            group_masks.append(mask)

    results_groups = []
    event_keys = list(trial_event_times[0].keys()) if len(trial_event_times) > 0 else []

    for mask in group_masks:
        if not np.any(mask):
            continue

        group_power_linear = spectrogram_array[mask]
        group_power_zscored = spectrogram_z[mask]

        with np.errstate(invalid='ignore'):
            # Average trials for each channel separately (keep channel dimension)
            # Shape: (n_trials, n_channels, n_freqs, n_times) -> (n_channels, n_freqs, n_times)
            power_linear_per_channel = np.nanmean(group_power_linear, axis=0)
            power_zscored_per_channel = np.nanmean(group_power_zscored, axis=0)

            # For plotting, compute region-averaged versions
            power_linear_mean = np.nanmean(power_linear_per_channel, axis=0)  # (n_freqs, n_times)
            power_zscored = np.nanmean(power_zscored_per_channel, axis=0)

            power_mean = 10 * np.log10(power_linear_mean + 1e-12)

        avg_event_times = {}
        for key in event_keys:
            values = [trial_event_times[i][key] for i in range(len(trial_event_times))
                      if mask[i] and not np.isnan(trial_event_times[i][key])]
            avg_event_times[key] = np.mean(values) if len(values) > 0 else np.nan

        original_indices = trial_positions[mask]

        results_groups.append({
            'time_vector': global_time_vector,
            'frequencies': freq_bins,
            'power_mean': power_mean,
            'power_zscored': power_zscored,
            'power_zscored_per_channel': power_zscored_per_channel,  # (n_channels, n_freqs, n_times)
            'n_trials': int(np.sum(mask)),
            'n_channels': power_zscored_per_channel.shape[0],
            'event_times': avg_event_times,
            'event_boundaries': {k: tuple(v) for k, v in event_boundaries.items()},
            'event_sequence': list(event_sequence),
            'region': region_name,
            'pooled_baseline_mean': pooled_baseline_mean,
            'pooled_baseline_std': pooled_baseline_std,
            'trial_indices': original_indices
        })

    return {
        'groups': results_groups,
        'meta': {
            'event_boundaries': {k: tuple(v) for k, v in event_boundaries.items()},
            'event_sequence': list(event_sequence),
            'region': region_name,
            'pooled_baseline_mean': pooled_baseline_mean,
            'pooled_baseline_std': pooled_baseline_std
        }
    }


def plot_spectrogram_by_groups(
    spectrogram_results,
    figsize=None,
    group_labels=None,
    title_suffix='by Groups',
    group_order=None,
    data_type='zscored',
    event_sequence=None
):
    """
    Plot time-frequency spectrograms for trial groupings with separate panels per event window.

    Parameters
    ----------
    spectrogram_results : dict
        Output of compute_spectrogram_by_groups with 'groups' and 'meta' keys.
    figsize : tuple, optional
        Matplotlib figure size. Defaults to (#events * 4, #groups * 2.5) if omitted.
    group_labels : tuple/list of str, optional
        Labels to use for each group row. Defaults to ['Group 1', 'Group 2', ...].
    title_suffix : str, optional
        Text appended after "Spectrograms" in the figure title. Defaults to 'by Groups'.
    group_order : tuple/list of int, optional
        Indices into groups list that determine the row order (default: [0, 1, 2, ...]).
    data_type : str, optional
        Type of data to plot. Must be 'zscored' or 'raw'. Defaults to 'zscored'.
        - 'zscored': Plot baseline z-scored power (symmetric colormap, centered at 0)
        - 'raw': Plot raw mean power in dB (sequential colormap, from 0)
    event_sequence : list, optional
        List of event names in order. If None, uses EVENT_SEQUENCE from config or from meta.
    """
    if event_sequence is None:
        event_sequence = EVENT_SEQUENCE
    
    if 'groups' not in spectrogram_results:
        raise ValueError("spectrogram_results must contain 'groups' key (from compute_spectrogram_by_groups)")
    
    # Validate data_type parameter
    if data_type not in ['zscored', 'raw']:
        raise ValueError("data_type must be 'zscored' or 'raw'")
    
    groups = spectrogram_results['groups']
    meta = spectrogram_results.get('meta', {})
    
    if len(groups) == 0:
        print("No groups found in spectrogram_results")
        return None
    
    n_groups = len(groups)
    
    # Default group labels
    if group_labels is None:
        group_labels = [f'Group {i+1}' for i in range(n_groups)]
    elif not isinstance(group_labels, (tuple, list)) or len(group_labels) != n_groups:
        raise ValueError(f"group_labels must contain exactly {n_groups} entries (one per group).")
    
    # Default group order
    if group_order is None:
        group_order = list(range(n_groups))
    elif not isinstance(group_order, (tuple, list)) or len(group_order) != n_groups:
        raise ValueError(f"group_order must contain exactly {n_groups} entries (indices 0-{n_groups-1}).")
    
    # Validate group_order indices
    if not all(isinstance(idx, int) and 0 <= idx < n_groups for idx in group_order):
        raise ValueError(f"group_order must contain valid indices in range [0, {n_groups-1})")
    
    if len(set(group_order)) != n_groups:
        raise ValueError("group_order must contain unique indices")
    
    # Reorder groups and labels based on group_order
    ordered_groups = [groups[i] for i in group_order]
    ordered_labels = [group_labels[i] for i in group_order]
    
    # Extract shared metadata
    events = meta.get('event_sequence', event_sequence)
    event_bounds = meta.get('event_boundaries', {})
    
    if len(events) == 0:
        # Fallback: try to get from first group
        events = groups[0].get('event_sequence', event_sequence)
        event_bounds = groups[0].get('event_boundaries', {})

    if figsize is None:
        figsize = (4.0 * max(1, len(events)), 2.5 * n_groups)
    
    # Get frequency range from first group (all groups should share same frequencies)
    freq_vals = ordered_groups[0]['frequencies']
    freq_min, freq_max = freq_vals[0], freq_vals[-1]
    
    # Determine data key and colormap based on data_type
    if data_type == 'zscored':
        data_key = 'power_zscored'
        cmap = 'RdBu_r'
        symmetric = True
    else:  # data_type == 'raw'
        data_key = 'power_mean'
        cmap = 'viridis'
        symmetric = False
    
    def _collect_vmax(group_dict):
        vmax_local = 0.0
        for event in events:
            bounds = event_bounds.get(event)
            if bounds is None:
                continue
            start, end = bounds
            mask = (group_dict['time_vector'] >= start) & (group_dict['time_vector'] <= end)
            if not np.any(mask):
                continue
            data_slice = group_dict[data_key][:, mask]
            if data_slice.size == 0:
                continue
            if symmetric:
                vmax_local = max(vmax_local, np.nanmax(np.abs(data_slice)))
            else:
                vmax_local = max(vmax_local, np.nanmax(data_slice))
        return vmax_local
    
    # Compute vmax across all groups
    vmax = max(_collect_vmax(group) for group in ordered_groups)
    if not np.isfinite(vmax) or vmax == 0:
        vmax = 1.0
    
    # Set vmin based on data type
    if symmetric:
        vmin = -vmax
    else:
        vmin = None  # Let matplotlib auto-scale from 0
    
    # Create figure with dynamic number of rows
    fig, axes = plt.subplots(
        n_groups, len(events),
        figsize=figsize,
        sharey=True,
        gridspec_kw={'wspace': 0.05, 'hspace': 0.25}
    )
    
    # Ensure axes is always 2D array for consistent indexing
    if n_groups == 1 and len(events) == 1:
        axes = np.array([[axes]])
    elif n_groups == 1:
        axes = axes.reshape(1, -1)
    elif len(events) == 1:
        axes = axes.reshape(-1, 1)
    
    colorbar_handle = None

    # Get event definitions to extract window bounds
    from .config import EVENT_DEFINITIONS

    for row_idx, (row_label, group_dict) in enumerate(zip(ordered_labels, ordered_groups)):
        ax_row = axes[row_idx]

        for col_idx, event in enumerate(events):
            ax = ax_row[col_idx]
            bounds = event_bounds.get(event)

            if bounds is None:
                ax.axis('off')
                continue

            # Global time bounds for data slicing
            global_start, global_end = bounds
            mask = (group_dict['time_vector'] >= global_start) & (group_dict['time_vector'] <= global_end)
            if not np.any(mask):
                ax.axis('off')
                continue

            data_slice = group_dict[data_key][:, mask]
            if data_slice.size == 0:
                ax.axis('off')
                continue

            # Get event-relative window from config
            event_config = EVENT_DEFINITIONS.get(event, {})
            event_window = event_config.get('window', (0, 0))
            event_relative_start, event_relative_end = event_window

            data_masked = np.ma.masked_invalid(data_slice)
            im = ax.imshow(
                data_masked,
                aspect='auto',
                origin='lower',
                cmap=cmap,
                extent=[event_relative_start, event_relative_end, freq_min, freq_max],
                interpolation='nearest',
                vmin=vmin,
                vmax=vmax
            )

            # Event onset marker at time 0 (relative to event)
            event_time = group_dict['event_times'].get(event, np.nan)
            if np.isfinite(event_time):
                # Event onset is always at 0 in event-relative time
                ax.axvline(0, color='black', linestyle='--', linewidth=1.5, alpha=0.7)

            # Response time marker (for probe epoch only)
            response_time = group_dict['event_times'].get('response', np.nan)
            if event == 'probe' and np.isfinite(response_time):
                # Convert response time to event-relative coordinates
                probe_onset = group_dict['event_times'].get('probe', np.nan)
                if np.isfinite(probe_onset):
                    response_relative = response_time - probe_onset
                    if event_relative_start <= response_relative <= event_relative_end:
                        ax.axvline(response_relative, color='gold', linestyle='-', linewidth=2.0, alpha=0.8)

            if col_idx == 0:
                ax.set_ylabel(f"{row_label}\nFrequency (Hz)", fontsize=11)
            else:
                ax.set_ylabel('')

            if row_idx == n_groups - 1:
                ax.set_xlabel('Time (s)', fontsize=11)
            else:
                ax.set_xlabel('')

            if row_idx == 0:
                ax.set_title(f"{event}", fontsize=12, fontweight='bold')

            for freq_line in [4, 8, 12, 30]:
                ax.axhline(freq_line, color='gray', linestyle=':', alpha=0.2, linewidth=1)

            ax.set_xlim(event_relative_start, event_relative_end)
            ax.grid(False)

            if colorbar_handle is None:
                colorbar_handle = im

    if colorbar_handle is not None:
        cbar = fig.colorbar(colorbar_handle, ax=axes.ravel().tolist(), fraction=0.02, pad=0.02)
        if data_type == 'zscored':
            cbar_label = 'Z-scored Power'
        else:  # data_type == 'raw'
            cbar_label = 'Power (dB)'
        cbar.set_label(cbar_label, fontsize=11)

    region_name = meta.get('region', 'Unknown Region')
    title_suffix_clean = title_suffix.strip()
    title_text = f"{region_name} Spectrograms"
    if title_suffix_clean:
        title_text = f"{title_text} {title_suffix_clean}"
    fig.suptitle(title_text, fontsize=16, fontweight='bold')

    return fig

