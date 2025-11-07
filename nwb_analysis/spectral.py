"""
Power spectrum and PSD computation functions.
"""

import numpy as np
from scipy import stats
from scipy.signal import welch


def extract_lfp_epoch_segment(lfp_series, start_time, end_time, sampling_rate, channel_idx):
    """
    Extract LFP segment for a specific epoch and channel.
    
    Args:
        lfp_series: ElectricalSeries object
        start_time: Epoch start time in seconds
        end_time: Epoch end time in seconds
        sampling_rate: LFP sampling rate in Hz
        channel_idx: Channel index to extract
    
    Returns:
        lfp_segment: 1D array of LFP values for this epoch
    """
    # Convert times to sample indices
    start_idx = int(start_time * sampling_rate)
    end_idx = int(end_time * sampling_rate)
    
    # Ensure indices are within bounds
    start_idx = max(0, start_idx)
    end_idx = min(lfp_series.data.shape[0], end_idx)
    
    # Extract segment
    if lfp_series.data.ndim > 1:
        lfp_segment = lfp_series.data[start_idx:end_idx, channel_idx]
    else:
        lfp_segment = lfp_series.data[start_idx:end_idx]
    
    return lfp_segment


def compute_psd_welch(lfp_segment, sampling_rate, nperseg_sec=2.0, overlap=0.5):
    """
    Compute power spectral density using Welch's method.
    
    Args:
        lfp_segment: 1D array of LFP values
        sampling_rate: Sampling rate in Hz
        nperseg_sec: Window length in seconds for Welch's method
        overlap: Overlap fraction (0-1)
    
    Returns:
        freqs: Array of frequencies
        psd: Power spectral density
    """
    # Calculate nperseg in samples
    nperseg = int(nperseg_sec * sampling_rate)
    noverlap = int(nperseg * overlap)
    
    # Compute PSD using Welch's method
    freqs, psd = welch(
        lfp_segment,
        fs=sampling_rate,
        nperseg=nperseg,
        noverlap=noverlap,
        scaling='density'
    )
    
    return freqs, psd


def compute_band_power(freqs, psd, freq_band):
    """
    Compute average power in a specific frequency band.
    
    Args:
        freqs: Array of frequencies
        psd: Power spectral density
        freq_band: Tuple of (low_freq, high_freq)
    
    Returns:
        band_power: Mean power in the specified band
    """
    # Find indices for the frequency band
    band_mask = (freqs >= freq_band[0]) & (freqs <= freq_band[1])
    
    if not np.any(band_mask):
        return np.nan
    
    # Compute mean power in band
    band_power = np.mean(psd[band_mask])
    
    return band_power


def fit_1f_background(freqs, psd, freq_range=(1, 150)):
    """
    Fit 1/f aperiodic component to power spectrum.
    Uses log-log linear fit: log(P) = a - b*log(f)
    
    Args:
        freqs: Array of frequencies
        psd: Power spectral density
        freq_range: Tuple of (min_freq, max_freq) for fitting
    
    Returns:
        dict with 'aperiodic_fit', 'residual', 'slope', 'intercept'
    """
    freq_mask = (freqs >= freq_range[0]) & (freqs <= freq_range[1])
    freqs_fit = freqs[freq_mask]
    power_fit = psd[freq_mask]
    
    # Avoid log of zero
    freqs_fit = freqs_fit[power_fit > 0]
    power_fit = power_fit[power_fit > 0]
    
    if len(freqs_fit) < 2:
        return {
            'aperiodic_fit': np.zeros_like(psd),
            'residual': psd,
            'slope': np.nan,
            'intercept': np.nan
        }
    
    # Log-log linear regression
    log_freqs = np.log10(freqs_fit)
    log_power = np.log10(power_fit)
    
    coeffs = np.polyfit(log_freqs, log_power, 1)
    aperiodic_log = np.polyval(coeffs, np.log10(freqs))
    aperiodic_fit = 10 ** aperiodic_log
    
    residual = psd - aperiodic_fit
    
    return {
        'aperiodic_fit': aperiodic_fit,
        'residual': residual,
        'slope': coeffs[0],
        'intercept': coeffs[1]
    }


def compute_epoch_psd_for_region(nwb_data, epochs_df, region_channels, region_name, 
                                  epoch_type, nperseg_sec=2.0):
    """
    Compute average PSD for all epochs of a specific type in a brain region.
    
    Args:
        nwb_data: Dict with NWB file data (from load_nwb_file)
        epochs_df: DataFrame with epoch timing information
        region_channels: Dict mapping region names to channel lists
        region_name: Name of brain region
        epoch_type: Type of epoch (e.g., 'maintenance', 'encoding1')
        nperseg_sec: Window length for Welch's method
    
    Returns:
        result: Dict with 'freqs', 'psd_mean', 'psd_sem', 'n_epochs', 'n_channels'
    """
    lfp_series = nwb_data['lfp']['series']
    sampling_rate = nwb_data['lfp']['sampling_rate']
    channels = region_channels.get(region_name, [])
    
    if len(channels) == 0:
        return None
    
    # Filter epochs by type
    epochs_filtered = epochs_df[epochs_df['epoch_type'] == epoch_type]
    
    if len(epochs_filtered) == 0:
        return None
    
    # Collect all PSDs
    all_psds = []
    freqs = None
    
    for _, epoch in epochs_filtered.iterrows():
        for channel_idx in channels:
            try:
                # Extract LFP segment
                lfp_segment = extract_lfp_epoch_segment(
                    lfp_series,
                    epoch['start_time'],
                    epoch['end_time'],
                    sampling_rate,
                    channel_idx
                )
                
                # Skip if segment too short
                if len(lfp_segment) < int(nperseg_sec * sampling_rate):
                    continue
                
                # Compute PSD
                freqs, psd = compute_psd_welch(lfp_segment, sampling_rate, nperseg_sec)
                all_psds.append(psd)
                
            except Exception as e:
                # Skip problematic segments
                continue
    
    if len(all_psds) == 0:
        return None
    
    # Compute mean and SEM across all epochs and channels
    all_psds = np.array(all_psds)
    psd_mean = np.mean(all_psds, axis=0)
    psd_sem = stats.sem(all_psds, axis=0)
    
    return {
        'freqs': freqs,
        'psd_mean': psd_mean,
        'psd_sem': psd_sem,
        'n_epochs': len(epochs_filtered),
        'n_channels': len(channels),
        'n_samples': len(all_psds)
    }


def compute_all_region_epoch_psds(nwb_data, epochs_df, region_channels, 
                                   epoch_types=None, region_names=None):
    """
    Compute PSDs for all combinations of regions and epochs.
    
    Args:
        nwb_data: Dict with NWB file data
        epochs_df: DataFrame with epoch timing information
        region_channels: Dict mapping region names to channel lists
        epoch_types: Optional list of epoch types to analyze (default: all in epochs_df)
        region_names: Optional list of region names to analyze (default: all in region_channels)
    
    Returns:
        results: Nested dict {region_name: {epoch_type: psd_result}}
    """
    if epoch_types is None:
        epoch_types = epochs_df['epoch_type'].unique()
    
    if region_names is None:
        region_names = region_channels.keys()
    
    results = {}
    
    print("Computing PSDs for all region × epoch combinations...")
    total_combos = len(region_names) * len(epoch_types)
    completed = 0
    
    for region_name in region_names:
        results[region_name] = {}
        
        for epoch_type in epoch_types:
            result = compute_epoch_psd_for_region(
                nwb_data, epochs_df, region_channels, 
                region_name, epoch_type
            )
            
            results[region_name][epoch_type] = result
            completed += 1
            
            if result is not None:
                print(f"  [{completed}/{total_combos}] {region_name} × {epoch_type}: "
                      f"{result['n_samples']} samples")
            else:
                print(f"  [{completed}/{total_combos}] {region_name} × {epoch_type}: No data")
    
    return results

