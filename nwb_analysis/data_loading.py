"""
NWB file loading and LFP data extraction functions.
"""

from pathlib import Path
from pynwb import NWBHDF5IO


def load_lfp_safe(lfp_series, start_idx=0, end_idx=None):
    """
    Load LFP data handling potential transposition.
    Returns data in (timepoints, channels) format.
    
    Args:
        lfp_series: ElectricalSeries object from NWB
        start_idx: Start timepoint index
        end_idx: End timepoint index (None = all)
    
    Returns:
        lfp_data: numpy array of shape (timepoints, channels)
    """
    # Load data using standard slicing
    if end_idx is None:
        lfp_data = lfp_series.data[start_idx:]
    else:
        lfp_data = lfp_series.data[start_idx:end_idx]
    
    # Ensure shape is (timepoints, channels)
    if lfp_data.ndim == 1:
        lfp_data = lfp_data.reshape(-1, 1)
    
    return lfp_data


def load_nwb_file(filepath):
    """
    Load NWB file and extract key data structures.
    
    Returns:
        dict with keys: 'io', 'nwbfile', 'lfp', 'electrodes', 'trials'
    """
    io = NWBHDF5IO(str(filepath), 'r')
    nwbfile = io.read()
    
    data = {
        'io': io,
        'nwbfile': nwbfile,
        'filepath': filepath
    }
    
    # Extract LFP metadata
    if 'LFPs' in nwbfile.acquisition:
        lfp_series = nwbfile.acquisition['LFPs']
        data['lfp'] = {
            'series': lfp_series,
            'sampling_rate': lfp_series.rate,
            'n_channels': lfp_series.data.shape[1] if lfp_series.data.ndim > 1 else 1,
            'n_samples': lfp_series.data.shape[0]
        }
    
    # Extract electrodes metadata
    if nwbfile.electrodes is not None:
        data['electrodes'] = nwbfile.electrodes.to_dataframe()
    
    # Extract trials
    if nwbfile.trials is not None:
        data['trials'] = nwbfile.trials.to_dataframe()
    
    return data


def get_subject_files(data_dir, subject_id=None):
    """
    Get all NWB files, optionally filtered by subject.
    
    Args:
        data_dir: Path to data directory
        subject_id: Optional subject ID (e.g., 1, 2, 3) to filter by
    
    Returns:
        files: Sorted list of NWB file paths
    """
    if subject_id:
        pattern = f"sub-{subject_id}/sub-{subject_id}_*.nwb"
    else:
        pattern = "sub-*/sub-*_*.nwb"
    
    files = sorted(data_dir.glob(pattern))
    return files

