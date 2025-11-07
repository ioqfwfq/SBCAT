"""
NWB Analysis Package

A package for analyzing NWB (Neurodata Without Borders) files, including:
- Data loading and LFP extraction
- Brain region mapping
- Task epoch extraction
- Power spectrum and PSD computation
- Time-frequency spectrogram analysis
"""

# Import main functions for easy access
from .data_loading import (
    load_lfp_safe,
    load_nwb_file,
    get_subject_files
)

from .regions import (
    extract_region_channels,
    summarize_region_coverage
)

from .epochs import (
    extract_epoch_times,
    summarize_epochs,
    group_trials_by_load
)

from .spectral import (
    extract_lfp_epoch_segment,
    compute_psd_welch,
    compute_band_power,
    fit_1f_background,
    compute_epoch_psd_for_region,
    compute_all_region_epoch_psds
)

from .spectrogram import (
    extract_event_aligned_segments,
    compute_spectrogram_by_groups_stft,
    compute_spectrogram_by_groups_cwt,
    plot_spectrogram_by_groups
)

# Import configuration constants
from .config import (
    FREQ_BANDS,
    BRAIN_REGIONS,
    EVENT_DEFINITIONS,
    EVENT_SEQUENCE,
    TASK_EPOCHS,
    WELCH_WINDOW_SEC,
    WELCH_OVERLAP,
    FREQ_RANGE
)

__all__ = [
    # Data loading
    'load_lfp_safe',
    'load_nwb_file',
    'get_subject_files',
    # Regions
    'extract_region_channels',
    'summarize_region_coverage',
    # Epochs
    'extract_epoch_times',
    'summarize_epochs',
    'group_trials_by_load',
    # Spectral
    'extract_lfp_epoch_segment',
    'compute_psd_welch',
    'compute_band_power',
    'fit_1f_background',
    'compute_epoch_psd_for_region',
    'compute_all_region_epoch_psds',
    # Spectrogram
    'extract_event_aligned_segments',
    'compute_spectrogram_by_groups_stft',
    'compute_spectrogram_by_groups_cwt',
    'plot_spectrogram_by_groups',
    # Config
    'FREQ_BANDS',
    'BRAIN_REGIONS',
    'EVENT_DEFINITIONS',
    'EVENT_SEQUENCE',
    'TASK_EPOCHS',
    'WELCH_WINDOW_SEC',
    'WELCH_OVERLAP',
    'FREQ_RANGE',
]

