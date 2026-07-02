"""
Configuration constants for NWB analysis.

This module contains all configuration parameters including frequency bands,
brain region mappings, event definitions, and spectral analysis parameters.
"""

# Frequency band definitions
FREQ_BANDS = {
    'delta': (1, 4),
    'theta': (4, 8),
    'alpha': (8, 12),
    'beta': (12, 30),
    'low_gamma': (30, 60),
    'high_gamma': (60, 120)
}

# Spectral analysis parameters
WELCH_WINDOW_SEC = 2.0  # 2-second windows for Welch's method
WELCH_OVERLAP = 0.5     # 50% overlap
FREQ_RANGE = (0.5, 150) # Frequency range for analysis

# Brain regions of interest - MERGED LEFT/RIGHT HEMISPHERES
BRAIN_REGIONS = {
    'Hippocampus': ['hippocampus_left', 'hippocampus_right'],
    'Amygdala': ['amygdala_left', 'amygdala_right'],
    'vmPFC': ['ventral_medial_prefrontal_cortex_left', 'ventral_medial_prefrontal_cortex_right'],
    'dACC': ['dorsal_anterior_cingulate_cortex_left', 'dorsal_anterior_cingulate_cortex_right'],
    'preSMA': ['pre_supplementary_motor_area_left', 'pre_supplementary_motor_area_right']
}

# Task epoch definitions (will be extracted from trials table)
TASK_EPOCHS = ['fixation', 'encoding1', 'encoding2', 'encoding3', 'maintenance', 'probe', 'RT']

# Event-aligned spectrogram windows (seconds relative to event onset)
EVENT_DEFINITIONS = {
    'encoding1': {'timestamp': 'timestamps_Encoding1', 'window': (-1.0, 2.0)},
    'encoding2': {'timestamp': 'timestamps_Encoding2', 'window': (0.0, 2.0)},
    'encoding3': {'timestamp': 'timestamps_Encoding3', 'window': (0.0, 2.0)},
    'maintenance': {'timestamp': 'timestamps_Maintenance', 'window': (0.0, 2.5)},
    'probe': {'timestamp': 'timestamps_Probe', 'window': (0.0, 2.0)}
}
EVENT_SEQUENCE = list(EVENT_DEFINITIONS.keys())

