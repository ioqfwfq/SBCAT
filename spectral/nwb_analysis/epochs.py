"""
Task epoch extraction and summary functions.
"""

import numpy as np
import pandas as pd


def extract_epoch_times(trials_df):
    """
    Extract start and end times for each task epoch from trials table.
    
    Args:
        trials_df: DataFrame with trial information
    
    Returns:
        epochs: DataFrame with columns:
            - trial_id: trial index
            - epoch_type: 'fixation', 'encoding1', etc.
            - start_time: epoch start in seconds
            - end_time: epoch end in seconds
            - duration: epoch duration
            - load: memory load for this trial
    """
    epochs = []
    
    for trial_idx, trial in trials_df.iterrows():
        trial_load = trial.get('loads', np.nan)
        
        # Fixation epoch
        if 'timestamps_FixationCross' in trial and 'timestamps_Encoding1' in trial:
            start = trial['timestamps_FixationCross']
            end = trial['timestamps_Encoding1']
            if start > 0 and end > start:
                epochs.append({
                    'trial_id': trial_idx,
                    'epoch_type': 'fixation',
                    'start_time': start,
                    'end_time': end,
                    'duration': end - start,
                    'load': trial_load
                })
        
        # Encoding epochs (1, 2, 3)
        for enc_num in [1, 2, 3]:
            start_col = f'timestamps_Encoding{enc_num}'
            end_col = f'timestamps_Encoding{enc_num}_end'
            
            if start_col in trial and end_col in trial:
                start = trial[start_col]
                end = trial[end_col]
                
                # Only include if valid (some trials have load=1, so only encoding1 exists)
                if start > 0 and end > start:
                    epochs.append({
                        'trial_id': trial_idx,
                        'epoch_type': f'encoding{enc_num}',
                        'start_time': start,
                        'end_time': end,
                        'duration': end - start,
                        'load': trial_load
                    })
        
        # Maintenance epoch (delay period)
        if 'timestamps_Maintenance' in trial and 'timestamps_Probe' in trial:
            start = trial['timestamps_Maintenance']
            end = trial['timestamps_Probe']
            if start > 0 and end > start:
                epochs.append({
                    'trial_id': trial_idx,
                    'epoch_type': 'maintenance',
                    'start_time': start,
                    'end_time': end,
                    'duration': end - start,
                    'load': trial_load
                })
        
        # Probe epoch (fixed 1-second window: -500ms to +500ms centered at probe onset)
        if 'timestamps_Probe' in trial:
            probe_time = trial['timestamps_Probe']
            if probe_time > 0:
                start = probe_time - 0.5  # -500ms before probe
                end = probe_time + 0.5    # +500ms after probe
                epochs.append({
                    'trial_id': trial_idx,
                    'epoch_type': 'probe',
                    'start_time': start,
                    'end_time': end,
                    'duration': 1.0,  # Always 1 second
                    'load': trial_load
                })
        
        # RT epoch (reaction time: probe onset to response)
        if 'timestamps_Probe' in trial and 'timestamps_Response' in trial:
            start = trial['timestamps_Probe']
            end = trial['timestamps_Response']
            if start > 0 and end > start:
                epochs.append({
                    'trial_id': trial_idx,
                    'epoch_type': 'RT',
                    'start_time': start,
                    'end_time': end,
                    'duration': end - start,
                    'load': trial_load
                })
    
    return pd.DataFrame(epochs)


def summarize_epochs(epochs_df):
    """
    Print summary statistics of extracted epochs.
    
    Args:
        epochs_df: DataFrame with epoch information (from extract_epoch_times)
    """
    print("=== Task Epoch Summary ===")
    print(f"Total epochs extracted: {len(epochs_df)}\n")
    
    # Count by epoch type
    epoch_counts = epochs_df['epoch_type'].value_counts()
    print("Epochs by type:")
    for epoch_type, count in epoch_counts.items():
        print(f"  {epoch_type}: {count}")
    
    # Duration statistics
    print("\nEpoch durations (mean ± std):")
    for epoch_type in epochs_df['epoch_type'].unique():
        durations = epochs_df[epochs_df['epoch_type'] == epoch_type]['duration']
        print(f"  {epoch_type}: {durations.mean():.3f} ± {durations.std():.3f} s")
    
    # Memory load distribution
    if 'load' in epochs_df.columns:
        print("\nMemory load distribution:")
        load_counts = epochs_df['load'].value_counts().sort_index()
        for load, count in load_counts.items():
            print(f"  Load {int(load)}: {count} epochs")


def group_trials_by_load(trials_df):
    """
    Group trials by memory load condition.

    This function organizes trial indices into groups based on the 'loads' column
    in the trials DataFrame. It's useful for comparing neural activity across
    different working memory load conditions.

    Parameters:
    -----------
    trials_df : pd.DataFrame
        Trials table with 'loads' column indicating memory load per trial

    Returns:
    --------
    dict : Dictionary with the following keys:
        - 'groups': list of np.array
            Trial indices (0-based row positions) for each load condition
        - 'load_values': list of int/float
            Unique load values in sorted order (e.g., [1, 3])
        - 'group_labels': list of str
            Descriptive labels for each group (e.g., ['Load 1', 'Load 3'])
        - 'group_counts': list of int
            Number of trials in each group
        - 'summary': str
            Formatted summary string for printing

    Example:
    --------
    >>> load_groups = group_trials_by_load(trials_df)
    >>> print(load_groups['summary'])
    >>> # Use groups for analysis
    >>> for group_idx, group_label in zip(load_groups['groups'],
    ...                                     load_groups['group_labels']):
    ...     print(f"{group_label}: {len(group_idx)} trials")
    """
    if 'loads' not in trials_df.columns:
        raise ValueError("trials_df must have a 'loads' column")

    # Get unique load values (sorted)
    load_values = sorted(trials_df['loads'].dropna().unique())

    if len(load_values) == 0:
        raise ValueError("No valid load values found in trials_df")

    # Group trial indices by load
    groups = []
    group_labels = []
    group_counts = []

    for load_val in load_values:
        # Get 0-based indices of trials with this load
        trial_indices = np.where(trials_df['loads'] == load_val)[0]
        groups.append(trial_indices)

        # Create label
        if load_val == int(load_val):  # If it's a whole number
            label = f"Load {int(load_val)}"
        else:
            label = f"Load {load_val}"
        group_labels.append(label)

        # Count
        group_counts.append(len(trial_indices))

    # Create summary string
    summary_lines = ["=== Trial Grouping by Memory Load ==="]
    summary_lines.append(f"Total trials: {len(trials_df)}")
    summary_lines.append(f"Number of groups: {len(groups)}")
    summary_lines.append("")

    for load_val, label, count in zip(load_values, group_labels, group_counts):
        summary_lines.append(f"  {label}: {count} trials")

    summary = "\n".join(summary_lines)

    return {
        'groups': groups,
        'load_values': load_values,
        'group_labels': group_labels,
        'group_counts': group_counts,
        'summary': summary
    }

