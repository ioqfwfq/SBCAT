"""
Brain region extraction and mapping functions.
"""

from .config import BRAIN_REGIONS


def extract_region_channels(electrodes_df, region_mapping=None):
    """
    Map electrodes to brain regions based on location information.
    
    Args:
        electrodes_df: DataFrame with electrode metadata
        region_mapping: Dict mapping region names to location strings.
                       If None, uses BRAIN_REGIONS from config.
    
    Returns:
        region_channels: Dict {region_name: list of channel indices}
        channel_to_region: Dict {channel_idx: region_name}
    """
    if region_mapping is None:
        region_mapping = BRAIN_REGIONS
    
    region_channels = {region: [] for region in region_mapping.keys()}
    channel_to_region = {}
    
    if 'location' not in electrodes_df.columns:
        print("WARNING: No 'location' column found in electrodes table")
        return region_channels, channel_to_region
    
    for idx, row in electrodes_df.iterrows():
        location = row['location']
        
        # Find which region this electrode belongs to
        for region_name, location_strings in region_mapping.items():
            if location in location_strings:
                region_channels[region_name].append(idx)
                channel_to_region[idx] = region_name
                break
    
    return region_channels, channel_to_region


def summarize_region_coverage(region_channels, electrodes_df):
    """
    Print summary of channel coverage by brain region.
    
    Args:
        region_channels: Dict mapping region names to channel lists
        electrodes_df: DataFrame with electrode metadata
    """
    print("=== Brain Region Channel Coverage ===")
    print(f"Total electrodes: {len(electrodes_df)}\n")
    
    for region_name, channels in region_channels.items():
        if len(channels) > 0:
            print(f"{region_name}: {len(channels)} channels")
            if 'location' in electrodes_df.columns:
                locations = electrodes_df.loc[channels, 'location'].value_counts()
                for loc, count in locations.items():
                    print(f"  - {loc}: {count}")
        else:
            print(f"{region_name}: No channels")
    
    # Channels not assigned to any region
    assigned_channels = sum([len(chs) for chs in region_channels.values()])
    unassigned = len(electrodes_df) - assigned_channels
    if unassigned > 0:
        print(f"\nUnassigned channels: {unassigned}")

