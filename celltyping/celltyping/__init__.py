"""celltyping — putative cell-type classification for human single-neuron WM datasets.

Task-agnostic toolkit built for the working-memory cell-typing project. It reads
DANDI NWB files (000673 Sternberg-category; 000469 Sternberg) and produces, per unit:

  * waveform-shape features   (trough-to-peak, FWHM, repolarization, amplitude ...)
  * spike-train / ACG features (rate, ISI CV/CV2, local variation, burst, refractory ...)

from which cells are split into narrow- vs broad-spiking groups, verified as putative
fast-spiking inhibitory interneurons via the autocorrelogram, and clustered with
WaveMAP. The single deliverable is a per-unit LABEL table (celltyping.labels).

Pipeline
--------
    features.build_unit_features   NWB -> per-unit feature table (+ mean waveforms)
    classify.assign_narrow_broad   trough-to-peak GMM antimode split
    classify.interneuron_verification   narrow-vs-broad ISI/ACG test
    wavemap.run_wavemap            UMAP graph -> Louvain communities
    labels.build_label_table       everything -> the per-unit label table

Design notes
------------
* Handles BOTH NWB waveform layouts: a precomputed 1-D units['waveform_mean']
  (WM-binding) and raw per-spike units['waveforms'] (n_ch,n_spikes,n_samp) (000673).
* Waveform sampling defaults to the Rutishauser/OSort convention (100 kHz, 256
  samples); pass fs_hz to override. trough_to_peak in ~0.2-1.0 ms confirms fs_hz.
"""

from .features import (
    build_unit_features,
    mean_waveform,
    polarity_normalized_waveform,
    waveform_shape_features,
    spike_train_features,
    autocorrelogram,
    OSORT_FS_HZ,
)
from .classify import assign_narrow_broad, interneuron_verification
from .wavemap import (
    preprocess_waveforms,
    run_wavemap,
    cluster_waveform_importance,
    characterize_clusters,
    check_dataset_confound,
)
from .labels import build_label_table, attach_labels, save_label_table, LABEL_COLS

__all__ = [
    "build_unit_features", "mean_waveform", "polarity_normalized_waveform",
    "waveform_shape_features", "spike_train_features", "autocorrelogram", "OSORT_FS_HZ",
    "assign_narrow_broad", "interneuron_verification",
    "preprocess_waveforms", "run_wavemap", "cluster_waveform_importance",
    "characterize_clusters", "check_dataset_confound",
    "build_label_table", "attach_labels", "save_label_table", "LABEL_COLS",
]
