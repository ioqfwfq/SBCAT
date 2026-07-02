"""Build the per-unit cell-type LABEL table — the project deliverable.

One row per unit with:
    identity     unit_id, session_id, subject, location
    features     trough_to_peak_ms, half_width_ms, mean_rate_hz, isi_cv, burst_index,
                 acg_refrac_ms, ... (from features.build_unit_features)
    qc           wf_valid, st_valid, prop_isi_viol, SNR/isolation (if present), qc_pass
    LABELS       wf_group        narrow | broad          (trough-to-peak GMM antimode)
                 wavemap_cluster 0..k   (data-driven WaveMAP; NA if not run / QC-failed)
                 putative_type   'narrow (putative interneuron)' | 'broad (putative pyramidal)'

Downstream WM analyses just join this table on unit_id and use a label column as the
grouping variable — see attach_labels().
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .features import build_unit_features, OSORT_FS_HZ
from .classify import assign_narrow_broad

_PUTATIVE = {"narrow": "narrow (putative interneuron)",
             "broad": "broad (putative pyramidal)"}

# columns kept in the compact label table (features retained for transparency/QC)
LABEL_COLS = ["unit_id", "session_id", "subject", "location",
              "wf_group", "wavemap_cluster", "putative_type", "qc_pass",
              "trough_to_peak_ms", "half_width_ms", "mean_rate_hz", "isi_cv",
              "burst_index", "acg_refrac_ms", "prop_isi_viol",
              "waveforms_peak_snr", "waveforms_isolation_distance",
              "wf_valid", "st_valid", "n_spikes"]


def build_label_table(nwb_files, fs_hz: float = OSORT_FS_HZ, dataset: str | None = None,
                      do_wavemap: bool = True, min_peak_snr: float | None = None,
                      max_isi_viol: float | None = 0.05,
                      preprocess_kwargs: dict | None = None,
                      wavemap_kwargs: dict | None = None,
                      verbose: bool = True):
    """Full labeling pipeline for one dataset -> (label_df, extras).

    Parameters
    ----------
    min_peak_snr, max_isi_viol : QC gates for `qc_pass` (and WaveMAP inclusion).
        Defaults are lenient; tighten to WaveMAP's SNR>=3 / ISI-viol<0.0025 if desired.
    do_wavemap : run WaveMAP on the QC-passing units and add `wavemap_cluster`.
    dataset : optional tag stored in a `dataset` column (use when pooling 000673+000469).

    Returns (label_df, extras) where extras holds the full feature frame, the
    narrow/broad split (ms), and the WaveMAP result dict (embedding/graph/reducer) if run.
    """
    df, wfs = build_unit_features(nwb_files, fs_hz=fs_hz, return_waveforms=True, verbose=verbose)
    if dataset is not None:
        df.insert(1, "dataset", dataset)

    grp, split_ms = assign_narrow_broad(df, method="antimode")
    df["wf_group"] = grp
    df["putative_type"] = df["wf_group"].map(_PUTATIVE)

    # QC pass flag
    qc = df["wf_valid"].fillna(False) & df["st_valid"].fillna(False)
    if max_isi_viol is not None:
        qc &= pd.to_numeric(df["prop_isi_viol"], errors="coerce").fillna(1.0) <= max_isi_viol
    if min_peak_snr is not None and "waveforms_peak_snr" in df.columns:
        qc &= pd.to_numeric(df["waveforms_peak_snr"], errors="coerce").fillna(0.0) >= min_peak_snr
    df["qc_pass"] = qc.to_numpy()

    df["wavemap_cluster"] = pd.array([pd.NA] * len(df), dtype="Int64")
    extras = {"features": df, "nb_split_ms": split_ms, "wavemap": None}

    if do_wavemap:
        from .wavemap import preprocess_waveforms, run_wavemap
        wf_present = np.array([w is not None for w in wfs])
        use_mask = df["qc_pass"].to_numpy() & wf_present
        use_idx = np.where(use_mask)[0]
        if use_idx.size >= 20:
            wfs_use = [wfs[i] for i in use_idx]
            X, valid = preprocess_waveforms(wfs_use, fs_hz=fs_hz, **(preprocess_kwargs or {}))
            res = run_wavemap(X, **(wavemap_kwargs or {}))
            kept_idx = use_idx[valid]                      # df-row positions that got clustered
            df.loc[df.index[kept_idx], "wavemap_cluster"] = res["labels"]
            res["X"] = X
            res["unit_id"] = df["unit_id"].to_numpy()[kept_idx]
            extras["wavemap"] = res
            if verbose:
                n_cl = len(np.unique(res["labels"]))
                print(f"WaveMAP: {X.shape[0]} units -> {n_cl} clusters "
                      f"(sizes {np.bincount(res['labels']).tolist()})")
        elif verbose:
            print(f"WaveMAP skipped: only {use_idx.size} QC-pass units (need >=20; >=300 ideal)")

    keep = [c for c in LABEL_COLS if c in df.columns]
    if dataset is not None:
        keep = ["dataset"] + keep
    label_df = df[keep].copy()
    return label_df, extras


def attach_labels(analysis_df, label_df, on: str = "unit_id",
                  cols=("wf_group", "wavemap_cluster", "putative_type", "qc_pass")):
    """Left-join cell-type labels onto any trial/unit analysis frame keyed by unit_id.

    Lets a 'normal' downstream WM analysis group by wf_group / wavemap_cluster.
    """
    take = [on] + [c for c in cols if c in label_df.columns]
    return analysis_df.merge(label_df[take], on=on, how="left")


def save_label_table(label_df: pd.DataFrame, path) -> None:
    from pathlib import Path
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    label_df.to_csv(path, index=False)
