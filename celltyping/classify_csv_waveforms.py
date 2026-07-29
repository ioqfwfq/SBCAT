"""Narrow/broad cell typing for a CSV of mean waveforms (no NWB, no spike times).

Same three steps as the NWB pipeline, reusing the same code:
    1. polarity-normalize   celltyping.features.polarity_normalized_waveform
       (baseline = median of the first 30 samples; flip so the main deflection is a
       downward trough -- ~84% of these units are recorded positive-going)
    2. shape features       celltyping.features.waveform_shape_features
       (trough_to_peak_ms, half_width_ms, repol_slope, peak_trough_ratio, amplitude)
    3. split                celltyping.classify.assign_narrow_broad
       (2-component GMM on trough_to_peak_ms, boundary at the density antimode)

Input CSV layout: one row per cell, an ID column, then the samples in columns
(e.g. Neuron, Waveform_1 ... Waveform_256). Sample columns are taken in numeric
suffix order, not file order, so Waveform_10 cannot land next to Waveform_1.

Run from E:\\SBCAT\\celltyping with the project venv:
    ..\\.venv\\Scripts\\python.exe classify_csv_waveforms.py ^
        --csv "C:\\Users\\zhuj3\\Downloads\\neuron_waveforms_word_screen.csv"

Writes <input>_celltyped.csv (original columns + label columns inserted after the
ID column) and a QC figure. Nothing is overwritten in place unless --in-place.
"""

import argparse
import re
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import numpy as np                                                    # noqa: E402
import pandas as pd                                                   # noqa: E402

from celltyping.features import (waveform_shape_features,             # noqa: E402
                                 polarity_normalized_waveform, OSORT_FS_HZ)
from celltyping.classify import assign_narrow_broad                   # noqa: E402
from celltyping.labels import _PUTATIVE                               # noqa: E402

# label/feature columns this script adds, in output order
NEW_COLS = ["wf_group", "putative_type", "trough_to_peak_ms", "half_width_ms",
            "repol_slope", "peak_trough_ratio", "amplitude", "trough_idx", "peak_idx",
            "inverted", "wf_valid"]


def find_wave_cols(df: pd.DataFrame, prefix: str) -> list[str]:
    """Sample columns matching `prefix<number>`, ordered by that number."""
    pat = re.compile(rf"^{re.escape(prefix)}(\d+)$")
    hits = [(int(m.group(1)), c) for c in df.columns if (m := pat.match(str(c)))]
    if not hits:
        raise SystemExit(f"no columns match '{prefix}<number>' -- pass --wave-prefix "
                         f"(columns look like: {list(df.columns[:6])})")
    hits.sort()
    idx = [i for i, _ in hits]
    if idx != list(range(idx[0], idx[0] + len(idx))):
        print(f"[warn] sample indices are not contiguous ({idx[0]}..{idx[-1]}, "
              f"n={len(idx)}) -- taken in ascending numeric order anyway")
    return [c for _, c in hits]


def extract_features(W: np.ndarray, fs_hz: float) -> pd.DataFrame:
    """Per-row waveform shape features, with a flat/degenerate-waveform guard."""
    rows = []
    for w in W:
        if not np.all(np.isfinite(w)) or np.ptp(w) <= 0:
            f = waveform_shape_features(None)          # all-NaN, wf_valid False
        else:
            f = waveform_shape_features(w, fs_hz=fs_hz)
        rows.append(f)
    return pd.DataFrame(rows)


def qc_figure(feat: pd.DataFrame, W: np.ndarray, split_ms: float, fs_hz: float, path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from celltyping import viz

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.2))
    viz.plot_waveform_split(feat, split_ms=split_ms, ax=axes[0])

    v = feat[feat["wf_valid"].astype(bool)]
    for g, c in (("broad", "#1f77b4"), ("narrow", "#d62728")):
        s = v[v["wf_group"] == g]
        axes[1].scatter(s["trough_to_peak_ms"], s["half_width_ms"], s=8, alpha=0.4, color=c, label=g)
    axes[1].axvline(split_ms, color="k", ls="--", lw=1)
    axes[1].set_xlabel("trough-to-peak (ms)"); axes[1].set_ylabel("half-width (ms)")
    axes[1].set_title("width features"); axes[1].legend(fontsize=8)

    # peak-normalized, trough-down mean waveform per group
    t = np.arange(W.shape[1]) / fs_hz * 1000.0
    for g, c in (("broad", "#1f77b4"), ("narrow", "#d62728")):
        rows = np.where((feat["wf_group"] == g).to_numpy() & feat["wf_valid"].to_numpy(bool))[0]
        if rows.size == 0:
            continue
        norm = []
        for i in rows:
            w, _ = polarity_normalized_waveform(W[i])
            if w is not None and np.ptp(w) > 0:
                norm.append(w / np.abs(w.min()))
        if not norm:
            continue
        M = np.vstack(norm)
        mu, sd = M.mean(0), M.std(0)
        axes[2].plot(t, mu, color=c, lw=2, label=f"{g} (n={len(M)})")
        axes[2].fill_between(t, mu - sd, mu + sd, color=c, alpha=0.2)
    axes[2].set_xlabel("time (ms)"); axes[2].set_ylabel("normalized amplitude")
    axes[2].set_title("mean waveform per group (trough-down)"); axes[2].legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", required=True, help="input CSV: one row per cell")
    ap.add_argument("--id-col", default="Neuron", help="cell identifier column (default: Neuron)")
    ap.add_argument("--wave-prefix", default="Waveform_",
                    help="sample column prefix, e.g. 'Waveform_' or 'w' (default: Waveform_)")
    ap.add_argument("--fs", type=float, default=OSORT_FS_HZ,
                    help="waveform sampling rate in Hz (default: 100000, the OSort convention)")
    ap.add_argument("--method", default="antimode", choices=["antimode", "kde", "median", "fixed"],
                    help="split rule (default: antimode = 2-component GMM valley, the rule "
                         "used by the NWB pipeline)")
    ap.add_argument("--kde-window", type=float, nargs=2, default=(0.25, 0.75),
                    metavar=("LO", "HI"),
                    help="ms range searched for the narrow/broad valley (default: 0.25 0.75)")
    ap.add_argument("--kde-bw", type=float, default=0.15, help="KDE bandwidth (default: 0.15)")
    ap.add_argument("--fixed-ms", type=float, default=0.5,
                    help="boundary used by --method fixed, and the fallback when --method kde "
                         "finds no valley (default: 0.5)")
    ap.add_argument("--split-ms", type=float, default=None,
                    help="apply this boundary directly (e.g. the split from another dataset) "
                         "instead of fitting one")
    ap.add_argument("--out", default=None, help="output CSV (default: <input>_celltyped.csv)")
    ap.add_argument("--in-place", action="store_true", help="overwrite the input CSV")
    ap.add_argument("--labels-only", action="store_true",
                    help="drop the sample columns from the output (ID + labels + features only)")
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args()

    src = Path(args.csv)
    df = pd.read_csv(src)
    if args.id_col not in df.columns:
        raise SystemExit(f"--id-col {args.id_col!r} not in {list(df.columns[:8])}")
    wcols = find_wave_cols(df, args.wave_prefix)
    W = df[wcols].to_numpy(dtype=float)
    dur_ms = W.shape[1] / args.fs * 1000.0
    print(f"{src.name}: {len(df)} cells x {W.shape[1]} samples @ {args.fs:g} Hz = {dur_ms:.2f} ms")
    if not 0.5 <= dur_ms <= 10.0:
        print(f"[warn] {dur_ms:.2f} ms per waveform looks off -- check --fs")
    if df[args.id_col].duplicated().any():
        print(f"[warn] {int(df[args.id_col].duplicated().sum())} duplicate {args.id_col} values")

    feat = extract_features(W, fs_hz=args.fs)
    n_valid = int(feat["wf_valid"].sum())
    print(f"valid waveforms: {n_valid}/{len(feat)} | flipped (positive-going): "
          f"{int(feat['inverted'].sum())} ({feat['inverted'].mean():.0%})")
    ttp = pd.to_numeric(feat["trough_to_peak_ms"], errors="coerce")[feat["wf_valid"]].to_numpy()
    hw = pd.to_numeric(feat["half_width_ms"], errors="coerce")[feat["wf_valid"]].to_numpy()
    pct = [5, 25, 50, 75, 95]
    print(f"trough-to-peak (ms) pctiles {pct}: {np.round(np.percentile(ttp, pct), 3).tolist()}")
    print(f"half-width     (ms) pctiles {pct}: {np.round(np.percentile(hw, pct), 3).tolist()}")

    if args.split_ms is not None:
        split_ms = float(args.split_ms)
        valid = feat["wf_valid"].to_numpy(bool) & np.isfinite(
            pd.to_numeric(feat["trough_to_peak_ms"], errors="coerce").to_numpy())
        grp = pd.Series(pd.NA, index=feat.index, dtype="object")
        x = pd.to_numeric(feat["trough_to_peak_ms"], errors="coerce")
        grp[valid & (x < split_ms)] = "narrow"
        grp[valid & (x >= split_ms)] = "broad"
        print(f"split: applied --split-ms {split_ms:.3f} ms (not fitted)")
    else:
        grp, split_ms = assign_narrow_broad(feat, method=args.method, fixed_ms=args.fixed_ms,
                                            kde_window=tuple(args.kde_window), kde_bw=args.kde_bw)
        print(f"split: {args.method} -> {split_ms:.3f} ms")
    feat["wf_group"] = grp
    feat["putative_type"] = feat["wf_group"].map(_PUTATIVE)

    vc = feat["wf_group"].value_counts().to_dict()
    n_lab = sum(vc.get(k, 0) for k in ("narrow", "broad"))
    print(f"labels: {vc} | narrow = {vc.get('narrow', 0) / max(n_lab, 1):.1%} of {n_lab}")

    out_df = df.copy()
    for c in NEW_COLS:
        if c in out_df.columns:
            out_df = out_df.drop(columns=c)          # re-runnable
    if args.labels_only:
        out_df = out_df.drop(columns=wcols)
    insert_at = out_df.columns.get_loc(args.id_col) + 1
    for c in reversed(NEW_COLS):
        out_df.insert(insert_at, c, feat[c].to_numpy())

    out = Path(args.out) if args.out else (src if args.in_place
                                           else src.with_name(src.stem + "_celltyped.csv"))
    out.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out, index=False)
    print(f"wrote {out}  ({len(out_df)} rows, {len(NEW_COLS)} label columns added)")

    if not args.no_plot:
        png = out.with_name(out.stem + "_qc.png")
        qc_figure(feat, W, split_ms, args.fs, png)
        print(f"wrote {png}")


if __name__ == "__main__":
    main()
