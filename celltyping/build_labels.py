"""Entry point: build the per-unit cell-type label table for a WM dataset.

Run from E:\\SBCAT with the project venv, e.g.:
    .venv\\Scripts\\python.exe celltyping\\build_labels.py --dataset 000673
    .venv\\Scripts\\python.exe celltyping\\build_labels.py --dataset 000469 --no-wavemap
    .venv\\Scripts\\python.exe celltyping\\build_labels.py --dataset both

Writes outputs/celltype/unit_labels_<dataset>.csv and, per dataset, a QC figure.
Nothing here is auto-run — execute + debug at your own pace.
"""

import argparse
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from celltyping.labels import build_label_table, save_label_table   # noqa: E402
from celltyping import viz                                          # noqa: E402

# fs_hz is the OSort 100 kHz convention; verify per dataset (trough_to_peak ~0.2-1.0 ms).
DATA_DIR = ROOT.parent / "data"   # datasets pooled at repo-root data/
DATASETS = {
    "000673": dict(root=DATA_DIR / "000673", glob="sub-*/*.nwb", fs_hz=100_000.0),
    "000469": dict(root=DATA_DIR / "000469", glob="sub-*/*.nwb", fs_hz=100_000.0),
}


def run_one(name: str, do_wavemap: bool, out_dir: Path):
    cfg = DATASETS[name]
    files = sorted(Path(cfg["root"]).glob(cfg["glob"]))
    if not files:
        print(f"[{name}] no NWB files under {cfg['root']} (downloaded yet?) — skipping")
        return None
    print(f"[{name}] {len(files)} files")
    label_df, extras = build_label_table(
        files, fs_hz=cfg["fs_hz"], dataset=name, do_wavemap=do_wavemap, verbose=True)

    csv = out_dir / f"unit_labels_{name}.csv"
    save_label_table(label_df, csv)
    print(f"[{name}] wrote {csv}  ({len(label_df)} units)")

    # QC figures
    feats = extras["features"]
    fig1 = viz.plot_waveform_split(feats, split_ms=extras["nb_split_ms"])
    fig1.savefig(out_dir / f"qc_wfsplit_{name}.png", dpi=150, bbox_inches="tight")
    fig2 = viz.plot_interneuron_verification(feats)
    fig2.savefig(out_dir / f"qc_interneuron_{name}.png", dpi=150, bbox_inches="tight")
    wm = extras.get("wavemap")
    if wm is not None:
        # feat rows aligned to the clustered subset (kept unit_ids), for embedding colors
        sub = feats.set_index("unit_id").loc[wm["unit_id"]].reset_index()
        fig3 = viz.plot_wavemap_embedding(wm["embedding"], feat_df=sub, labels=wm["labels"])
        fig3.savefig(out_dir / f"wavemap_{name}.png", dpi=150, bbox_inches="tight")
    print(f"[{name}] QC figures saved to {out_dir}")
    return label_df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="000673", choices=["000673", "000469", "both"])
    ap.add_argument("--no-wavemap", action="store_true", help="skip WaveMAP clustering")
    ap.add_argument("--out", default=str(ROOT / "outputs" / "celltype"))
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    names = ["000673", "000469"] if args.dataset == "both" else [args.dataset]
    for name in names:
        run_one(name, do_wavemap=not args.no_wavemap, out_dir=out_dir)


if __name__ == "__main__":
    main()
