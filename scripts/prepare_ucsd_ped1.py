"""
Prepare the UCSD Ped1 dataset for evaluation (frames already extracted).

UCSD Ped1 structure (as shipped with the anomaly dataset archive):
  data/raw/UCSD_Anomaly_Dataset.v1p2/UCSDped1/
    Train/  Train001 ... Train034      ← normal frames (.tif)
    Test/   Test001  ... Test036       ← frames (.tif)
            Test<N>_gt/                ← binary GT masks (.bmp) — only for anomalous scenes

Scenes WITHOUT a matching _gt folder contain no anomalies; all frames are
labelled 0. Scenes WITH a _gt folder get frame-level labels from the masks
(label=1 if any pixel in the mask is non-zero).

Output layout mirrors ucsd_ped2/:
    data/ucsd_ped1/train/<scene>/frame_XXXXXX.png
    data/ucsd_ped1/test/<scene>/frame_XXXXXX.png
    data/ucsd_ped1/test_labels.csv   (columns: scene, frame_idx, label)

Usage:
    python scripts/prepare_ucsd_ped1.py --raw data/raw --dest data
"""

import argparse
import csv
from pathlib import Path

import cv2
from tqdm import tqdm

PED1_SUBDIR = Path("UCSD_Anomaly_Dataset.v1p2") / "UCSDped1"


def _tif_to_png(src: Path, dst: Path):
    img = cv2.imread(str(src), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise RuntimeError(f"Cannot read: {src}")
    bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.imwrite(str(dst), bgr)


def organise_train(ped1_root: Path, out_dir: Path):
    train_src = ped1_root / "Train"
    train_dst = out_dir / "train"
    print("  Organising training frames ...")

    for scene_dir in sorted(train_src.iterdir()):
        if not scene_dir.is_dir():
            continue
        dst_scene = train_dst / scene_dir.name
        dst_scene.mkdir(parents=True, exist_ok=True)
        for i, tif in enumerate(sorted(scene_dir.glob("*.tif"))):
            dst_frame = dst_scene / f"frame_{i:06d}.png"
            if not dst_frame.exists():
                _tif_to_png(tif, dst_frame)

    print(f"  Training frames -> {train_dst}")


def organise_test(ped1_root: Path, out_dir: Path) -> Path:
    test_src = ped1_root / "Test"
    test_dst = out_dir / "test"
    csv_path = out_dir / "test_labels.csv"
    print("  Organising test frames + building ground-truth CSV ...")

    rows = []
    scene_dirs = sorted(
        d for d in test_src.iterdir()
        if d.is_dir() and not d.name.endswith("_gt")
    )

    for scene_dir in scene_dirs:
        gt_dir = test_src / f"{scene_dir.name}_gt"
        dst_scene = test_dst / scene_dir.name
        dst_scene.mkdir(parents=True, exist_ok=True)

        tif_files = sorted(scene_dir.glob("*.tif"))
        gt_files  = sorted(gt_dir.glob("*.bmp")) if gt_dir.exists() else []

        for i, tif in enumerate(tqdm(tif_files, desc=f"    {scene_dir.name}", leave=False)):
            dst_frame = dst_scene / f"frame_{i:06d}.png"
            if not dst_frame.exists():
                _tif_to_png(tif, dst_frame)

            label = 0
            if i < len(gt_files):
                gt_mask = cv2.imread(str(gt_files[i]), cv2.IMREAD_GRAYSCALE)
                if gt_mask is not None and gt_mask.max() > 0:
                    label = 1

            rows.append((scene_dir.name, i, label))

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["scene", "frame_idx", "label"])
        writer.writerows(rows)

    total   = len(rows)
    anomaly = sum(r[2] for r in rows)
    print(f"  Test frames  -> {test_dst}")
    print(f"  GT CSV       -> {csv_path}")
    print(f"  Frames: {total} total | {anomaly} anomalous ({100*anomaly/total:.1f}%)")
    return csv_path


def main():
    parser = argparse.ArgumentParser(description="Prepare UCSD Ped1 for evaluation")
    parser.add_argument("--raw",  default="data/raw",
                        help="Directory containing UCSD_Anomaly_Dataset.v1p2/ (default: data/raw)")
    parser.add_argument("--dest", default="data",
                        help="Root output directory (default: data/)")
    args = parser.parse_args()

    raw_dir  = Path(args.raw)
    ped1_root = raw_dir / PED1_SUBDIR
    ped1_out  = Path(args.dest) / "ucsd_ped1"

    if not ped1_root.exists():
        print(f"ERROR: UCSDped1 not found at {ped1_root}")
        print("Run: python scripts/prepare_ucsd.py --dest data  (to extract the archive)")
        raise SystemExit(1)

    print("\n=== UCSD Ped1 Dataset Preparation ===\n")
    organise_train(ped1_root, ped1_out)
    organise_test(ped1_root, ped1_out)
    print("\n=== Done! ===")
    print(f"  Training data : {ped1_out / 'train'}")
    print(f"  Test data     : {ped1_out / 'test'}")
    print(f"  GT labels     : {ped1_out / 'test_labels.csv'}")


if __name__ == "__main__":
    main()
