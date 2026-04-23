"""
Prepare the CUHK Avenue dataset.

The CUHK Avenue dataset must be downloaded manually from:
  http://www.cse.cuhk.edu.hk/leojia/projects/detectabnormal/dataset.html

Expected raw structure after extracting Avenue_Dataset.zip:
  Avenue_Dataset/
    training/
      videos/  01.avi ... 16.avi   (normal only)
    testing/
      videos/  01.avi ... 21.avi
    ground_truth_demo/
      testing_label_mask/
        1.mat  2.mat ... 21.mat    (per-frame pixel-level GT masks)

Output layout:
    data/avenue/train/<scene>/frame_XXXXXX.png
    data/avenue/test/<scene>/frame_XXXXXX.png
    data/avenue/test_labels.csv   (columns: scene, frame_idx, label)

Requires: scipy (for loading .mat files), opencv-python

Usage:
    python scripts/prepare_avenue.py --raw data/raw/Avenue_Dataset --dest data
"""

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

try:
    from scipy.io import loadmat
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


def _extract_frames(video_path: Path, out_dir: Path):
    """Extract all frames from a video to out_dir/frame_XXXXXX.png."""
    out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    i = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        dst = out_dir / f"frame_{i:06d}.png"
        if not dst.exists():
            cv2.imwrite(str(dst), frame)
        i += 1
    cap.release()
    return i


def organise_train(raw_root: Path, out_dir: Path):
    train_src = raw_root / "training" / "videos"
    train_dst = out_dir / "train"
    print("  Organising training videos -> frames ...")

    videos = sorted(train_src.glob("*.avi"))
    if not videos:
        raise RuntimeError(f"No .avi files found in {train_src}")

    for vid in tqdm(videos, desc="  Training"):
        scene_name = f"Train{int(vid.stem):03d}"
        dst_scene  = train_dst / scene_name
        n = _extract_frames(vid, dst_scene)
        tqdm.write(f"    {vid.name} -> {n} frames")

    print(f"  Training frames -> {train_dst}")


def organise_test(raw_root: Path, out_dir: Path) -> Path:
    test_src = raw_root / "testing" / "videos"
    gt_src   = raw_root / "ground_truth_demo" / "testing_label_mask"
    test_dst = out_dir / "test"
    csv_path = out_dir / "test_labels.csv"
    print("  Organising test videos -> frames + building GT CSV ...")

    if not HAS_SCIPY:
        raise RuntimeError("scipy is required to load .mat GT files. "
                           "Install with: pip install scipy")

    videos = sorted(test_src.glob("*.avi"))
    if not videos:
        raise RuntimeError(f"No .avi files found in {test_src}")

    rows = []
    for vid in tqdm(videos, desc="  Testing"):
        scene_idx  = int(vid.stem)
        scene_name = f"Test{scene_idx:03d}"
        dst_scene  = test_dst / scene_name
        n_frames   = _extract_frames(vid, dst_scene)

        # Load GT mask (shape: H x W x T or T x H x W depending on matlab version)
        mat_file = gt_src / f"{scene_idx}.mat"
        if mat_file.exists():
            mat  = loadmat(str(mat_file))
            # The key is typically 'volLabel' or the first non-meta key
            key  = [k for k in mat if not k.startswith("_")][0]
            mask = mat[key]  # H×W×T or T×H×W
            if mask.ndim == 3:
                # Ensure last dim is time
                if mask.shape[2] < mask.shape[0]:
                    mask = mask.transpose(2, 0, 1)  # now T×H×W
                else:
                    mask = mask.transpose(2, 0, 1)
                labels = [1 if mask[t].max() > 0 else 0
                          for t in range(min(mask.shape[0], n_frames))]
            else:
                labels = [0] * n_frames
        else:
            labels = [0] * n_frames

        for i in range(n_frames):
            rows.append((scene_name, i, labels[i] if i < len(labels) else 0))

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
    parser = argparse.ArgumentParser(description="Prepare CUHK Avenue for evaluation")
    parser.add_argument("--raw",  default="data/raw/Avenue_Dataset",
                        help="Path to extracted Avenue_Dataset/ (default: data/raw/Avenue_Dataset)")
    parser.add_argument("--dest", default="data",
                        help="Root output directory (default: data/)")
    args = parser.parse_args()

    raw_root = Path(args.raw)
    out_dir  = Path(args.dest) / "avenue"

    if not raw_root.exists():
        print(f"ERROR: Avenue dataset not found at {raw_root}")
        print("\nManual download steps:")
        print("  1. Visit: http://www.cse.cuhk.edu.hk/leojia/projects/detectabnormal/dataset.html")
        print("  2. Download Avenue_Dataset.zip")
        print(f"  3. Extract into {raw_root.parent}/")
        print(f"  4. Re-run: python scripts/prepare_avenue.py --raw {raw_root} --dest {args.dest}")
        raise SystemExit(1)

    print("\n=== CUHK Avenue Dataset Preparation ===\n")
    organise_train(raw_root, out_dir)
    organise_test(raw_root, out_dir)
    print("\n=== Done! ===")
    print(f"  Training data : {out_dir / 'train'}")
    print(f"  Test data     : {out_dir / 'test'}")
    print(f"  GT labels     : {out_dir / 'test_labels.csv'}")


if __name__ == "__main__":
    main()
