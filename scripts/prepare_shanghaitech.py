"""
Prepare the ShanghaiTech Campus dataset.

The ShanghaiTech dataset must be downloaded manually.
Download from the authors' release (requires registration/Google Drive):
  https://github.com/StevenLiuWen/sRNN_TSC_Anomaly_Detection

Expected raw structure after extraction:
  shanghaitech/
    training/
      videos/   01_001.avi ... 13_xxx.avi   (normal only, <scene>_<clip>.avi)
    testing/
      frames/   01_0014/  01_0015/ ...      (pre-extracted PNG frames)
      test_frame_mask/
        01_0014/ ... (PNG masks: white=anomaly)
      test_pixel_mask/  (optional, pixel-level)

If you have the dataset as pre-extracted frames + masks (the common distribution
format), use --frames-only to skip video extraction.

Output layout:
    data/shanghaitech/train/<scene>/frame_XXXXXX.png
    data/shanghaitech/test/<scene>/frame_XXXXXX.png
    data/shanghaitech/test_labels.csv  (columns: scene, frame_idx, label)

Usage:
    # If you have video files:
    python scripts/prepare_shanghaitech.py --raw data/raw/shanghaitech --dest data

    # If you have pre-extracted frames (common distribution):
    python scripts/prepare_shanghaitech.py --raw data/raw/shanghaitech --dest data --frames-only
"""

import argparse
import csv
import shutil
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


def _extract_frames_from_video(video_path: Path, out_dir: Path) -> int:
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


def _copy_frames(src_dir: Path, out_dir: Path) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    frames = sorted(src_dir.glob("*.png")) + sorted(src_dir.glob("*.jpg"))
    for i, f in enumerate(frames):
        dst = out_dir / f"frame_{i:06d}.png"
        if not dst.exists():
            img = cv2.imread(str(f))
            cv2.imwrite(str(dst), img)
    return len(frames)


def organise_train(raw_root: Path, out_dir: Path, frames_only: bool):
    train_dst = out_dir / "train"
    print("  Organising training data -> frames ...")

    if frames_only:
        train_src = raw_root / "training" / "frames"
        if not train_src.exists():
            train_src = raw_root / "training" / "videos"
    else:
        train_src = raw_root / "training" / "videos"

    if not train_src.exists():
        print(f"  WARNING: training source not found at {train_src}, skipping.")
        return

    if frames_only and (train_src.parent / "frames").exists():
        scene_dirs = sorted(d for d in train_src.iterdir() if d.is_dir())
        for scene_dir in tqdm(scene_dirs, desc="  Training scenes"):
            scene_name = scene_dir.name
            n = _copy_frames(scene_dir, train_dst / scene_name)
            tqdm.write(f"    {scene_name}: {n} frames")
    else:
        videos = sorted(train_src.glob("*.avi"))
        for vid in tqdm(videos, desc="  Training"):
            scene_name = vid.stem
            n = _extract_frames_from_video(vid, train_dst / scene_name)
            tqdm.write(f"    {vid.name}: {n} frames")

    print(f"  Training frames -> {train_dst}")


def organise_test(raw_root: Path, out_dir: Path, frames_only: bool) -> Path:
    test_dst = out_dir / "test"
    csv_path = out_dir / "test_labels.csv"
    print("  Organising test data -> frames + GT CSV ...")

    test_frames_src = raw_root / "testing" / "frames"
    gt_mask_src     = raw_root / "testing" / "test_frame_mask"

    if not test_frames_src.exists():
        raise RuntimeError(f"Test frames directory not found: {test_frames_src}")

    scene_dirs = sorted(d for d in test_frames_src.iterdir() if d.is_dir())
    rows = []

    for scene_dir in tqdm(scene_dirs, desc="  Test scenes"):
        scene_name = scene_dir.name
        n = _copy_frames(scene_dir, test_dst / scene_name)

        # Frame-level GT from mask images
        gt_dir = gt_mask_src / scene_name if gt_mask_src.exists() else None
        labels = [0] * n

        if gt_dir and gt_dir.exists():
            mask_files = sorted(gt_dir.glob("*.png"))
            for i, mf in enumerate(mask_files):
                if i >= n:
                    break
                mask = cv2.imread(str(mf), cv2.IMREAD_GRAYSCALE)
                if mask is not None and mask.max() > 0:
                    labels[i] = 1

        for i in range(n):
            rows.append((scene_name, i, labels[i]))

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
    parser = argparse.ArgumentParser(description="Prepare ShanghaiTech for evaluation")
    parser.add_argument("--raw",  default="data/raw/shanghaitech",
                        help="Path to extracted shanghaitech/ directory")
    parser.add_argument("--dest", default="data",
                        help="Root output directory (default: data/)")
    parser.add_argument("--frames-only", action="store_true",
                        help="Data is pre-extracted frames, not videos")
    args = parser.parse_args()

    raw_root = Path(args.raw)
    out_dir  = Path(args.dest) / "shanghaitech"

    if not raw_root.exists():
        print(f"ERROR: ShanghaiTech dataset not found at {raw_root}")
        print("\nManual download steps:")
        print("  1. Request access: https://github.com/StevenLiuWen/sRNN_TSC_Anomaly_Detection")
        print("     (or search for 'ShanghaiTech Campus dataset' — multiple mirrors exist)")
        print(f"  2. Extract into {raw_root}/")
        print(f"  3. Re-run: python scripts/prepare_shanghaitech.py --raw {raw_root} --dest {args.dest}")
        raise SystemExit(1)

    print("\n=== ShanghaiTech Campus Dataset Preparation ===\n")
    organise_train(raw_root, out_dir, args.frames_only)
    organise_test(raw_root, out_dir, args.frames_only)
    print("\n=== Done! ===")
    print(f"  Training data : {out_dir / 'train'}")
    print(f"  Test data     : {out_dir / 'test'}")
    print(f"  GT labels     : {out_dir / 'test_labels.csv'}")


if __name__ == "__main__":
    main()
