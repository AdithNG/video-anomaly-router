"""
Download and prepare the UCSD Ped2 dataset.

UCSD Ped2 structure (after extraction):
  UCSD_Anomaly_Dataset.v1p2/
    UCSDped2/
      Train/
        Train001/ ... Train016/   ← normal frames (.tif)
      Test/
        Test001/ ... Test012/     ← frames (.tif)
        Test001_gt/ ... Test012_gt/  ← binary ground truth masks (.bmp)

This script:
  1. Downloads the archive (~500 MB) if not already present
  2. Extracts it
  3. Copies frames into data/ucsd_ped2/train/ and data/ucsd_ped2/test/
  4. Builds a frame-level ground-truth CSV for evaluation

Usage:
    python scripts/prepare_ucsd.py --dest data
"""

import argparse
import csv
import os
import shutil
import tarfile
import urllib.request
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

# UCSD Ped2 is openly hosted by the original authors
UCSD_URL = "http://www.svcl.ucsd.edu/projects/anomaly/UCSD_Anomaly_Dataset.tar.gz"
ARCHIVE_NAME = "UCSD_Anomaly_Dataset.tar.gz"

# After extraction the relevant sub-dataset lives here
PED2_SUBDIR = Path("UCSD_Anomaly_Dataset.v1p2") / "UCSDped2"


# ---------------------------------------------------------------------------
# Download helpers
# ---------------------------------------------------------------------------

def _reporthook(downloaded: int, block_size: int, total: int):
    """Simple download progress bar using tqdm."""
    if not hasattr(_reporthook, "_pbar"):
        _reporthook._pbar = tqdm(
            total=total, unit="B", unit_scale=True, desc="  Downloading"
        )
    _reporthook._pbar.update(downloaded * block_size - _reporthook._pbar.n
                             if downloaded > 0 else 0)


def download_ucsd(dest_dir: Path) -> Path:
    """
    Download the UCSD Anomaly Dataset archive into dest_dir.
    Skips the download if the archive already exists.

    Returns the path to the downloaded archive.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    archive_path = dest_dir / ARCHIVE_NAME

    if archive_path.exists():
        print(f"  Archive already exists: {archive_path}")
        return archive_path

    print(f"  Downloading UCSD Ped2 from:\n    {UCSD_URL}")
    urllib.request.urlretrieve(UCSD_URL, archive_path, _reporthook)
    print()  # newline after progress bar
    return archive_path


def extract_ucsd(archive_path: Path, dest_dir: Path) -> Path:
    """
    Extract the archive. Returns the root extraction directory.
    Skips extraction if already done.
    """
    extracted_root = dest_dir / "UCSD_Anomaly_Dataset.v1p2"
    if extracted_root.exists():
        print(f"  Already extracted: {extracted_root}")
        return extracted_root

    print(f"  Extracting {archive_path.name} ...")
    with tarfile.open(archive_path, "r:gz") as tar:
        tar.extractall(dest_dir)
    print("  Done.")
    return extracted_root


# ---------------------------------------------------------------------------
# Frame organisation
# ---------------------------------------------------------------------------

def _tif_to_png(src: Path, dst: Path):
    """Read a .tif frame and write it as a .png (OpenCV handles this natively)."""
    img = cv2.imread(str(src), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise RuntimeError(f"Cannot read: {src}")
    # Convert grayscale to BGR so our RGB preprocessing pipeline works uniformly
    bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.imwrite(str(dst), bgr)


def organise_train(ped2_root: Path, out_dir: Path):
    """
    Copy all training frames (normal only) into:
        out_dir/train/<scene>/frame_XXXXXX.png
    """
    train_src = ped2_root / "Train"
    train_dst = out_dir / "train"

    print("  Organising training frames ...")
    for scene_dir in sorted(train_src.iterdir()):
        if not scene_dir.is_dir():
            continue
        dst_scene = train_dst / scene_dir.name
        dst_scene.mkdir(parents=True, exist_ok=True)

        tif_files = sorted(scene_dir.glob("*.tif"))
        for i, tif in enumerate(tqdm(tif_files, desc=f"    {scene_dir.name}", leave=False)):
            dst_frame = dst_scene / f"frame_{i:06d}.png"
            if not dst_frame.exists():
                _tif_to_png(tif, dst_frame)

    print(f"  Training frames -> {train_dst}")


def organise_test(ped2_root: Path, out_dir: Path) -> Path:
    """
    Copy test frames into:
        out_dir/test/<scene>/frame_XXXXXX.png

    Builds a CSV at out_dir/test_labels.csv:
        scene, frame_idx, label  (0=normal, 1=anomalous)

    Ground truth is provided as binary mask images in Test<N>_gt/ folders.
    A frame is labelled anomalous if the corresponding GT mask has any white pixel.

    Returns the path to the CSV.
    """
    test_src = ped2_root / "Test"
    test_dst = out_dir / "test"
    csv_path = out_dir / "test_labels.csv"

    print("  Organising test frames + building ground-truth CSV ...")
    rows = []  # (scene_name, frame_idx, label)

    # Collect scene dirs (exclude _gt dirs)
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

            # Ground truth: 1 if any pixel in the GT mask is white (anomaly)
            label = 0
            if i < len(gt_files):
                gt_mask = cv2.imread(str(gt_files[i]), cv2.IMREAD_GRAYSCALE)
                if gt_mask is not None and gt_mask.max() > 0:
                    label = 1

            rows.append((scene_dir.name, i, label))

    # Write CSV
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


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Download & prepare UCSD Ped2")
    parser.add_argument("--dest", default="data",
                        help="Root directory for all datasets (default: data/)")
    parser.add_argument("--skip-download", action="store_true",
                        help="Skip download (use if archive is already present)")
    args = parser.parse_args()

    dest = Path(args.dest)
    raw_dir  = dest / "raw"
    ped2_out = dest / "ucsd_ped2"

    print("\n=== UCSD Ped2 Dataset Preparation ===\n")

    # Step 1: Download
    if not args.skip_download:
        archive = download_ucsd(raw_dir)
    else:
        archive = raw_dir / ARCHIVE_NAME
        print(f"  Skipping download. Expected archive: {archive}")

    # Step 2: Extract
    extracted_root = extract_ucsd(archive, raw_dir)
    ped2_root = extracted_root / "UCSDped2"

    # Step 3: Organise train frames
    organise_train(ped2_root, ped2_out)

    # Step 4: Organise test frames + GT
    organise_test(ped2_root, ped2_out)

    print("\n=== Done! ===")
    print(f"  Training data : {ped2_out / 'train'}")
    print(f"  Test data     : {ped2_out / 'test'}")
    print(f"  GT labels     : {ped2_out / 'test_labels.csv'}")
    print(f"\nTo train:")
    print(f"  python train.py --model small --data {ped2_out / 'train'} --epochs 50\n")


if __name__ == "__main__":
    main()
