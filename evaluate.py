"""
Evaluation script for the video anomaly detection pipeline.

Runs the full small -> router -> large model pipeline on UCSD Ped2 test data,
computes frame-level anomaly scores, and reports:
  - Frame-level AUC-ROC
  - Frame-level AUROC (Equal Error Rate)
  - Escalation rate (% of clips sent to large model)
  - Per-scene breakdown

Usage:
    python evaluate.py \
        --small-ckpt checkpoints/small_ae_best.pt \
        --test-data  data/ucsd_ped2/test \
        --labels     data/ucsd_ped2/test_labels.csv
"""

import argparse
import csv
import logging
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")   # non-interactive backend; safe on any machine
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import roc_auc_score, roc_curve
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent / "src"))
from device import get_device, safe_to_device
from models.small_autoencoder import SmallAutoencoder
from models.small_autoencoder import anomaly_score as small_anomaly_score
from models.large_autoencoder import LargeAutoencoder
from models.large_autoencoder import anomaly_score as large_anomaly_score
from preprocessing import build_clips
from routing import Router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Evaluate anomaly detection pipeline")
    p.add_argument("--small-ckpt",  required=True,
                   help="Path to small AE checkpoint (.pt)")
    p.add_argument("--large-ckpt",  default=None,
                   help="Path to large AE checkpoint (.pt). Omit to use only small model.")
    p.add_argument("--test-data",   default="data/ucsd_ped2/test",
                   help="Directory containing per-scene test frame folders")
    p.add_argument("--labels",      default="data/ucsd_ped2/test_labels.csv",
                   help="CSV with columns: scene, frame_idx, label")
    p.add_argument("--clip-len",    type=int, default=16)
    p.add_argument("--frame-size",  type=int, default=64)
    p.add_argument("--threshold",   type=float, default=None,
                   help="Routing threshold (auto-calibrated from test normal frames if not set)")
    p.add_argument("--out-dir",     default="logs",
                   help="Directory to save ROC curve plots and score CSVs")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_small(ckpt_path: str, clip_len: int, frame_size: int,
               device: torch.device) -> SmallAutoencoder:
    model = SmallAutoencoder(clip_len=clip_len, frame_size=frame_size).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    epoch = ckpt.get("epoch", "?")
    val   = ckpt.get("val_loss", float("nan"))
    logger.info(f"Loaded small AE  (epoch={epoch}, val_loss={val:.5f})")
    return model


def load_large(ckpt_path: str, clip_len: int, frame_size: int,
               device: torch.device) -> LargeAutoencoder:
    model = LargeAutoencoder(clip_len=clip_len, frame_size=frame_size).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    logger.info(f"Loaded large AE  (epoch={ckpt.get('epoch','?')})")
    return model


# ---------------------------------------------------------------------------
# Ground-truth loading
# ---------------------------------------------------------------------------

def load_labels(csv_path: str) -> dict:
    """
    Returns {scene_name: [label_frame0, label_frame1, ...]}
    where label is 0 (normal) or 1 (anomalous).
    """
    labels = defaultdict(list)
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            labels[row["scene"]].append(int(row["label"]))
    return dict(labels)


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

@torch.no_grad()
def score_scene(
    scene_dir: Path,
    small_model: SmallAutoencoder,
    large_model,          # LargeAutoencoder or None
    router: Router,
    clip_len: int,
    frame_size: int,
    device: torch.device,
) -> tuple:
    """
    Score all frames in a scene directory.

    Returns:
        frame_scores  – np.array of anomaly scores, one per frame
        escalated     – number of clips escalated to the large model
        total_clips   – total clips scored
    """
    frame_paths = sorted(scene_dir.glob("*.png"))
    if not frame_paths:
        return np.array([]), 0, 0

    frame_paths = [str(p) for p in frame_paths]
    clips, scene_cuts = build_clips(
        frame_paths,
        clip_len=clip_len,
        stride=1,           # stride=1 gives a score for every frame
        frame_size=(frame_size, frame_size),
        detect_scene_cuts=True,
    )

    # Map clip index -> starting frame index (with stride=1, clip i starts at frame i)
    n_frames  = len(frame_paths)
    clip_scores = np.zeros(n_frames, dtype=np.float32)
    clip_counts = np.zeros(n_frames, dtype=np.int32)
    escalated   = 0

    for i, (clip, cut) in enumerate(zip(clips, scene_cuts)):
        clip_gpu = safe_to_device(clip.unsqueeze(0), device)  # (1, C, T, H, W)

        # Score with small model
        small_score, embedding, uncertainty = small_anomaly_score(small_model, clip_gpu)

        # Routing decision
        decision = router.route(small_score, embedding, uncertainty, scene_cut=cut)

        if decision.escalate and large_model is not None:
            final_score, _, _ = large_anomaly_score(large_model, clip_gpu)
            escalated += 1
        else:
            final_score = small_score

        # Distribute the clip score across its frames (overlap-add, average later)
        start = i
        end   = min(i + clip_len, n_frames)
        clip_scores[start:end] += final_score
        clip_counts[start:end] += 1

    # Average overlapping clip scores per frame
    clip_counts = np.maximum(clip_counts, 1)
    frame_scores = clip_scores / clip_counts
    return frame_scores, escalated, len(clips)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(all_scores: np.ndarray, all_labels: np.ndarray) -> dict:
    """Compute AUC-ROC and Equal Error Rate."""
    auc = roc_auc_score(all_labels, all_scores)
    fpr, tpr, thresholds = roc_curve(all_labels, all_scores)

    # Equal Error Rate: point where FPR == FNR
    fnr  = 1 - tpr
    eer_idx = np.argmin(np.abs(fpr - fnr))
    eer = float((fpr[eer_idx] + fnr[eer_idx]) / 2)

    return {"auc_roc": float(auc), "eer": eer, "fpr": fpr, "tpr": tpr}


def plot_roc(fpr, tpr, auc, out_path: str):
    """Save an ROC curve plot to `out_path`."""
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color="steelblue", lw=2,
             label=f"AUC = {auc:.4f}")
    plt.plot([0, 1], [0, 1], "k--", lw=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve — UCSD Ped2")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    logger.info(f"ROC curve saved -> {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    device = get_device()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load models
    small_model = load_small(args.small_ckpt, args.clip_len, args.frame_size, device)
    large_model = None
    if args.large_ckpt:
        large_model = load_large(args.large_ckpt, args.clip_len, args.frame_size, device)

    # Load ground-truth labels
    gt_labels = load_labels(args.labels)

    # Build router (threshold auto-calibrated if not provided)
    router = Router(
        threshold=args.threshold if args.threshold else 0.02,
        gray_zone_margin=0.3,
        ood_threshold=0.4,
        budget_capacity=10_000,
        budget_refill=1.0,   # effectively unlimited for evaluation
    )

    # Score every test scene
    test_root = Path(args.test_data)
    all_scores, all_labels = [], []
    total_clips, total_escalated = 0, 0
    scene_results = {}

    scenes = sorted(d for d in test_root.iterdir() if d.is_dir())
    logger.info(f"Scoring {len(scenes)} test scenes ...")

    for scene_dir in tqdm(scenes, desc="Scenes"):
        scene_name = scene_dir.name
        if scene_name not in gt_labels:
            logger.warning(f"No labels for scene {scene_name}, skipping.")
            continue

        frame_scores, esc, n_clips = score_scene(
            scene_dir, small_model, large_model, router,
            args.clip_len, args.frame_size, device,
        )
        labels = np.array(gt_labels[scene_name])

        # Align lengths (score array may be shorter due to clip_len stride)
        min_len = min(len(frame_scores), len(labels))
        frame_scores = frame_scores[:min_len]
        labels       = labels[:min_len]

        all_scores.extend(frame_scores.tolist())
        all_labels.extend(labels.tolist())
        total_clips     += n_clips
        total_escalated += esc

        # Per-scene AUC (only meaningful if both classes are present)
        if labels.sum() > 0 and (1 - labels).sum() > 0:
            scene_auc = roc_auc_score(labels, frame_scores)
        else:
            scene_auc = float("nan")
        scene_results[scene_name] = scene_auc

    all_scores = np.array(all_scores)
    all_labels = np.array(all_labels)

    # Global metrics
    metrics = compute_metrics(all_scores, all_labels)
    esc_rate = 100 * total_escalated / max(total_clips, 1)

    # ── Print results ────────────────────────────────────────────────────────
    print("\n" + "=" * 55)
    print("  UCSD Ped2 Evaluation Results")
    print("=" * 55)
    print(f"  Frame-level AUC-ROC : {metrics['auc_roc']:.4f}")
    print(f"  Equal Error Rate    : {metrics['eer']:.4f}")
    print(f"  Total clips scored  : {total_clips}")
    print(f"  Escalated clips     : {total_escalated}  ({esc_rate:.1f}%)")
    if large_model is None:
        print("  [Large model not used — small model only]")
    print("-" * 55)
    print("  Per-scene AUC-ROC:")
    for scene, auc in sorted(scene_results.items()):
        print(f"    {scene:12s}  {auc:.4f}" if not np.isnan(auc) else f"    {scene:12s}  N/A")
    print("=" * 55 + "\n")

    # ── Save ROC plot ────────────────────────────────────────────────────────
    plot_roc(
        metrics["fpr"], metrics["tpr"], metrics["auc_roc"],
        str(out_dir / "roc_curve.png"),
    )

    # ── Save raw scores CSV ─────────────────────────────────────────────────
    scores_csv = out_dir / "frame_scores.csv"
    with open(scores_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame_idx", "score", "label"])
        for i, (s, l) in enumerate(zip(all_scores.tolist(), all_labels.tolist())):
            writer.writerow([i, f"{s:.6f}", int(l)])
    logger.info(f"Scores saved -> {scores_csv}")


if __name__ == "__main__":
    main()
