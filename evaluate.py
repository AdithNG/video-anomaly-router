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
# Score normalisation helpers
# ---------------------------------------------------------------------------

def build_normaliser(sorted_scores: np.ndarray):
    """
    Returns a callable that maps a raw reconstruction error to a percentile rank
    within the training distribution (0 = min normal score, 1 = max normal score+).
    Uses binary search (searchsorted) for O(log n) lookups.
    """
    def normalise(raw_score: float) -> float:
        rank = float(np.searchsorted(sorted_scores, raw_score, side="right"))
        return rank / len(sorted_scores)   # in [0, ~1+]
    return normalise


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Evaluate anomaly detection pipeline")
    p.add_argument("--small-ckpt",    required=True,
                   help="Path to small AE checkpoint (.pt)")
    p.add_argument("--large-ckpt",    default=None,
                   help="Path to large AE checkpoint (.pt). Omit to use only small model.")
    p.add_argument("--router-state",  default=None,
                   help="Path to router_state.pt from calibrate_router.py "
                        "(sets threshold + centroid automatically)")
    p.add_argument("--test-data",     default="data/ucsd_ped2/test",
                   help="Directory containing per-scene test frame folders")
    p.add_argument("--labels",        default="data/ucsd_ped2/test_labels.csv",
                   help="CSV with columns: scene, frame_idx, label")
    p.add_argument("--clip-len",      type=int, default=16)
    p.add_argument("--frame-size",    type=int, default=64)
    p.add_argument("--threshold",     type=float, default=None,
                   help="Manual routing threshold. Overridden by --router-state if both given.")
    p.add_argument("--gray-zone-margin", type=float, default=0.3,
                   help="Gray-zone half-width as fraction of threshold (default 0.3)")
    p.add_argument("--ood-threshold", type=float, default=0.4,
                   help="OOD cosine-distance cutoff (default 0.4)")
    p.add_argument("--normalize-scores", action="store_true",
                   help="Map raw MSE to percentile rank within training distribution "
                        "(requires --router-state from calibrate_router.py with normalisers). "
                        "Makes small + large model scores comparable.")
    p.add_argument("--out-dir",       default="logs",
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
    small_normaliser=None,   # callable(raw_score) -> percentile rank, or None
    large_normaliser=None,   # callable(raw_score) -> percentile rank, or None
) -> tuple:
    """
    Score all frames in a scene directory.

    Returns:
        frame_scores       – np.array of anomaly scores, one per frame
        routing_decisions  – number of clips the router flagged for escalation
        actual_escalations – number of clips actually sent to the large model
        total_clips        – total clips scored
        signal_stats       – dict with mean values of each routing signal
    """
    frame_paths = sorted(scene_dir.glob("*.png"))
    if not frame_paths:
        return np.array([]), 0, 0, 0, {}

    frame_paths = [str(p) for p in frame_paths]
    clips, scene_cuts = build_clips(
        frame_paths,
        clip_len=clip_len,
        stride=1,           # stride=1 gives a score for every frame
        frame_size=(frame_size, frame_size),
        detect_scene_cuts=True,
    )

    n_frames          = len(frame_paths)
    clip_scores       = np.zeros(n_frames, dtype=np.float32)
    clip_counts       = np.zeros(n_frames, dtype=np.int32)
    routing_decisions = 0   # clips the router flagged (regardless of large model)
    actual_escalations = 0  # clips actually sent to the large model

    # Accumulators for routing signal diagnostics
    small_scores_list = []
    gz_dists     = []
    ood_scores   = []
    instabs      = []

    for i, (clip, cut) in enumerate(zip(clips, scene_cuts)):
        clip_gpu = safe_to_device(clip.unsqueeze(0), device)  # (1, C, T, H, W)

        # Score with small model
        raw_small, embedding, uncertainty = small_anomaly_score(small_model, clip_gpu)
        # Apply normalisation if available (converts to percentile rank)
        small_score = small_normaliser(raw_small) if small_normaliser else raw_small

        # Routing decision (always uses the raw or normalised small score consistently)
        decision = router.route(small_score, embedding, uncertainty, scene_cut=cut)

        small_scores_list.append(small_score)
        gz_dists.append(decision.gray_zone_dist)
        ood_scores.append(decision.ood_score)
        instabs.append(decision.temporal_instability)

        if decision.escalate:
            routing_decisions += 1
            if large_model is not None:
                raw_large, _, _ = large_anomaly_score(large_model, clip_gpu)
                # Apply large model normalisation if available
                final_score = large_normaliser(raw_large) if large_normaliser else raw_large
                actual_escalations += 1
            else:
                final_score = small_score  # no large model available yet
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

    signal_stats = {
        "small_score_mean": float(np.mean(small_scores_list)) if small_scores_list else 0.0,
        "small_score_min":  float(np.min(small_scores_list))  if small_scores_list else 0.0,
        "small_score_max":  float(np.max(small_scores_list))  if small_scores_list else 0.0,
        "gz_dist_mean":     float(np.mean(gz_dists))          if gz_dists else 0.0,
        "ood_score_mean":   float(np.mean(ood_scores))        if ood_scores else 0.0,
        "instab_mean":      float(np.mean(instabs))           if instabs else 0.0,
    }
    return frame_scores, routing_decisions, actual_escalations, len(clips), signal_stats


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

    # ── Build router ─────────────────────────────────────────────────────────
    # Priority: --router-state > --threshold > hardcoded default (uncalibrated)
    router_threshold = 0.02   # uncalibrated placeholder
    router_centroid  = None

    if args.router_state:
        state = torch.load(args.router_state, map_location="cpu", weights_only=False)
        router_threshold = state["threshold"]
        router_centroid  = state.get("centroid", None)
        logger.info(
            f"Loaded router state from {args.router_state}  "
            f"(threshold={router_threshold:.6f}, "
            f"calibrated from {state.get('n_clips', '?')} clips "
            f"at q={state.get('quantile', '?')})"
        )
        if router_centroid is not None:
            logger.info(f"  OOD centroid loaded (dim={router_centroid.shape[0]})")
    elif args.threshold is not None:
        router_threshold = args.threshold
        logger.info(f"Using manual threshold: {router_threshold:.6f}")
    else:
        logger.warning(
            "No --router-state or --threshold provided. "
            "Using uncalibrated default threshold=0.02. "
            "Run scripts/calibrate_router.py first for meaningful routing."
        )

    router = Router(
        threshold=router_threshold,
        gray_zone_margin=args.gray_zone_margin,
        ood_threshold=args.ood_threshold,
        budget_capacity=10_000,
        budget_refill=1.0,   # effectively unlimited for evaluation
    )
    if router_centroid is not None:
        router.set_centroid(router_centroid)

    # ── Build per-model score normalisers (optional) ──────────────────────────
    small_normaliser = None
    large_normaliser = None
    if args.normalize_scores:
        if not args.router_state:
            logger.warning("--normalize-scores requires --router-state with normaliser data. Skipping.")
        else:
            small_norm_data = state.get("small_norm", None)
            large_norm_data = state.get("large_norm", None)
            if small_norm_data is not None:
                sorted_small = np.array(small_norm_data["sorted_scores"])
                small_normaliser = build_normaliser(sorted_small)
                logger.info(f"Small model score normaliser loaded "
                            f"({len(sorted_small)} training-score reference points)")
                # Switch router threshold to normalised space (percentile rank scale)
                norm_thresh = state.get("normalized_threshold", None)
                if norm_thresh is not None:
                    router.threshold = norm_thresh
                    logger.info(f"Router threshold updated to normalised scale: {norm_thresh:.4f}")
            if large_norm_data is not None:
                sorted_large = np.array(large_norm_data["sorted_scores"])
                large_normaliser = build_normaliser(sorted_large)
                logger.info(f"Large model score normaliser loaded "
                            f"({len(sorted_large)} training-score reference points)")
            elif args.large_ckpt and large_norm_data is None:
                logger.warning(
                    "Large model normaliser not found in router state. "
                    "Re-run calibrate_router.py with --large-ckpt to generate it."
                )

    # Score every test scene
    test_root = Path(args.test_data)
    all_scores, all_labels = [], []
    total_clips               = 0
    total_routing_decisions   = 0   # clips router flagged for escalation
    total_actual_escalations  = 0   # clips actually sent to large model
    scene_results    = {}
    scene_signals    = {}           # diagnostic routing signals per scene

    scenes = sorted(d for d in test_root.iterdir() if d.is_dir())
    logger.info(f"Scoring {len(scenes)} test scenes ...")

    for scene_dir in tqdm(scenes, desc="Scenes"):
        scene_name = scene_dir.name
        if scene_name not in gt_labels:
            logger.warning(f"No labels for scene {scene_name}, skipping.")
            continue

        frame_scores, routing_dec, actual_esc, n_clips, sig_stats = score_scene(
            scene_dir, small_model, large_model, router,
            args.clip_len, args.frame_size, device,
            small_normaliser=small_normaliser,
            large_normaliser=large_normaliser,
        )
        labels = np.array(gt_labels[scene_name])

        # Align lengths (score array may be shorter due to clip_len stride)
        min_len = min(len(frame_scores), len(labels))
        frame_scores = frame_scores[:min_len]
        labels       = labels[:min_len]

        all_scores.extend(frame_scores.tolist())
        all_labels.extend(labels.tolist())
        total_clips                += n_clips
        total_routing_decisions    += routing_dec
        total_actual_escalations   += actual_esc
        scene_signals[scene_name]   = sig_stats

        # Per-scene AUC (only meaningful if both classes are present)
        if labels.sum() > 0 and (1 - labels).sum() > 0:
            scene_auc = roc_auc_score(labels, frame_scores)
        else:
            scene_auc = float("nan")
        scene_results[scene_name] = scene_auc

    all_scores = np.array(all_scores)
    all_labels = np.array(all_labels)

    # Global metrics
    metrics        = compute_metrics(all_scores, all_labels)
    routing_rate   = 100 * total_routing_decisions  / max(total_clips, 1)
    esc_rate       = 100 * total_actual_escalations / max(total_clips, 1)

    # ── Print results ────────────────────────────────────────────────────────
    calibrated_str = (
        f"calibrated (q={state.get('quantile','?')})" if args.router_state
        else ("manual" if args.threshold else "UNCALIBRATED DEFAULT")
    )
    print("\n" + "=" * 65)
    print("  UCSD Ped2 Evaluation Results")
    print("=" * 65)
    print(f"  Frame-level AUC-ROC  : {metrics['auc_roc']:.4f}")
    print(f"  Equal Error Rate     : {metrics['eer']:.4f}")
    print(f"  Total clips scored   : {total_clips}")
    print(f"  Router flagged       : {total_routing_decisions}  ({routing_rate:.1f}%)  "
          f"[clips router wanted to escalate]")
    print(f"  Large model called   : {total_actual_escalations}  ({esc_rate:.1f}%)  "
          f"[clips actually re-scored]")
    print(f"  Router threshold     : {router_threshold:.6f}  [{calibrated_str}]")
    print(f"  Gray-zone margin     : {args.gray_zone_margin}  |  "
          f"OOD threshold: {args.ood_threshold}")
    if large_model is None:
        print("  [No large model loaded — router decisions tracked but not acted on]")
    print("-" * 65)
    print("  Per-scene AUC-ROC  +  routing signal diagnostics:")
    print(f"  {'Scene':12s}  {'AUC':>6}  {'score_mean':>10}  "
          f"{'gz_dist':>7}  {'ood':>6}  {'instab':>8}")
    print("  " + "-" * 60)
    for scene in sorted(scene_results.keys()):
        auc   = scene_results[scene]
        sig   = scene_signals.get(scene, {})
        auc_s = f"{auc:.4f}" if not np.isnan(auc) else "  N/A"
        flag  = ""
        if not np.isnan(auc):
            if auc >= 0.9:
                flag = " (*)"
            elif auc < 0.5:
                flag = " (!)"
        print(
            f"  {scene:12s}  {auc_s:>6}{flag}  "
            f"{sig.get('small_score_mean', 0):10.5f}  "
            f"{sig.get('gz_dist_mean', 0):7.3f}  "
            f"{sig.get('ood_score_mean', 0):6.3f}  "
            f"{sig.get('instab_mean', 0):8.5f}"
        )
    print("=" * 65)
    print("  (*) = strong   (!) = worse than random")
    print("=" * 65 + "\n")

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

    # ── Save summary JSON ────────────────────────────────────────────────────
    import json, datetime
    summary = {
        "timestamp":           datetime.datetime.now().isoformat(timespec="seconds"),
        "auc_roc":             metrics["auc_roc"],
        "eer":                 metrics["eer"],
        "total_clips":         total_clips,
        "router_flagged":      total_routing_decisions,
        "routing_rate_pct":    round(routing_rate, 2),
        "large_model_called":  total_actual_escalations,
        "escalation_rate_pct": round(esc_rate, 2),
        "router_threshold":    router_threshold,
        "threshold_source":    calibrated_str,
        "gray_zone_margin":    args.gray_zone_margin,
        "ood_threshold":       args.ood_threshold,
        "large_model_used":    large_model is not None,
        "score_normalised":    args.normalize_scores and small_normaliser is not None,
        "per_scene_auc":       {
            k: (round(v, 4) if not np.isnan(v) else None)
            for k, v in sorted(scene_results.items())
        },
        "per_scene_signals":   {
            k: {sk: round(sv, 6) for sk, sv in sv_dict.items()}
            for k, sv_dict in scene_signals.items()
        },
    }
    summary_path = out_dir / "results_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Summary saved -> {summary_path}")


if __name__ == "__main__":
    main()
