"""
Router calibration script.

Scores all normal (training) clips with the trained small autoencoder, then:
  1. Calibrates the routing threshold at a given quantile of reconstruction errors
  2. Computes the training-set embedding centroid for OOD detection
  3. Saves both to a router_state.pt file that evaluate.py can load

Usage:
    python scripts/calibrate_router.py \
        --small-ckpt checkpoints/small_ae_best.pt \
        --train-data  data/ucsd_ped2/train \
        --quantile    0.95 \
        --out         checkpoints/router_state.pt
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

# Make src importable when run from project root
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from device import get_device, safe_to_device
from models.small_autoencoder import SmallAutoencoder, anomaly_score as small_anomaly_score
from models.large_autoencoder import LargeAutoencoder
from preprocessing import build_clips

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
    p = argparse.ArgumentParser(description="Calibrate routing threshold from training data")
    p.add_argument("--small-ckpt",  required=True,
                   help="Path to trained small AE checkpoint (.pt)")
    p.add_argument("--large-ckpt",  default=None,
                   help="Path to trained large AE checkpoint (.pt). "
                        "If given, the large model's training distribution is also saved "
                        "for per-model score normalisation.")
    p.add_argument("--train-data",  default="data/ucsd_ped2/train",
                   help="Directory of training frame folders (normal only)")
    p.add_argument("--quantile",    type=float, default=0.95,
                   help="Quantile of normal scores to use as threshold (default: 0.95)")
    p.add_argument("--clip-len",    type=int, default=16)
    p.add_argument("--frame-size",  type=int, default=64)
    p.add_argument("--stride",      type=int, default=8,
                   help="Clip stride for calibration (need not be 1 — coarser is fine)")
    p.add_argument("--out",         default="checkpoints/router_state.pt",
                   help="Output path for the saved router state")
    p.add_argument("--batch",       type=int, default=32,
                   help="Number of clips to score at once (GPU memory budget)")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Per-scene clip collection
# ---------------------------------------------------------------------------

def collect_clips(train_root: Path, clip_len: int, stride: int, frame_size: int):
    """
    Walk every scene directory in train_root, build (C,T,H,W) clips.
    Returns a list of tensors.
    """
    all_clips = []
    scene_dirs = sorted(d for d in train_root.iterdir() if d.is_dir())
    logger.info(f"Found {len(scene_dirs)} training scenes under {train_root}")

    for scene_dir in scene_dirs:
        frame_paths = sorted(scene_dir.glob("*.png"))
        if not frame_paths:
            logger.warning(f"  No PNGs in {scene_dir.name}, skipping.")
            continue

        clips, _ = build_clips(
            [str(p) for p in frame_paths],
            clip_len=clip_len,
            stride=stride,
            frame_size=(frame_size, frame_size),
            detect_scene_cuts=False,   # not needed for calibration
        )
        all_clips.extend(clips)
        logger.info(f"  {scene_dir.name}: {len(frame_paths)} frames -> {len(clips)} clips")

    return all_clips


# ---------------------------------------------------------------------------
# Batch scoring
# ---------------------------------------------------------------------------

@torch.no_grad()
def score_clips(model: SmallAutoencoder, clips: list, device: torch.device, batch_size: int):
    """
    Score a list of (C,T,H,W) tensors in batches.
    Returns (scores_np, embeddings_np) as numpy arrays.
    """
    all_scores    = []
    all_embeddings = []

    for start in tqdm(range(0, len(clips), batch_size), desc="Scoring clips"):
        batch = clips[start : start + batch_size]
        # Stack to (B, C, T, H, W)
        batch_tensor = torch.stack(batch, dim=0)
        batch_tensor = safe_to_device(batch_tensor, device)

        # anomaly_score expects a single clip (B, C, T, H, W); score batch manually
        recon, mu, log_var = model(batch_tensor)
        # MSE per clip
        mse = ((recon - batch_tensor) ** 2).mean(dim=(1, 2, 3, 4))  # (B,)

        all_scores.extend(mse.cpu().tolist())
        all_embeddings.append(mu.cpu())   # (B, latent_dim)

    embeddings = torch.cat(all_embeddings, dim=0)   # (N, latent_dim)
    return np.array(all_scores, dtype=np.float32), embeddings


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _percentile_normalisers(scores: np.ndarray) -> dict:
    """
    Build a lookup for fast percentile-rank normalisation.
    Returns sorted training scores so we can use np.searchsorted at eval time.
    The normalised score of a new value x is:
        rank(x) / len(sorted_scores)
    which gives a value in [0, 1] representing how anomalous x is
    relative to the training distribution.
    """
    sorted_scores = np.sort(scores)
    return {
        "sorted_scores": sorted_scores,
        "min":  float(sorted_scores[0]),
        "max":  float(sorted_scores[-1]),
        "mean": float(sorted_scores.mean()),
    }


def main():
    args = parse_args()
    device = get_device()

    # ── Load small model ──────────────────────────────────────────────────────
    small_model = SmallAutoencoder(clip_len=args.clip_len, frame_size=args.frame_size).to(device)
    ckpt = torch.load(args.small_ckpt, map_location=device)
    small_model.load_state_dict(ckpt["model_state"])
    small_model.eval()
    epoch    = ckpt.get("epoch", "?")
    val_loss = ckpt.get("val_loss", float("nan"))
    logger.info(f"Loaded small AE checkpoint (epoch={epoch}, val_loss={val_loss:.5f})")

    # ── Optionally load large model ───────────────────────────────────────────
    large_model = None
    if args.large_ckpt:
        large_model = LargeAutoencoder(clip_len=args.clip_len, frame_size=args.frame_size).to(device)
        lckpt = torch.load(args.large_ckpt, map_location=device)
        large_model.load_state_dict(lckpt["model_state"])
        large_model.eval()
        logger.info(f"Loaded large AE checkpoint (epoch={lckpt.get('epoch','?')})")

    # ── Collect training clips ────────────────────────────────────────────────
    train_root = Path(args.train_data)
    if not train_root.exists():
        logger.error(f"Training data not found: {train_root}")
        sys.exit(1)

    clips = collect_clips(train_root, args.clip_len, args.stride, args.frame_size)
    if not clips:
        logger.error("No clips collected. Check --train-data path.")
        sys.exit(1)

    logger.info(f"Total calibration clips: {len(clips)}")

    # ── Score all clips with SMALL model ─────────────────────────────────────
    logger.info("Scoring with small model ...")
    small_scores, embeddings = score_clips(small_model, clips, device, args.batch)

    # ── Calibrate threshold ───────────────────────────────────────────────────
    threshold = float(np.quantile(small_scores, args.quantile))
    logger.info("Small model score distribution:")
    logger.info(f"  min    = {small_scores.min():.6f}")
    logger.info(f"  median = {float(np.median(small_scores)):.6f}")
    logger.info(f"  mean   = {small_scores.mean():.6f}")
    logger.info(f"  p95    = {float(np.quantile(small_scores, 0.95)):.6f}")
    logger.info(f"  p99    = {float(np.quantile(small_scores, 0.99)):.6f}")
    logger.info(f"  max    = {small_scores.max():.6f}")
    logger.info(f"Calibrated threshold (q={args.quantile:.2f}): {threshold:.6f}")

    # ── Compute embedding centroid ────────────────────────────────────────────
    centroid = embeddings.mean(dim=0)   # (latent_dim,)
    logger.info(f"Centroid computed from {len(embeddings)} embeddings "
                f"(dim={centroid.shape[0]})")

    # ── Build per-model score normalisers ─────────────────────────────────────
    small_norm = _percentile_normalisers(small_scores)
    logger.info(f"Small model percentile normaliser built ({len(small_norm['sorted_scores'])} points)")

    large_norm = None
    if large_model is not None:
        logger.info("Scoring with large model ...")
        large_scores, _ = score_clips(large_model, clips, device, args.batch)
        large_norm = _percentile_normalisers(large_scores)
        logger.info("Large model score distribution:")
        logger.info(f"  min    = {large_scores.min():.6f}")
        logger.info(f"  median = {float(np.median(large_scores)):.6f}")
        logger.info(f"  mean   = {large_scores.mean():.6f}")
        logger.info(f"  p95    = {float(np.quantile(large_scores, 0.95)):.6f}")
        logger.info(f"  p99    = {float(np.quantile(large_scores, 0.99)):.6f}")
        logger.info(f"  max    = {large_scores.max():.6f}")
        logger.info(f"Large model percentile normaliser built ({len(large_norm['sorted_scores'])} points)")

    # ── Save router state ─────────────────────────────────────────────────────
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "threshold":    threshold,
        "centroid":     centroid,
        "quantile":     args.quantile,
        "n_clips":      len(clips),
        # Small model distribution stats
        "small_score_min":    float(small_scores.min()),
        "small_score_median": float(np.median(small_scores)),
        "small_score_mean":   float(small_scores.mean()),
        "small_score_p95":    float(np.quantile(small_scores, 0.95)),
        "small_score_p99":    float(np.quantile(small_scores, 0.99)),
        "small_score_max":    float(small_scores.max()),
        # Percentile normalisers (sorted training scores for searchsorted)
        "small_norm":   small_norm,
        "large_norm":   large_norm,   # None if large model not provided
        # When --normalize-scores is used, scores are percentile ranks.
        # The threshold in that space equals the quantile itself (by definition).
        "normalized_threshold": args.quantile,
        # Metadata
        "small_ckpt":  str(args.small_ckpt),
        "large_ckpt":  str(args.large_ckpt) if args.large_ckpt else None,
        "epoch":       epoch,
    }
    torch.save(state, out_path)
    logger.info(f"Router state saved -> {out_path}")
    print(f"\nCalibration complete.")
    print(f"  Threshold  : {threshold:.6f}")
    print(f"  Centroid   : shape {list(centroid.shape)}")
    print(f"  Large norm : {'yes' if large_norm else 'no (large-ckpt not provided)'}")
    print(f"  Saved to   : {out_path}")
    print(f"\nRun evaluate.py with:  --router-state {out_path}")


if __name__ == "__main__":
    main()
