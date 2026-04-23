"""
Cost-quality Pareto curve sweep.

Scores every test clip once with both models, then sweeps gray-zone-margin
across 50 values without re-loading or re-scoring.  Records (escalation_rate,
AUC) at each point and plots the Pareto frontier against two flat baselines:
  - Small-only  (escalation = 0%)
  - Large-only  (escalation = 100%)

Usage:
    python scripts/pareto_sweep.py \
        --small-ckpt checkpoints/small_ae_baseline.pt \
        --large-ckpt checkpoints/large_ae_best.pt \
        --router-state checkpoints/router_state.pt \
        --test-data data/ucsd_ped2/test \
        --labels data/ucsd_ped2/test_labels.csv \
        --out-dir logs/pareto
"""

import argparse
import csv
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from device import get_device, safe_to_device
from models.small_autoencoder import SmallAutoencoder
from models.large_autoencoder import LargeAutoencoder
from preprocessing import build_clips
from routing import Router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--small-ckpt",   required=True)
    p.add_argument("--large-ckpt",   required=True)
    p.add_argument("--router-state", required=True)
    p.add_argument("--test-data",    default="data/ucsd_ped2/test")
    p.add_argument("--labels",       default="data/ucsd_ped2/test_labels.csv")
    p.add_argument("--clip-len",     type=int, default=16)
    p.add_argument("--frame-size",   type=int, default=64)
    p.add_argument("--margins",      default="0.01:1.0:0.02",
                   help="start:stop:step for gray-zone-margin sweep (default 0.01:1.0:0.02)")
    p.add_argument("--out-dir",      default="logs/pareto")
    return p.parse_args()


def load_labels(csv_path):
    labels = defaultdict(list)
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            labels[row["scene"]].append(int(row["label"]))
    return dict(labels)


def build_normaliser(sorted_scores):
    def norm(x):
        return float(np.searchsorted(sorted_scores, x, side="right") / len(sorted_scores))
    return norm


# ---------------------------------------------------------------------------
# Score every scene with both models — do this ONCE
# ---------------------------------------------------------------------------

@torch.no_grad()
def score_all_scenes(
    test_root, small_model, large_model, router_state,
    clip_len, frame_size, device, gt_labels, batch_size=32,
):
    """
    Scores every test clip with both models in batches.
    Returns (clip_records, scene_labels).
    """
    ckpt_args    = torch.load(router_state, map_location="cpu", weights_only=False)
    small_norm   = build_normaliser(np.array(ckpt_args["small_norm"]["sorted_scores"]))
    large_norm   = build_normaliser(np.array(ckpt_args["large_norm"]["sorted_scores"]))
    centroid     = ckpt_args["centroid"].float().to(device)
    threshold    = ckpt_args["normalized_threshold"]

    scenes = sorted(d for d in Path(test_root).iterdir() if d.is_dir())
    clip_records = []
    scene_labels = {}

    logger.info(f"Scoring {len(scenes)} scenes with both models ...")

    for scene_dir in tqdm(scenes, desc="Scenes"):
        name = scene_dir.name
        if name not in gt_labels:
            continue

        frame_paths = sorted(scene_dir.glob("*.png"))
        if not frame_paths:
            continue
        frame_paths = [str(p) for p in frame_paths]
        clips, cuts = build_clips(
            frame_paths, clip_len=clip_len, stride=1,
            frame_size=(frame_size, frame_size), detect_scene_cuts=True,
        )
        scene_labels[name] = np.array(gt_labels[name])
        n_frames = len(frame_paths)

        for start in range(0, len(clips), batch_size):
            batch_clips = clips[start:start + batch_size]
            batch_cuts  = cuts[start:start + batch_size]
            batch_t     = safe_to_device(torch.stack(batch_clips), device)  # (B,C,T,H,W)

            recon_s, mu, _ = small_model(batch_t)
            raw_s = ((recon_s - batch_t) ** 2).mean(dim=(1, 2, 3, 4)).cpu()

            recon_l, _, _  = large_model(batch_t)
            raw_l = ((recon_l - batch_t) ** 2).mean(dim=(1, 2, 3, 4)).cpu()

            cos = torch.nn.functional.cosine_similarity(
                mu.cpu().float(), centroid.cpu().float().unsqueeze(0).expand(len(batch_clips), -1)
            )
            ood_scores = (1.0 - cos).tolist()

            for j, (clip_cut, rs, rl, ood) in enumerate(
                zip(batch_cuts, raw_s.tolist(), raw_l.tolist(), ood_scores)
            ):
                norm_s = small_norm(rs)
                norm_l = large_norm(rl)
                gz_raw = abs(norm_s - threshold) / (threshold + 1e-8)
                clip_records.append(dict(
                    scene=name,
                    frame_start=start + j,
                    clip_len=clip_len,
                    n_frames=n_frames,
                    small_score=norm_s,
                    large_score=norm_l,
                    ood_score=ood,
                    gz_raw=gz_raw,
                    cut=clip_cut,
                ))

    return clip_records, scene_labels


# ---------------------------------------------------------------------------
# Apply routing at a given margin (no model re-scoring)
# ---------------------------------------------------------------------------

def group_by_scene(clip_records):
    """Group pre-scored clip records by scene name (call once, reuse across margins)."""
    groups = defaultdict(list)
    for rec in clip_records:
        groups[rec["scene"]].append(rec)
    return groups


def flat_scores(scene_clips, scene_labels, score_key):
    """Overlap-add average of a single model's scores with no routing."""
    all_s, all_l = [], []
    for name, clips in sorted(scene_clips.items()):
        if name not in scene_labels:
            continue
        n_frames = clips[0]["n_frames"]
        cs = np.zeros(n_frames, dtype=np.float32)
        cc = np.zeros(n_frames, dtype=np.int32)
        for rec in clips:
            st = rec["frame_start"]
            en = min(st + rec["clip_len"], n_frames)
            cs[st:en] += rec[score_key]
            cc[st:en] += 1
        cc = np.maximum(cc, 1)
        fs = cs / cc
        lb = scene_labels[name]
        mn = min(len(fs), len(lb))
        all_s.extend(fs[:mn].tolist())
        all_l.extend(lb[:mn].tolist())
    return np.array(all_s), np.array(all_l)


def apply_routing(scene_clips, scene_labels, margin, ood_threshold=0.4,
                  instab_threshold=0.005, history_len=10):
    """
    Apply routing decisions at `margin` over pre-grouped, pre-scored clips.
    Returns (all_scores, all_labels, escalation_rate).
    """
    all_scores  = []
    all_labels  = []
    total_clips = 0
    total_escal = 0

    for name, clips in sorted(scene_clips.items()):
        if name not in scene_labels:
            continue
        n_frames    = clips[0]["n_frames"]
        clip_scores = np.zeros(n_frames, dtype=np.float32)
        clip_counts = np.zeros(n_frames, dtype=np.int32)
        score_hist  = []

        for rec in clips:
            s_score = rec["small_score"]
            gz_dist = rec["gz_raw"]
            ood     = rec["ood_score"]
            cut     = rec["cut"]

            # Temporal instability from recent scores
            score_hist.append(s_score)
            if len(score_hist) > history_len:
                score_hist.pop(0)
            instab = float(np.var(score_hist)) if len(score_hist) >= 2 else 0.0

            escalate = (
                gz_dist < margin or
                ood > ood_threshold or
                instab > instab_threshold or
                cut
            )

            final = rec["large_score"] if escalate else s_score
            if escalate:
                total_escal += 1
            total_clips += 1

            start = rec["frame_start"]
            end   = min(start + rec["clip_len"], n_frames)
            clip_scores[start:end] += final
            clip_counts[start:end] += 1

        clip_counts = np.maximum(clip_counts, 1)
        frame_scores = clip_scores / clip_counts
        labels       = scene_labels[name]
        min_len      = min(len(frame_scores), len(labels))

        all_scores.extend(frame_scores[:min_len].tolist())
        all_labels.extend(labels[:min_len].tolist())

    all_scores  = np.array(all_scores)
    all_labels  = np.array(all_labels)
    escal_rate  = total_escal / max(total_clips, 1)
    return all_scores, all_labels, escal_rate


def compute_auc(scores, labels):
    if labels.sum() == 0 or (1 - labels).sum() == 0:
        return float("nan")
    return float(roc_auc_score(labels, scores))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args   = parse_args()
    device = get_device()
    out    = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ── Load models ───────────────────────────────────────────────────────────
    ckpt_s = torch.load(args.small_ckpt, map_location=device, weights_only=False)
    use_aux = ckpt_s.get("args", {}).get("pseudo_anomaly", False)
    small_model = SmallAutoencoder(
        clip_len=args.clip_len, frame_size=args.frame_size, use_aux_head=use_aux
    ).to(device)
    small_model.load_state_dict(ckpt_s["model_state"])
    small_model.eval()
    logger.info(f"Loaded small AE  (epoch={ckpt_s.get('epoch','?')}, "
                f"val_loss={ckpt_s.get('val_loss', float('nan')):.5f})")

    large_model = LargeAutoencoder(
        clip_len=args.clip_len, frame_size=args.frame_size
    ).to(device)
    large_model.load_state_dict(
        torch.load(args.large_ckpt, map_location=device, weights_only=False)["model_state"]
    )
    large_model.eval()
    logger.info("Loaded large AE")

    gt_labels = load_labels(args.labels)

    # ── Score everything once ─────────────────────────────────────────────────
    clip_records, scene_labels = score_all_scenes(
        args.test_data, small_model, large_model, args.router_state,
        args.clip_len, args.frame_size, device, gt_labels,
    )
    logger.info(f"Scored {len(clip_records)} clips total.")
    scene_clips = group_by_scene(clip_records)

    small_scores, small_labels = flat_scores(scene_clips, scene_labels, "small_score")
    large_scores, large_labels = flat_scores(scene_clips, scene_labels, "large_score")
    auc_small_only = compute_auc(small_scores, small_labels)
    auc_large_only = compute_auc(large_scores, large_labels)
    logger.info(f"Small-only AUC : {auc_small_only:.4f}")
    logger.info(f"Large-only AUC : {auc_large_only:.4f}")

    start_m, stop_m, step_m = (float(x) for x in args.margins.split(":"))
    margins = np.arange(start_m, stop_m + step_m / 2, step_m)
    logger.info(f"Sweeping {len(margins)} margin values ...")

    results = []
    for margin in tqdm(margins, desc="Margin sweep"):
        scores, labels, esc_rate = apply_routing(
            scene_clips, scene_labels, margin=float(margin)
        )
        auc = compute_auc(scores, labels)
        results.append({"margin": round(float(margin), 4),
                        "escalation_rate": round(esc_rate, 4),
                        "auc": round(auc, 4)})

    # ── Save JSON ─────────────────────────────────────────────────────────────
    summary = {
        "auc_small_only": round(auc_small_only, 4),
        "auc_large_only": round(auc_large_only, 4),
        "sweep": results,
    }
    json_path = out / "pareto_results.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Results saved -> {json_path}")

    # ── Plot ──────────────────────────────────────────────────────────────────
    esc  = [r["escalation_rate"] * 100 for r in results]
    aucs = [r["auc"] for r in results]

    fig, ax = plt.subplots(figsize=(7, 5))

    # Routing curve
    ax.plot(esc, aucs, "o-", color="steelblue", lw=2, ms=4,
            label="Routing (small→large)")

    # Baselines
    ax.axhline(auc_small_only, color="green",  ls="--", lw=1.5,
               label=f"Small-only  (AUC={auc_small_only:.4f})")
    ax.axhline(auc_large_only, color="orange", ls="--", lw=1.5,
               label=f"Large-only  (AUC={auc_large_only:.4f})")

    ax.set_xlabel("Escalation rate (%)")
    ax.set_ylabel("Frame-level AUC-ROC")
    ax.set_title("Cost-Quality Pareto Curve — UCSD Ped2")
    ax.legend(loc="lower right")
    ax.set_xlim(-2, 102)
    ax.grid(True, alpha=0.3)

    plot_path = out / "pareto_curve.png"
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    plt.close()
    logger.info(f"Pareto curve saved -> {plot_path}")

    # ── Print summary table ───────────────────────────────────────────────────
    print("\n" + "=" * 55)
    print("  Pareto Sweep — UCSD Ped2")
    print("=" * 55)
    print(f"  Small-only  AUC : {auc_small_only:.4f}  (0% escalation)")
    print(f"  Large-only  AUC : {auc_large_only:.4f}  (100% escalation)")
    print("-" * 55)
    print(f"  {'Margin':>8}  {'Escal %':>8}  {'AUC':>8}")
    print("  " + "-" * 33)
    for r in results:
        flag = " <-- best" if r["auc"] == max(x["auc"] for x in results) else ""
        print(f"  {r['margin']:8.2f}  {r['escalation_rate']*100:8.1f}  {r['auc']:8.4f}{flag}")
    print("=" * 55 + "\n")


if __name__ == "__main__":
    main()
