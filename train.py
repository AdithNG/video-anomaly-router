"""
Training script for the small and large spatiotemporal autoencoders.

Usage:
    python train.py --model small --data data/train --epochs 50
    python train.py --model large --data data/train --epochs 30
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# ── Project imports ──────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent / "src"))
from device import get_device, safe_to_device
from preprocessing import VideoClipDataset
from models.small_autoencoder import SmallAutoencoder, small_ae_loss
from models.large_autoencoder import LargeAutoencoder, large_ae_loss

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Train small or large autoencoder")
    p.add_argument("--model",   choices=["small", "large"], default="small")
    p.add_argument("--data",    default="data/train", help="Path to training frames")
    p.add_argument("--epochs",  type=int, default=50)
    p.add_argument("--batch",   type=int, default=8)
    p.add_argument("--lr",      type=float, default=1e-3)
    p.add_argument("--clip-len",type=int, default=16)
    p.add_argument("--frame-size", type=int, default=64)
    p.add_argument("--kl-weight",  type=float, default=1e-4)
    p.add_argument("--save-dir",   default="checkpoints")
    p.add_argument("--workers",    type=int, default=0,
                   help="DataLoader workers (0 = main process, safe on Windows)")
    return p.parse_args()


def build_model(name: str, clip_len: int, frame_size: int, device: torch.device):
    if name == "small":
        model = SmallAutoencoder(clip_len=clip_len, frame_size=frame_size)
    else:
        model = LargeAutoencoder(clip_len=clip_len, frame_size=frame_size)
    return model.to(device)


def train_one_epoch(model, loader, optimizer, loss_fn, device, kl_weight):
    model.train()
    totals = {"recon": 0.0, "kl": 0.0, "total": 0.0}
    n = 0

    for clips, _ in tqdm(loader, desc="  train", leave=False):
        clips = safe_to_device(clips, device)
        optimizer.zero_grad()
        recon, mu, log_var = model(clips)
        loss, info = loss_fn(recon, clips, log_var, kl_weight)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        for k in totals:
            totals[k] += info[k]
        n += 1

    return {k: v / max(n, 1) for k, v in totals.items()}


@torch.no_grad()
def validate(model, loader, loss_fn, device, kl_weight):
    model.eval()
    totals = {"recon": 0.0, "kl": 0.0, "total": 0.0}
    n = 0

    for clips, _ in tqdm(loader, desc="  val  ", leave=False):
        clips = safe_to_device(clips, device)
        recon, mu, log_var = model(clips)
        _, info = loss_fn(recon, clips, log_var, kl_weight)
        for k in totals:
            totals[k] += info[k]
        n += 1

    return {k: v / max(n, 1) for k, v in totals.items()}


def main():
    args = parse_args()
    device = get_device()
    os.makedirs(args.save_dir, exist_ok=True)

    # ── Dataset ──────────────────────────────────────────────────────────────
    logger.info(f"Loading dataset from: {args.data}")
    dataset = VideoClipDataset(
        root=args.data,
        clip_len=args.clip_len,
        stride=args.clip_len // 2,
        frame_size=(args.frame_size, args.frame_size),
    )

    if len(dataset) == 0:
        logger.error("No clips found. Check --data path and ensure frame PNGs exist.")
        sys.exit(1)

    n_val = max(1, int(0.1 * len(dataset)))
    n_train = len(dataset) - n_val
    train_set, val_set = torch.utils.data.random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(train_set, batch_size=args.batch, shuffle=True,
                              num_workers=args.workers, pin_memory=(device.type == "cuda"))
    val_loader   = DataLoader(val_set,   batch_size=args.batch, shuffle=False,
                              num_workers=args.workers, pin_memory=(device.type == "cuda"))

    logger.info(f"Train clips: {n_train}  |  Val clips: {n_val}")

    # ── Model ─────────────────────────────────────────────────────────────────
    model = build_model(args.model, args.clip_len, args.frame_size, device)
    loss_fn = small_ae_loss if args.model == "small" else large_ae_loss
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    logger.info(f"Model: {args.model}  |  Params: {n_params:.2f}M  |  Device: {device}")

    # ── Training loop ─────────────────────────────────────────────────────────
    best_val_loss = float("inf")
    save_path = Path(args.save_dir) / f"{args.model}_ae_best.pt"

    for epoch in range(1, args.epochs + 1):
        train_info = train_one_epoch(model, train_loader, optimizer, loss_fn, device, args.kl_weight)
        val_info   = validate(model, val_loader, loss_fn, device, args.kl_weight)
        scheduler.step()

        logger.info(
            f"Epoch {epoch:3d}/{args.epochs}  "
            f"train: recon={train_info['recon']:.5f} kl={train_info['kl']:.5f}  |  "
            f"val: recon={val_info['recon']:.5f} kl={val_info['kl']:.5f}"
        )

        if val_info["total"] < best_val_loss:
            best_val_loss = val_info["total"]
            torch.save(
                {"epoch": epoch, "model_state": model.state_dict(),
                 "val_loss": best_val_loss, "args": vars(args)},
                save_path,
            )
            logger.info(f"  ✓ Saved best checkpoint → {save_path}")

    logger.info("Training complete.")


if __name__ == "__main__":
    main()
