"""
Pseudo-anomaly augmentations for self-supervised auxiliary head training.

Generates synthetic negatives from normal clips at training time without
requiring real anomaly labels. Three strategies:
  - random_erasing    : mask a random spatiotemporal region with noise
  - temporal_shuffle  : permute frames in time (disrupts motion coherence)
  - spatial_jitter    : apply per-frame random translation

make_pseudo_anomaly_batch() applies a random strategy to every clip in a batch.
"""

import random

import torch
import torch.nn.functional as F


def random_erasing(
    clip: torch.Tensor,
    p: float = 0.9,
    area_lo: float = 0.05,
    area_hi: float = 0.40,
    time_lo: float = 0.25,
    time_hi: float = 0.75,
) -> torch.Tensor:
    """Mask a random spatiotemporal region with Gaussian noise."""
    if random.random() > p:
        return clip.clone()
    clip = clip.clone()
    C, T, H, W = clip.shape

    t_start = random.randint(0, max(0, T - 1))
    t_len   = max(1, int(random.uniform(time_lo, time_hi) * T))
    t_end   = min(t_start + t_len, T)

    erase_area = random.uniform(area_lo, area_hi) * H * W
    aspect     = random.uniform(0.3, 3.0)
    eh = max(1, min(int((erase_area * aspect) ** 0.5), H))
    ew = max(1, min(int((erase_area / max(aspect, 1e-6)) ** 0.5), W))

    y = random.randint(0, H - eh)
    x = random.randint(0, W - ew)
    clip[:, t_start:t_end, y:y + eh, x:x + ew] = torch.randn(
        C, t_end - t_start, eh, ew, device=clip.device
    )
    return clip


def temporal_shuffle(clip: torch.Tensor, p: float = 0.95) -> torch.Tensor:
    """Randomly permute frames — breaks temporal coherence."""
    if random.random() > p:
        return clip.clone()
    C, T, H, W = clip.shape
    perm = torch.randperm(T, device=clip.device)
    return clip[:, perm].contiguous()


def spatial_jitter(
    clip: torch.Tensor,
    p: float = 0.85,
    max_shift: float = 0.15,
) -> torch.Tensor:
    """
    Apply a random (but fixed per-clip) spatial translation.
    max_shift is a fraction of H / W.
    """
    if random.random() > p:
        return clip.clone()
    C, T, H, W = clip.shape

    dx = random.uniform(-max_shift, max_shift)
    dy = random.uniform(-max_shift, max_shift)
    # Build affine grid for translation
    theta = torch.tensor(
        [[1.0, 0.0, dx], [0.0, 1.0, dy]],
        dtype=torch.float32,
        device=clip.device,
    ).unsqueeze(0)  # (1, 2, 3)

    # Rearrange to (T, C, H, W) for grid_sample, then back
    frames = clip.permute(1, 0, 2, 3)  # (T, C, H, W)
    grid   = F.affine_grid(theta.expand(T, -1, -1), (T, C, H, W), align_corners=False)
    jittered = F.grid_sample(frames, grid, align_corners=False, padding_mode="border")
    return jittered.permute(1, 0, 2, 3).contiguous()  # (C, T, H, W)


_AUGMENTATIONS = [random_erasing, temporal_shuffle, spatial_jitter]


def make_pseudo_anomaly(clip: torch.Tensor) -> torch.Tensor:
    """Apply one randomly chosen augmentation to a single (C, T, H, W) clip."""
    aug = random.choice(_AUGMENTATIONS)
    return aug(clip)


def make_pseudo_anomaly_batch(clips: torch.Tensor) -> torch.Tensor:
    """
    Apply random augmentations to a batch of clips (B, C, T, H, W).
    Each clip in the batch gets an independently sampled augmentation.
    Operates on CPU or GPU (affine_grid stays on same device as input).
    """
    B = clips.shape[0]
    out = []
    for i in range(B):
        out.append(make_pseudo_anomaly(clips[i]))
    return torch.stack(out, dim=0)
