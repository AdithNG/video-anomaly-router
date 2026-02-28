"""
Large spatiotemporal autoencoder.
Uses a deeper 3-D CNN encoder + optional Swin-style attention bottleneck
for higher-fidelity reconstruction on ambiguous/OOD clips.
"""

import logging
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class ResBlock3D(nn.Module):
    """3-D residual block with optional downsampling."""

    def __init__(self, in_ch: int, out_ch: int, stride: Tuple = (1, 1, 1)):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm3d(out_ch),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm3d(out_ch),
        )
        self.skip = (
            nn.Sequential(
                nn.Conv3d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm3d(out_ch),
            )
            if in_ch != out_ch or stride != (1, 1, 1)
            else nn.Identity()
        )
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        return self.act(self.conv(x) + self.skip(x))


class SpatiotemporalAttention(nn.Module):
    """
    Lightweight multi-head self-attention over the flattened spatial+temporal
    positions of the bottleneck feature map.
    Used as a cheap substitute for a full Swin transformer.
    """

    def __init__(self, dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C, T, H, W) → same shape."""
        B, C, T, H, W = x.shape
        tokens = x.flatten(2).permute(0, 2, 1)        # (B, T*H*W, C)
        attn_out, _ = self.attn(tokens, tokens, tokens)
        tokens = self.norm(tokens + attn_out)
        return tokens.permute(0, 2, 1).view(B, C, T, H, W)


# ---------------------------------------------------------------------------
# Large autoencoder
# ---------------------------------------------------------------------------

class LargeAutoencoder(nn.Module):
    """
    Deep spatiotemporal autoencoder with residual blocks and an attention
    bottleneck for high-fidelity reconstruction.

    Input:  (B, C, T, H, W)
    Output: reconstruction (B, C, T, H, W), mu (B, latent_dim), log_var (B, latent_dim)
    """

    def __init__(
        self,
        in_channels: int = 3,
        base_ch: int = 64,
        latent_dim: int = 1024,
        clip_len: int = 16,
        frame_size: int = 64,
        num_heads: int = 8,
    ):
        super().__init__()
        self.latent_dim = latent_dim

        # ── Encoder ──────────────────────────────────────────────────────────
        self.enc1 = ResBlock3D(in_channels, base_ch,    stride=(1,1,1))    # (B, 64,  T,   H,   W)
        self.enc2 = ResBlock3D(base_ch,     base_ch*2,  stride=(2,2,2))    # (B, 128, T/2, H/2, W/2)
        self.enc3 = ResBlock3D(base_ch*2,   base_ch*4,  stride=(1,2,2))    # (B, 256, T/2, H/4, W/4)
        self.enc4 = ResBlock3D(base_ch*4,   base_ch*8,  stride=(1,2,2))    # (B, 512, T/2, H/8, W/8)
        self.enc5 = ResBlock3D(base_ch*8,   base_ch*8,  stride=(1,2,2))    # (B, 512, T/2, H/16,W/16)

        # Attention bottleneck
        self.attn = SpatiotemporalAttention(base_ch * 8, num_heads=num_heads)

        t_enc = clip_len // 2
        s_enc = frame_size // 16
        self._enc_shape = (base_ch * 8, t_enc, s_enc, s_enc)
        flat = base_ch * 8 * t_enc * s_enc * s_enc

        self.fc_mu      = nn.Linear(flat, latent_dim)
        self.fc_log_var = nn.Linear(flat, latent_dim)
        self.fc_decode  = nn.Linear(latent_dim, flat)

        # ── Decoder ──────────────────────────────────────────────────────────
        self.dec5 = nn.Sequential(
            nn.ConvTranspose3d(base_ch*8, base_ch*8, 4, stride=(1,2,2), padding=1, bias=False),
            nn.BatchNorm3d(base_ch*8), nn.ReLU(inplace=True),
        )
        self.dec4 = nn.Sequential(
            nn.ConvTranspose3d(base_ch*8, base_ch*4, 4, stride=(1,2,2), padding=1, bias=False),
            nn.BatchNorm3d(base_ch*4), nn.ReLU(inplace=True),
        )
        self.dec3 = nn.Sequential(
            nn.ConvTranspose3d(base_ch*4, base_ch*2, 4, stride=(1,2,2), padding=1, bias=False),
            nn.BatchNorm3d(base_ch*2), nn.ReLU(inplace=True),
        )
        self.dec2 = nn.Sequential(
            nn.ConvTranspose3d(base_ch*2, base_ch, 4, stride=(2,2,2), padding=1, bias=False),
            nn.BatchNorm3d(base_ch), nn.ReLU(inplace=True),
        )
        self.dec1 = nn.Sequential(
            nn.Conv3d(base_ch, in_channels, 3, padding=1),
            nn.Tanh(),
        )

    # ── Forward ──────────────────────────────────────────────────────────────

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.enc1(x)
        h = self.enc2(h)
        h = self.enc3(h)
        h = self.enc4(h)
        h = self.enc5(h)
        h = self.attn(h)
        h = h.flatten(1)
        return self.fc_mu(h), self.fc_log_var(h)

    def decode(self, z: torch.Tensor, target_shape: Tuple) -> torch.Tensor:
        B, C, T, H, W = target_shape
        h = self.fc_decode(z).view(B, *self._enc_shape)
        h = self.dec5(h)
        h = self.dec4(h)
        h = self.dec3(h)
        h = self.dec2(h)
        recon = self.dec1(h)
        if recon.shape != target_shape:
            recon = F.interpolate(recon, size=(T, H, W), mode="trilinear", align_corners=False)
        return recon

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, log_var = self.encode(x)
        if self.training:
            std = torch.exp(0.5 * log_var)
            z = mu + std * torch.randn_like(std)
        else:
            z = mu
        recon = self.decode(z, x.shape)
        return recon, mu, log_var


# ---------------------------------------------------------------------------
# Loss (shared with small AE)
# ---------------------------------------------------------------------------

def large_ae_loss(
    recon: torch.Tensor,
    x: torch.Tensor,
    log_var: torch.Tensor,
    kl_weight: float = 1e-4,
) -> Tuple[torch.Tensor, dict]:
    recon_loss = F.mse_loss(recon, x, reduction="mean")
    kl_loss = -0.5 * torch.mean(1 + log_var - log_var.exp())
    total = recon_loss + kl_weight * kl_loss
    return total, {"recon": recon_loss.item(), "kl": kl_loss.item(), "total": total.item()}


# ---------------------------------------------------------------------------
# Anomaly scoring
# ---------------------------------------------------------------------------

@torch.no_grad()
def anomaly_score(
    model: LargeAutoencoder,
    clip: torch.Tensor,
) -> Tuple[float, torch.Tensor, float]:
    """
    Returns (score, embedding, uncertainty) — same interface as small AE.
    """
    if clip.dim() == 4:
        clip = clip.unsqueeze(0)
    recon, mu, log_var = model(clip)
    score = F.mse_loss(recon, clip, reduction="mean").item()
    embedding = mu.squeeze(0)
    uncertainty = log_var.exp().mean().item()
    return score, embedding, uncertainty
