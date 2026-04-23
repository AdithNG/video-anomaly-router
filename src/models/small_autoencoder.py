"""
Small spatiotemporal autoencoder.
Uses 3-D convolutions to encode/decode (C, T, H, W) clips.
Produces: reconstruction, latent embedding, uncertainty proxy.
"""

import logging
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class ConvBlock3D(nn.Module):
    """Conv3D → BatchNorm → LeakyReLU."""

    def __init__(self, in_ch: int, out_ch: int, kernel: int = 3,
                 stride: int = 1, padding: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel, stride=stride, padding=padding, bias=False),
            nn.BatchNorm3d(out_ch),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class DeconvBlock3D(nn.Module):
    """ConvTranspose3D → BatchNorm → ReLU."""

    def __init__(self, in_ch: int, out_ch: int, kernel: int = 4,
                 stride: int = 2, padding: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            nn.ConvTranspose3d(in_ch, out_ch, kernel, stride=stride, padding=padding, bias=False),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class SmallAutoencoder(nn.Module):
    """
    Lightweight 3-D convolutional autoencoder.

    Input:  (B, C, T, H, W)  e.g. (B, 3, 16, 64, 64)
    Output: reconstruction   (B, C, T, H, W)
            latent           (B, latent_dim)
            log_var          (B, latent_dim)  — uncertainty proxy

    When use_aux_head=True an auxiliary binary-classification head is attached
    to the encoder bottleneck.  It is trained to distinguish normal clips from
    pseudo-anomalous augmentations and its sigmoid output becomes an extra
    anomaly signal at inference time.
    """

    def __init__(
        self,
        in_channels: int = 3,
        base_ch: int = 32,
        latent_dim: int = 512,
        clip_len: int = 16,
        frame_size: int = 64,
        use_aux_head: bool = False,
    ):
        super().__init__()
        self.latent_dim = latent_dim

        # ── Encoder ──────────────────────────────────────────────────────────
        # Each stage halves spatial dims; temporal dim also halved at first stage.
        self.enc1 = ConvBlock3D(in_channels, base_ch,    stride=1)         # (B, 32,  T,   H,   W)
        self.enc2 = ConvBlock3D(base_ch,     base_ch*2,  stride=(2,2,2))   # (B, 64,  T/2, H/2, W/2)
        self.enc3 = ConvBlock3D(base_ch*2,   base_ch*4,  stride=(1,2,2))   # (B, 128, T/2, H/4, W/4)
        self.enc4 = ConvBlock3D(base_ch*4,   base_ch*8,  stride=(1,2,2))   # (B, 256, T/2, H/8, W/8)

        # Compute flattened size for FC layers
        t_enc = clip_len // 2
        s_enc = frame_size // 8
        self._enc_shape = (base_ch * 8, t_enc, s_enc, s_enc)
        flat = base_ch * 8 * t_enc * s_enc * s_enc

        self.fc_mu      = nn.Linear(flat, latent_dim)
        self.fc_log_var = nn.Linear(flat, latent_dim)   # for uncertainty estimation
        self.fc_decode  = nn.Linear(latent_dim, flat)

        # ── Optional auxiliary head (normal vs. pseudo-anomaly classifier) ────
        if use_aux_head:
            self.aux_head = nn.Sequential(
                nn.Linear(latent_dim, 128),
                nn.ReLU(inplace=True),
                nn.Linear(128, 1),
            )

        # ── Decoder ──────────────────────────────────────────────────────────
        self.dec4 = DeconvBlock3D(base_ch*8, base_ch*4, stride=(1,2,2))
        self.dec3 = DeconvBlock3D(base_ch*4, base_ch*2, stride=(1,2,2))
        self.dec2 = DeconvBlock3D(base_ch*2, base_ch,   stride=(2,2,2))
        self.dec1 = nn.Sequential(
            nn.Conv3d(base_ch, in_channels, kernel_size=3, padding=1),
            nn.Tanh(),
        )

    # ── Forward ──────────────────────────────────────────────────────────────

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (mu, log_var) — both (B, latent_dim)."""
        h = self.enc1(x)
        h = self.enc2(h)
        h = self.enc3(h)
        h = self.enc4(h)
        h = h.flatten(1)
        return self.fc_mu(h), self.fc_log_var(h)

    def decode(self, z: torch.Tensor, target_shape: Tuple) -> torch.Tensor:
        """Decode latent z → reconstruction matching target_shape."""
        B, C, T, H, W = target_shape
        h = self.fc_decode(z)
        h = h.view(B, *self._enc_shape)
        h = self.dec4(h)
        h = self.dec3(h)
        h = self.dec2(h)
        recon = self.dec1(h)
        # Ensure spatial dims match input exactly
        if recon.shape != target_shape:
            recon = F.interpolate(recon, size=(T, H, W), mode="trilinear", align_corners=False)
        return recon

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            recon   – (B, C, T, H, W) reconstruction
            mu      – (B, latent_dim) mean embedding
            log_var – (B, latent_dim) log variance (uncertainty proxy)
        """
        mu, log_var = self.encode(x)
        # During inference we use the mean; during training add noise (VAE-style)
        if self.training:
            std = torch.exp(0.5 * log_var)
            z = mu + std * torch.randn_like(std)
        else:
            z = mu
        recon = self.decode(z, x.shape)
        return recon, mu, log_var


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def small_ae_loss(
    recon: torch.Tensor,
    x: torch.Tensor,
    log_var: torch.Tensor,
    kl_weight: float = 1e-4,
    aux_logit: torch.Tensor = None,
    aux_labels: torch.Tensor = None,
    aux_weight: float = 0.1,
) -> Tuple[torch.Tensor, dict]:
    """
    Combined reconstruction + KL loss (VAE-style).

    Optional aux_logit / aux_labels add a BCE term from the pseudo-anomaly
    auxiliary head.  Pass both or neither; aux_weight scales the term.
    """
    recon_loss = F.mse_loss(recon, x, reduction="mean")
    kl_loss    = -0.5 * torch.mean(1 + log_var - log_var.exp())
    total      = recon_loss + kl_weight * kl_loss
    info       = {"recon": recon_loss.item(), "kl": kl_loss.item(), "total": total.item()}

    if aux_logit is not None and aux_labels is not None:
        aux_loss = F.binary_cross_entropy_with_logits(
            aux_logit.squeeze(1), aux_labels.float()
        )
        total = total + aux_weight * aux_loss
        info["aux"]   = aux_loss.item()
        info["total"] = total.item()

    return total, info


# ---------------------------------------------------------------------------
# Anomaly scoring
# ---------------------------------------------------------------------------

@torch.no_grad()
def anomaly_score(
    model: SmallAutoencoder,
    clip: torch.Tensor,
    n_passes: int = 1,
    return_aux: bool = False,
) -> Tuple:
    """
    Compute anomaly score for a single clip (C, T, H, W) or batch.

    Args:
        model:      SmallAutoencoder in eval mode.
        clip:       Tensor of shape (C, T, H, W) or (B, C, T, H, W).
        n_passes:   Stochastic forward passes (>1 only with dropout layers).
        return_aux: If True and model has an aux_head, also return the aux score.

    Returns (return_aux=False):
        score, embedding, uncertainty
    Returns (return_aux=True):
        score, embedding, uncertainty, aux_score
        aux_score is 0.0 when the model has no aux_head.
    """
    if clip.dim() == 4:
        clip = clip.unsqueeze(0)

    scores, embeddings, variances = [], [], []
    for _ in range(n_passes):
        recon, mu, log_var = model(clip)
        err = F.mse_loss(recon, clip, reduction="mean").item()
        scores.append(err)
        embeddings.append(mu)
        variances.append(log_var.exp().mean().item())

    score       = float(sum(scores) / len(scores))
    embedding   = embeddings[0].squeeze(0)   # (latent_dim,)
    uncertainty = float(sum(variances) / len(variances))

    if return_aux:
        if hasattr(model, "aux_head"):
            aux_logit = model.aux_head(embeddings[0])          # (1, 1)
            aux_score = float(torch.sigmoid(aux_logit).mean())
        else:
            aux_score = 0.0
        return score, embedding, uncertainty, aux_score

    return score, embedding, uncertainty
