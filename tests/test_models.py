"""
Tests for src/models/small_autoencoder.py and src/models/large_autoencoder.py

Covers:
  - Forward pass shapes (recon, mu, log_var)
  - Training vs inference mode (reparameterisation)
  - anomaly_score() return types and value ranges
  - Loss functions return finite values
  - Models move correctly between CPU and GPU
"""

import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from device import get_device, safe_to_device
from models.small_autoencoder import (
    SmallAutoencoder,
    anomaly_score as small_anomaly_score,
    small_ae_loss,
)
from models.large_autoencoder import (
    LargeAutoencoder,
    anomaly_score as large_anomaly_score,
    large_ae_loss,
)

# ---------------------------------------------------------------------------
# Shared fixtures / constants
# ---------------------------------------------------------------------------

DEVICE = get_device(verbose=False)
BATCH  = 2
C, T, H, W = 3, 16, 64, 64   # standard clip shape


def _fake_clip(batch: int = BATCH) -> torch.Tensor:
    """Return a random (B, C, T, H, W) tensor on DEVICE."""
    return safe_to_device(torch.randn(batch, C, T, H, W), DEVICE)


# ---------------------------------------------------------------------------
# SmallAutoencoder
# ---------------------------------------------------------------------------

class TestSmallAutoencoder:
    def setup_method(self):
        self.model = SmallAutoencoder(clip_len=T, frame_size=H).to(DEVICE)

    # ── Shape checks ────────────────────────────────────────────────────────

    def test_forward_recon_shape(self):
        """Reconstruction must match input shape exactly."""
        clip = _fake_clip()
        recon, _, _ = self.model(clip)
        assert recon.shape == clip.shape, f"Expected {clip.shape}, got {recon.shape}"

    def test_forward_mu_shape(self):
        """mu must be (B, latent_dim)."""
        clip = _fake_clip()
        _, mu, _ = self.model(clip)
        assert mu.shape == (BATCH, self.model.latent_dim)

    def test_forward_log_var_shape(self):
        """log_var must be (B, latent_dim)."""
        clip = _fake_clip()
        _, _, log_var = self.model(clip)
        assert log_var.shape == (BATCH, self.model.latent_dim)

    def test_outputs_are_finite(self):
        """No NaN or Inf in any output tensor."""
        clip = _fake_clip()
        recon, mu, log_var = self.model(clip)
        for tensor, name in [(recon, "recon"), (mu, "mu"), (log_var, "log_var")]:
            assert torch.isfinite(tensor).all(), f"{name} contains non-finite values"

    # ── Training vs eval ────────────────────────────────────────────────────

    def test_eval_mode_uses_mean(self):
        """In eval mode two forward passes on the same input should be identical."""
        self.model.eval()
        clip = _fake_clip()
        with torch.no_grad():
            recon1, mu1, _ = self.model(clip)
            recon2, mu2, _ = self.model(clip)
        assert torch.allclose(mu1, mu2), "mu should be deterministic in eval mode"
        assert torch.allclose(recon1, recon2), "recon should be deterministic in eval mode"

    def test_train_mode_is_stochastic(self):
        """In train mode (reparameterisation) successive passes should differ."""
        self.model.train()
        clip = _fake_clip()
        recon1, _, _ = self.model(clip)
        recon2, _, _ = self.model(clip)
        # With random noise they should almost certainly differ
        assert not torch.allclose(recon1, recon2), "recon should be stochastic in train mode"

    # ── Device placement ────────────────────────────────────────────────────

    def test_output_on_correct_device(self):
        """All outputs must reside on the same device as the input."""
        clip = _fake_clip()
        recon, mu, log_var = self.model(clip)
        for t, name in [(recon, "recon"), (mu, "mu"), (log_var, "log_var")]:
            assert t.device.type == DEVICE.type, f"{name} is on wrong device"

    # ── Loss ────────────────────────────────────────────────────────────────

    def test_loss_is_finite(self):
        """small_ae_loss must return a finite scalar."""
        clip = _fake_clip()
        recon, _, log_var = self.model(clip)
        loss, info = small_ae_loss(recon, clip, log_var)
        assert torch.isfinite(loss), "Loss is not finite"
        assert all(torch.isfinite(torch.tensor(v)) for v in info.values())

    def test_loss_keys(self):
        """Loss info dict must contain 'recon', 'kl', 'total'."""
        clip = _fake_clip()
        recon, _, log_var = self.model(clip)
        _, info = small_ae_loss(recon, clip, log_var)
        assert set(info.keys()) == {"recon", "kl", "total"}

    def test_loss_decreases_with_perfect_recon(self):
        """If recon == input, MSE component should be ~0."""
        clip = _fake_clip()
        _, _, log_var = self.model(clip)
        loss, info = small_ae_loss(clip, clip, log_var)   # perfect recon
        assert info["recon"] < 1e-6

    # ── anomaly_score ────────────────────────────────────────────────────────

    def test_anomaly_score_returns_three_values(self):
        self.model.eval()
        clip = _fake_clip(batch=1).squeeze(0)   # (C, T, H, W)
        result = small_anomaly_score(self.model, clip)
        assert len(result) == 3

    def test_anomaly_score_scalar_score(self):
        self.model.eval()
        clip = _fake_clip(batch=1).squeeze(0)
        score, _, _ = small_anomaly_score(self.model, clip)
        assert isinstance(score, float) and score >= 0.0

    def test_anomaly_score_embedding_shape(self):
        self.model.eval()
        clip = _fake_clip(batch=1).squeeze(0)
        _, embedding, _ = small_anomaly_score(self.model, clip)
        assert embedding.shape == (self.model.latent_dim,)

    def test_anomaly_score_uncertainty_positive(self):
        self.model.eval()
        clip = _fake_clip(batch=1).squeeze(0)
        _, _, uncertainty = small_anomaly_score(self.model, clip)
        assert isinstance(uncertainty, float) and uncertainty > 0.0

    def test_anomaly_score_accepts_batch_input(self):
        """anomaly_score should also work when a batch is passed."""
        self.model.eval()
        clip = _fake_clip(batch=2)   # (B, C, T, H, W)
        score, emb, unc = small_anomaly_score(self.model, clip)
        assert isinstance(score, float)


# ---------------------------------------------------------------------------
# LargeAutoencoder
# ---------------------------------------------------------------------------

class TestLargeAutoencoder:
    def setup_method(self):
        self.model = LargeAutoencoder(clip_len=T, frame_size=H).to(DEVICE)

    def test_forward_recon_shape(self):
        clip = _fake_clip()
        recon, _, _ = self.model(clip)
        assert recon.shape == clip.shape

    def test_forward_mu_shape(self):
        clip = _fake_clip()
        _, mu, _ = self.model(clip)
        assert mu.shape == (BATCH, self.model.latent_dim)

    def test_outputs_are_finite(self):
        clip = _fake_clip()
        recon, mu, log_var = self.model(clip)
        for tensor, name in [(recon, "recon"), (mu, "mu"), (log_var, "log_var")]:
            assert torch.isfinite(tensor).all(), f"{name} contains non-finite values"

    def test_eval_deterministic(self):
        self.model.eval()
        clip = _fake_clip()
        with torch.no_grad():
            _, mu1, _ = self.model(clip)
            _, mu2, _ = self.model(clip)
        assert torch.allclose(mu1, mu2)

    def test_loss_is_finite(self):
        clip = _fake_clip()
        recon, _, log_var = self.model(clip)
        loss, info = large_ae_loss(recon, clip, log_var)
        assert torch.isfinite(loss)

    def test_loss_keys(self):
        clip = _fake_clip()
        recon, _, log_var = self.model(clip)
        _, info = large_ae_loss(recon, clip, log_var)
        assert set(info.keys()) == {"recon", "kl", "total"}

    def test_anomaly_score_types(self):
        self.model.eval()
        clip = _fake_clip(batch=1).squeeze(0)
        score, emb, unc = large_anomaly_score(self.model, clip)
        assert isinstance(score, float)
        assert isinstance(emb, torch.Tensor)
        assert isinstance(unc, float)

    def test_large_has_more_params_than_small(self):
        """Sanity check: large model should have more parameters."""
        small = SmallAutoencoder(clip_len=T, frame_size=H)
        large = LargeAutoencoder(clip_len=T, frame_size=H)
        small_params = sum(p.numel() for p in small.parameters())
        large_params = sum(p.numel() for p in large.parameters())
        assert large_params > small_params, (
            f"Large ({large_params}) should have more params than small ({small_params})"
        )
