"""
Routing module.
Decides whether a clip should be escalated from the small to the large model.

Routing signals (from the proposal):
  1. Gray-zone distance  – proximity of recon error to the decision threshold
  2. OOD score           – cosine distance of the embedding from the training centroid
  3. Temporal instability – variance of recon error across overlapping windows
  4. Scene-cut flag       – detected scene cut in the clip
"""

import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Optional

import torch

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Routing decision data class
# ---------------------------------------------------------------------------

@dataclass
class RoutingDecision:
    escalate: bool
    gray_zone_dist: float    # |score - threshold| / threshold
    ood_score: float         # cosine distance from training centroid
    temporal_instability: float
    scene_cut: bool
    small_score: float
    reason: str              # human-readable explanation


# ---------------------------------------------------------------------------
# Token bucket — enforces escalation budget
# ---------------------------------------------------------------------------

class TokenBucket:
    """
    Simple token-bucket rate limiter for escalation budget control.
    `capacity` = max tokens, `refill_rate` = tokens added per call.
    """

    def __init__(self, capacity: int = 100, refill_rate: float = 0.1):
        self.capacity = capacity
        self.refill_rate = refill_rate
        self._tokens = float(capacity)

    def consume(self) -> bool:
        """Returns True if a token was available (escalation allowed)."""
        self._tokens = min(self.capacity, self._tokens + self.refill_rate)
        if self._tokens >= 1.0:
            self._tokens -= 1.0
            return True
        return False

    @property
    def fill_ratio(self) -> float:
        return self._tokens / self.capacity


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

class Router:
    """
    Stateful routing module that combines multiple signals to decide
    whether a clip should be escalated to the large model.

    Args:
        threshold:             Reconstruction-error anomaly threshold.
        gray_zone_margin:      Fraction of threshold defining the gray zone.
                               Clips within ±margin are ambiguous.
        ood_threshold:         Cosine-distance cutoff for OOD detection.
        instability_threshold: Temporal variance cutoff.
        scene_cut_escalate:    Always escalate clips with scene cuts.
        budget_capacity:       Token-bucket capacity (total escalations allowed).
        budget_refill:         Tokens refilled per clip (controls escalation rate).
        history_len:           Number of recent scores kept for temporal analysis.
    """

    def __init__(
        self,
        threshold: float = 0.02,
        gray_zone_margin: float = 0.3,
        ood_threshold: float = 0.4,
        instability_threshold: float = 0.005,
        scene_cut_escalate: bool = True,
        budget_capacity: int = 200,
        budget_refill: float = 0.2,
        history_len: int = 10,
    ):
        self.threshold = threshold
        self.gray_zone_margin = gray_zone_margin
        self.ood_threshold = ood_threshold
        self.instability_threshold = instability_threshold
        self.scene_cut_escalate = scene_cut_escalate
        self.bucket = TokenBucket(budget_capacity, budget_refill)
        self._score_history: Deque[float] = deque(maxlen=history_len)
        self._centroid: Optional[torch.Tensor] = None   # updated during calibration

    # ── Centroid management ──────────────────────────────────────────────────

    def set_centroid(self, centroid: torch.Tensor):
        """Set the training-set embedding centroid for OOD scoring."""
        self._centroid = centroid.detach().cpu()

    def update_centroid(self, embeddings: torch.Tensor):
        """
        Online update of the centroid as a running mean.
        embeddings: (N, latent_dim)
        """
        new_mean = embeddings.detach().cpu().mean(0)
        if self._centroid is None:
            self._centroid = new_mean
        else:
            self._centroid = 0.9 * self._centroid + 0.1 * new_mean

    # ── Individual signal computations ───────────────────────────────────────

    def _gray_zone_dist(self, score: float) -> float:
        """Normalised distance from threshold (0 = at threshold, 1 = far away)."""
        return abs(score - self.threshold) / (self.threshold + 1e-8)

    def _ood_score(self, embedding: torch.Tensor) -> float:
        """Cosine distance from training centroid. 0 = identical, 2 = opposite."""
        if self._centroid is None:
            return 0.0
        e = embedding.detach().cpu().float()
        c = self._centroid.float()
        cos_sim = torch.nn.functional.cosine_similarity(e.unsqueeze(0), c.unsqueeze(0)).item()
        return 1.0 - cos_sim   # cosine distance

    def _temporal_instability(self, score: float) -> float:
        """Variance of recent scores (higher = less stable)."""
        self._score_history.append(score)
        if len(self._score_history) < 2:
            return 0.0
        hist = torch.tensor(list(self._score_history))
        return hist.var().item()

    # ── Main routing decision ────────────────────────────────────────────────

    def route(
        self,
        small_score: float,
        embedding: torch.Tensor,
        uncertainty: float,
        scene_cut: bool = False,
    ) -> RoutingDecision:
        """
        Evaluate all routing signals and return a RoutingDecision.

        Args:
            small_score:  Reconstruction MSE from the small model.
            embedding:    Latent vector from the small model (latent_dim,).
            uncertainty:  Mean variance proxy from the small model.
            scene_cut:    Whether a scene cut was detected in this clip.
        """
        gz_dist = self._gray_zone_dist(small_score)
        ood     = self._ood_score(embedding)
        instab  = self._temporal_instability(small_score)

        in_gray_zone  = gz_dist < self.gray_zone_margin
        is_ood        = ood > self.ood_threshold
        is_unstable   = instab > self.instability_threshold
        has_scene_cut = scene_cut and self.scene_cut_escalate

        should_escalate = in_gray_zone or is_ood or is_unstable or has_scene_cut

        reason_parts = []
        if in_gray_zone:
            reason_parts.append(f"gray-zone(dist={gz_dist:.3f})")
        if is_ood:
            reason_parts.append(f"OOD(dist={ood:.3f})")
        if is_unstable:
            reason_parts.append(f"temporal-instability({instab:.5f})")
        if has_scene_cut:
            reason_parts.append("scene-cut")

        # Apply budget constraint
        if should_escalate and not self.bucket.consume():
            should_escalate = False
            reason_parts.append("[budget-exceeded]")
            logger.debug("Escalation suppressed — budget exhausted.")

        reason = ", ".join(reason_parts) if reason_parts else "confident-small"

        return RoutingDecision(
            escalate=should_escalate,
            gray_zone_dist=gz_dist,
            ood_score=ood,
            temporal_instability=instab,
            scene_cut=scene_cut,
            small_score=small_score,
            reason=reason,
        )

    # ── Threshold calibration ────────────────────────────────────────────────

    def calibrate_threshold(self, normal_scores: list, quantile: float = 0.95):
        """
        Set threshold from the empirical quantile of normal-data scores.
        Call this after computing scores on the validation (normal) set.
        """
        import torch
        t = torch.tensor(normal_scores)
        self.threshold = torch.quantile(t, quantile).item()
        logger.info(f"Calibrated threshold (q={quantile}): {self.threshold:.6f}")
