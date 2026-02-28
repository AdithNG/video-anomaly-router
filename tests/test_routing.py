"""
Tests for src/routing.py

Covers:
  - TokenBucket: refill, consume, budget exhaustion
  - Router._gray_zone_dist(): correct normalisation
  - Router._ood_score(): 0 when no centroid, distance when centroid is set
  - Router._temporal_instability(): variance computation
  - Router.route(): escalation logic for each individual signal
  - Router.route(): budget enforcement suppresses escalation
  - Router.calibrate_threshold(): sets threshold from quantile
"""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from routing import Router, RoutingDecision, TokenBucket


# ---------------------------------------------------------------------------
# TokenBucket
# ---------------------------------------------------------------------------

class TestTokenBucket:
    def test_full_bucket_allows_consume(self):
        """A freshly created bucket with capacity > 0 should allow a consume."""
        bucket = TokenBucket(capacity=10, refill_rate=0.0)
        assert bucket.consume() is True

    def test_empty_bucket_denies_consume(self):
        """After draining all tokens, consume() must return False."""
        bucket = TokenBucket(capacity=2, refill_rate=0.0)
        bucket.consume()
        bucket.consume()
        assert bucket.consume() is False

    def test_refill_eventually_allows_consume(self):
        """With a positive refill rate, the bucket should refill after enough calls.

        Token trace with capacity=1, refill=0.5:
          consume() #1 — refill: min(1, 1.0+0.5)=1.0, spend → tokens=0.0 (True)
          consume() #2 — refill: 0.0+0.5=0.5 < 1.0                        (False)
          consume() #3 — refill: 0.5+0.5=1.0, spend → tokens=0.0          (True)
        """
        bucket = TokenBucket(capacity=1, refill_rate=0.5)
        bucket.consume()                          # drain the initial token
        assert bucket.consume() is False          # only 0.5 tokens — not enough
        assert bucket.consume() is True           # 1.0 tokens — refilled, consume

    def test_fill_ratio_starts_at_one(self):
        bucket = TokenBucket(capacity=10, refill_rate=0.0)
        assert bucket.fill_ratio == pytest.approx(1.0)

    def test_fill_ratio_after_draining(self):
        bucket = TokenBucket(capacity=10, refill_rate=0.0)
        bucket.consume()
        assert bucket.fill_ratio == pytest.approx(0.9)

    def test_capacity_not_exceeded(self):
        """Tokens must never exceed the bucket capacity."""
        bucket = TokenBucket(capacity=5, refill_rate=2.0)
        for _ in range(20):
            bucket.consume()  # repeatedly trigger refills
        assert bucket._tokens <= bucket.capacity


# ---------------------------------------------------------------------------
# Router internals
# ---------------------------------------------------------------------------

class TestRouterGrayZone:
    def setup_method(self):
        self.router = Router(threshold=0.02)

    def test_at_threshold_dist_is_zero(self):
        """A score exactly at the threshold should return distance 0."""
        dist = self.router._gray_zone_dist(0.02)
        assert dist == pytest.approx(0.0)

    def test_far_from_threshold_large_dist(self):
        """A score far above the threshold should return a large distance."""
        dist = self.router._gray_zone_dist(1.0)
        assert dist > 1.0

    def test_below_threshold_positive_dist(self):
        """Distance is always non-negative (absolute value)."""
        dist = self.router._gray_zone_dist(0.0)
        assert dist >= 0.0


class TestRouterOOD:
    def setup_method(self):
        self.router = Router()

    def test_ood_score_zero_without_centroid(self):
        """Without a centroid set, OOD score should be 0 (no escalation)."""
        emb = torch.randn(512)
        assert self.router._ood_score(emb) == pytest.approx(0.0)

    def test_ood_score_zero_for_identical_embedding(self):
        """Embedding identical to the centroid → cosine distance 0."""
        emb = torch.ones(512)
        self.router.set_centroid(emb.clone())
        assert self.router._ood_score(emb) == pytest.approx(0.0, abs=1e-5)

    def test_ood_score_max_for_opposite_embedding(self):
        """Embedding opposite to centroid → cosine distance ≈ 2."""
        centroid = torch.ones(512)
        opposite = -centroid
        self.router.set_centroid(centroid)
        dist = self.router._ood_score(opposite)
        assert dist == pytest.approx(2.0, abs=1e-4)

    def test_update_centroid_changes_centroid(self):
        """update_centroid() should shift the centroid toward new embeddings."""
        self.router.set_centroid(torch.zeros(512))
        self.router.update_centroid(torch.ones(1, 512))
        # centroid should now be > 0 in all dimensions
        assert (self.router._centroid > 0).all()


class TestRouterTemporalInstability:
    def setup_method(self):
        self.router = Router()

    def test_single_score_returns_zero(self):
        """With only one data point, variance is 0."""
        assert self.router._temporal_instability(0.01) == pytest.approx(0.0)

    def test_constant_scores_zero_variance(self):
        for _ in range(5):
            val = self.router._temporal_instability(0.01)
        assert val == pytest.approx(0.0, abs=1e-8)

    def test_varied_scores_nonzero_variance(self):
        values = [0.01, 0.5, 0.01, 0.5, 0.01]
        instab = 0.0
        for v in values:
            instab = self.router._temporal_instability(v)
        assert instab > 0.0


# ---------------------------------------------------------------------------
# Router.route()
# ---------------------------------------------------------------------------

class TestRouterRoute:
    """Integration tests for the full routing decision logic."""

    def _dummy_embedding(self, dim: int = 512) -> torch.Tensor:
        return torch.randn(dim)

    def test_returns_routing_decision(self):
        router = Router()
        decision = router.route(0.05, self._dummy_embedding(), 0.001)
        assert isinstance(decision, RoutingDecision)

    def test_no_escalation_when_far_from_threshold(self):
        """A score far above the threshold, no OOD, stable → no escalation."""
        router = Router(threshold=0.02, gray_zone_margin=0.3)
        emb = torch.ones(512)
        router.set_centroid(emb.clone())   # centroid == embedding → OOD = 0
        # score = 1.0 is far from threshold 0.02 → gray-zone dist >> 0.3
        decision = router.route(1.0, emb, 0.0, scene_cut=False)
        assert decision.escalate is False

    def test_escalation_in_gray_zone(self):
        """Score very close to threshold should trigger escalation."""
        router = Router(threshold=0.02, gray_zone_margin=0.5,
                        budget_capacity=1000, budget_refill=1.0)
        emb = torch.ones(512)
        router.set_centroid(emb.clone())
        # Score exactly at threshold → dist = 0 < margin → escalate
        decision = router.route(0.02, emb, 0.0, scene_cut=False)
        assert decision.escalate is True
        assert "gray-zone" in decision.reason

    def test_escalation_on_ood(self):
        """Embedding far from centroid should trigger escalation."""
        router = Router(threshold=0.02, ood_threshold=0.3,
                        budget_capacity=1000, budget_refill=1.0)
        centroid = torch.ones(512)
        router.set_centroid(centroid)
        opposite = -centroid   # cosine dist ≈ 2 >> 0.3
        # score = 1.0 keeps us out of gray zone; OOD alone triggers
        decision = router.route(1.0, opposite, 0.0, scene_cut=False)
        assert decision.escalate is True
        assert "OOD" in decision.reason

    def test_escalation_on_scene_cut(self):
        """A scene-cut flag should always trigger escalation (if budget allows)."""
        router = Router(scene_cut_escalate=True,
                        budget_capacity=1000, budget_refill=1.0)
        emb = torch.ones(512)
        router.set_centroid(emb.clone())
        decision = router.route(1.0, emb, 0.0, scene_cut=True)
        assert decision.escalate is True
        assert "scene-cut" in decision.reason

    def test_budget_suppresses_escalation(self):
        """With budget exhausted, even a gray-zone clip should not be escalated."""
        router = Router(threshold=0.02, gray_zone_margin=1.0,
                        budget_capacity=1, budget_refill=0.0)
        emb = torch.ones(512)
        router.set_centroid(emb.clone())
        router.route(0.02, emb, 0.0)   # consumes the only token
        decision = router.route(0.02, emb, 0.0)
        assert decision.escalate is False
        assert "budget-exceeded" in decision.reason

    def test_decision_fields_populated(self):
        """All RoutingDecision fields should be set to reasonable values."""
        router = Router()
        emb = torch.randn(512)
        dec = router.route(0.05, emb, 0.001, scene_cut=False)
        assert isinstance(dec.gray_zone_dist, float)
        assert isinstance(dec.ood_score, float)
        assert isinstance(dec.temporal_instability, float)
        assert isinstance(dec.scene_cut, bool)
        assert isinstance(dec.small_score, float)
        assert isinstance(dec.reason, str) and len(dec.reason) > 0


# ---------------------------------------------------------------------------
# Router.calibrate_threshold()
# ---------------------------------------------------------------------------

class TestRouterCalibration:
    def test_calibrate_sets_threshold(self):
        """After calibration the threshold should equal the specified quantile."""
        router = Router(threshold=0.02)
        scores = [float(i) / 100 for i in range(101)]  # 0.00 … 1.00
        router.calibrate_threshold(scores, quantile=0.95)
        assert router.threshold == pytest.approx(0.95, abs=0.01)

    def test_calibrate_with_uniform_scores(self):
        """All-equal scores → threshold equals that constant."""
        router = Router()
        router.calibrate_threshold([0.5] * 100, quantile=0.90)
        assert router.threshold == pytest.approx(0.5, abs=1e-5)
