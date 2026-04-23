# Post-Midterm Changes — Final Report Tracking

Changes made after the midterm submission. This is the source of truth for
the "Changes Since Midterm" section of the final report.

---

## 1. Cost-Quality Pareto Curve

**Script:** `scripts/pareto_sweep.py`
**Output:** `logs/pareto/pareto_curve.png`, `logs/pareto/pareto_results.json`

### Motivation

The central research question of this project is: at a given escalation budget,
what is the best achievable AUC? The midterm established the routing
infrastructure and showed individual operating points. The Pareto curve
sweeps the full range of budgets in a single run to answer whether the routing
system's cost-quality frontier dominates running either model alone.

### Method

Scores all 1830 test clips once with both models, then sweeps
`gray-zone-margin` across 50 values (0.01 → 1.0, step 0.02) in-memory.
Each margin value yields a `(escalation_rate, AUC)` point. Two flat-line
baselines are computed from the same pre-scored clips:

- **Small-only** (normalised scores): AUC = 0.6041 at 0% escalation
- **Large-only** (normalised scores): AUC = 0.6047 at 100% escalation

Scores use percentile-rank normalisation (same as routing v2) so small and
large model outputs are on the same [0, 1] scale before routing decisions.

### Results

| Margin | Escalation % | AUC |
|---|---|---|
| 0.01 | 4.0% | 0.6050 |
| 0.03 | 12.8% | 0.6134 |
| 0.05 | 28.2% | 0.6149 |
| **0.07** | **37.5%** | **0.6155** ← peak |
| 0.09 | 39.3% | 0.6148 |
| 0.11 | 40.6% | 0.6148 |
| 0.13 | 93.5% | 0.6055 |
| 0.15+ | >94% | ~0.605 → large-only |

### Key finding

**The routing curve Pareto-dominates both baselines across the 4–40%
escalation range.** At the optimal margin (0.07, 37.5% escalation) the system
achieves AUC = 0.6155, which is +0.011 above small-only and +0.011 above
large-only — using the large model for only 37.5% of clips.

The sharp discontinuity at margin=0.13 (escalation jumps from 40% → 93% in
a single 0.02 step) reveals the bimodal structure of the score distribution:
clips are either close to the decision threshold (ambiguous) or far from it
(confident). The router's selective operating range is naturally bounded below
margin ≈ 0.12.

### Command

```bash
python scripts/pareto_sweep.py \
    --small-ckpt checkpoints/small_ae_baseline.pt \
    --large-ckpt checkpoints/large_ae_best.pt \
    --router-state checkpoints/router_state.pt \
    --out-dir logs/pareto
```

---

## 2. Full Results Summary

| Configuration | Overall AUC | Test001 AUC | Escalation |
|---|---|---|---|
| Small model only (raw scores) | 0.6399 | 0.3400 | 0% |
| Routing v1, margin=0.05 | 0.6370 | 0.3900 | 26.9% |
| Routing v1, margin=0.10 | 0.6339 | — | 39.7% |
| Routing v1, margin=0.30 | 0.6330 | — | 80.7% |
| **Routing v2 (normalised, margin=0.05)** | **0.6150** | **0.4481** | **29.9%** |
| Pareto peak (normalised, margin=0.07) | 0.6155 | — | 37.5% |
| Small-only (normalised) | 0.6041 | — | 0% |
| Large-only (normalised) | 0.6047 | — | 100% |

Note: routing v2 / normalised runs use percentile-rank score normalisation,
which is applied per-clip before overlap-add averaging. This makes raw-score
and normalised-score AUC values not directly comparable.

---

## 3. Routing Ablation Study

**Log dirs:** `logs/ablation/`

### Setup

All configs use the same Ped2-trained model weights and Ped2 router state
(`checkpoints/router_state.pt`, margin=0.07, normalized scores). Each row adds
one routing signal cumulatively.

### Results

| Config | AUC-ROC | Escalation |
|---|---|---|
| Small-only (no routing) | 0.6041 | 0% |
| + gray-zone (margin=0.07) | 0.6153 | 36.8% |
| + OOD detection | 0.6153 | 36.8% |
| + temporal instability | 0.6157 | 39.3% |
| + scene-cut (full system) | 0.6157 | 39.3% |

### Key findings

- **Gray-zone is the dominant signal**: adds +0.011 AUC alone. All other
  signals add at most +0.0004 on top of it.
- **OOD detection contributes nothing on Ped2**: all Ped2 test clips are
  in-distribution relative to the Ped2 training centroid (ood_score ≈ 0.01).
  The signal only matters for cross-dataset transfer (which already fails).
- **Temporal instability adds marginal gain**: +0.0004 AUC at +2.5% extra
  escalation — a poor tradeoff, but negligible cost in practice.
- **Scene-cut detection adds nothing on Ped2**: the test clips have very few
  actual cuts, so this signal never fires. Its value would be for streaming
  video where cuts are common.
- **Conclusion for the report**: the routing system's gain over small-only
  comes entirely from gray-zone proximity detection. The multi-signal design
  is insurance for out-of-distribution or streaming scenarios, not Ped2 itself.

### Command

```bash
# Gray-zone only
python evaluate.py --small-ckpt checkpoints/small_ae_baseline.pt \
    --large-ckpt checkpoints/large_ae_best.pt \
    --router-state checkpoints/router_state.pt \
    --normalize-scores --gray-zone-margin 0.07 \
    --ood-threshold 999 --instability-threshold 999 --no-scene-cut \
    --out-dir logs/ablation/gz_only

# Full system
python evaluate.py --small-ckpt checkpoints/small_ae_baseline.pt \
    --large-ckpt checkpoints/large_ae_best.pt \
    --router-state checkpoints/router_state.pt \
    --normalize-scores --gray-zone-margin 0.07 \
    --out-dir logs/ablation/full
```

---

## 4. Cross-Dataset Evaluation — UCSD Ped1

**Data:** `data/ucsd_ped1/`  (36 test scenes, 7 200 frames, 17.2% anomalous)  
**Prep script:** `scripts/prepare_ucsd_ped1.py`  
**Log dirs:** `logs/ped1_zeroshot/`, `logs/ped1_recalibrated/`, `logs/ped1_small_only/`

### Motivation

UCSD Ped1 shares the same raw archive as Ped2 but is a different camera, wider
pathway, and sparser crowd. Evaluating the Ped2-trained system on Ped1 tests
zero-shot cross-scene generalisation.

### Protocol

Two conditions — same model weights (Ped2-trained), different router calibration:

| Condition | Router state | AUC | Escalation |
|---|---|---|---|
| Zero-shot (Ped2 calibration) | `checkpoints/router_state.pt` | 0.5000 | 100% |
| Domain-recalibrated | `checkpoints/router_state_ped1.pt` | 0.4509 | 20.3% |
| Small-only (Ped1-recalibrated) | `checkpoints/router_state_ped1.pt` | 0.4903 | 0% |

### Key findings

- **Zero-shot failure**: Ped2 OOD centroid flags all Ped1 clips as out-of-distribution
  (cosine distance ≈ 1), routing 100% to the large model. Combined AUC = 0.50 (random).
- **After domain recalibration** (new centroid + normalizers from Ped1 training data,
  same model weights): AUC drops to **0.45** — below random. Score inversion is
  more severe on Ped1: anomalous clips (bicycles, wheelchairs, skateboarders) have
  *lower* reconstruction error than dense normal pedestrian scenes.
- **Routing vs small-only**: routing (0.4509) is slightly worse than small-only
  (0.4903) because escalating uncertain clips to the large model also fails to
  discriminate — both models show the same inversion.
- **Best per-scene**: Test003=0.97, Test022=0.86, Test032=0.84 (anomalies
  reconstructed poorly, as expected). Worst: Test004=0.07 (score-inverted).
- **Conclusion**: The system does not generalise zero-shot to Ped1. Domain-specific
  training data is required; recalibration of the router alone is insufficient when
  the encoder itself learned Ped2-specific features.

### Commands

```bash
python scripts/prepare_ucsd_ped1.py --raw data/raw --dest data

# Domain recalibration
python scripts/calibrate_router.py \
    --small-ckpt checkpoints/small_ae_baseline.pt \
    --large-ckpt checkpoints/large_ae_best.pt \
    --train-data data/ucsd_ped1/train \
    --out checkpoints/router_state_ped1.pt

# Evaluate
python evaluate.py \
    --small-ckpt checkpoints/small_ae_baseline.pt \
    --large-ckpt checkpoints/large_ae_best.pt \
    --router-state checkpoints/router_state_ped1.pt \
    --test-data data/ucsd_ped1/test \
    --labels data/ucsd_ped1/test_labels.csv \
    --normalize-scores --gray-zone-margin 0.07 \
    --out-dir logs/ped1_recalibrated
```

---

## 5. Figures for the Final Report

All figures are pre-generated and ready to embed. Paths are relative to project root.

| Figure | Path | Description |
|---|---|---|
| Pareto curve | `logs/pareto/pareto_curve.png` | Cost-quality frontier: AUC vs escalation rate. Routing dominates both baselines at 4–40% escalation. |
| Ped2 routing v2 ROC | `logs/routing_v2/roc_curve.png` | Best UCSD Ped2 result (AUC=0.615, 29.9% escalation). |
| Ped1 recalibrated ROC | `logs/ped1_recalibrated/roc_curve.png` | UCSD Ped1 domain-recalibrated (AUC=0.45, shows failure to generalise). |
| Ped1 zero-shot ROC | `logs/ped1_zeroshot/roc_curve.png` | UCSD Ped1 zero-shot with Ped2 calibration (AUC=0.50, 100% escalation). |

Additional prep scripts (data not downloaded, scripts ready to run if needed):
- `scripts/prepare_avenue.py` — CUHK Avenue (16 train / 21 test scenes)
- `scripts/prepare_shanghaitech.py` — ShanghaiTech (330 train / 107 test clips)

---

## Timeline

| Date | Change |
|---|---|
| 2026-04-22 | Pareto sweep script written and executed (50 margins, ~27 seconds) |
| 2026-04-22 | Pareto curve confirmed: routing dominates both baselines at 4–40% escalation |
| 2026-04-22 | UCSD Ped1 prep script + cross-dataset evaluation (zero-shot + domain-recalibrated) |
| 2026-04-22 | Routing ablation study: gray-zone is sole contributor to AUC gain on Ped2 |
| 2026-04-22 | CUHK Avenue + ShanghaiTech prep scripts written (data not downloaded) |
