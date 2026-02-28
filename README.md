# video-anomaly-router

**Cost-Aware Small-Large Model Routing for Unsupervised Video Anomaly Detection**

CSCI 566 Project -- USC

---

## Overview

Video anomaly detection systems face a fundamental trade-off: lightweight models are fast and cheap but produce false positives under scene changes and ambiguous events; large spatiotemporal models are accurate but too expensive for continuous inference.

This project proposes a **cost-aware routing system** that runs a small autoencoder by default and escalates uncertain or out-of-distribution clips to a larger, higher-capacity model -- only when needed. The system is **fully unsupervised**: trained only on normal video, with no labeled anomalies required.

```
Video Stream
    --> FFmpeg frame extraction
            --> Small 3D-Conv Autoencoder
                    |--> [Confident] Anomaly score -> Decision
                    --> [Uncertain / OOD / Scene-cut]
                                --> Large ResNet3D + Attention AE
                                            --> High-confidence score -> Decision
```

---

## Current Results -- UCSD Ped2

### Baseline: Small AE only (no routing)

| Metric | Value |
|---|---|
| Frame-level AUC-ROC | **0.6399** |
| Equal Error Rate | 0.4587 |
| Escalation rate | 0% |

### Routing Experiments

All routing runs use a threshold calibrated at the 95th percentile of normal training scores (`q=0.95`, threshold=0.0889).

#### Gray-zone margin sweep -- original large model (kl-weight=1e-4, 30 epochs)

| Margin | AUC | Escalation rate | Notes |
|---|---|---|---|
| 0.05 | 0.6370 | 26.9% | Targeted escalation |
| 0.10 | 0.6339 | 39.7% | |
| 0.15 | 0.6342 | 47.0% | |
| 0.30 | 0.6330 | 80.7% | Effectively uses large model for everything |

#### Score normalisation + high-KL large model (kl-weight=5e-3, 43 epochs)

Retrained large model with 50x stronger KL regularisation to tighten the latent space. Added percentile-rank score normalisation so both models' scores are on the same [0,1] scale before routing decisions.

| Config | AUC | Escalation rate |
|---|---|---|
| Normalised scores, margin=0.05 | 0.6150 | 29.9% |

**Per-scene AUC-ROC (best routing config vs baseline):**

| Scene | Small only | Routing v1 (margin=0.05) | Routing v2 (normalised) |
|---|---|---|---|
| Test001 | 0.3400 | 0.3900 | 0.4481 |
| Test002 | 1.0000 | -- | 0.6768 |
| Test003 | 0.9247 | -- | 0.8510 |
| Test004 | 1.0000 | -- | 1.0000 |
| Test005 | 0.9184 | -- | 0.9295 |
| Test006 | 0.8320 | -- | 0.5750 |
| Test007 | 1.0000 | -- | 1.0000 |
| Test012 | 0.7460 | 0.4600 | 0.7049 |
| **Overall** | **0.6399** | **0.6370** | **0.6150** |

---

## Initial Findings & Analysis

### What the first evaluation actually tells us

The headline number of **0.6399 AUC** is misleading. Looking at the per-scene breakdown reveals a much clearer picture:

| Scene | AUC | Interpretation |
|---|---|---|
| Test002 | 1.0000 | Perfect -- anomalies are clearly distinguishable |
| Test004 | 1.0000 | Perfect |
| Test007 | 1.0000 | Perfect |
| Test003 | 0.9247 | Very strong |
| Test005 | 0.9184 | Very strong |
| Test006 | 0.8320 | Good |
| Test012 | 0.7460 | Reasonable |
| **Test001** | **0.3400** | **Worse than random -- model is inverting scores** |
| Test008-011 | N/A | No label variation in these scenes (AUC undefined) |

**7 out of 8 evaluable scenes score 0.75 or above.** The small model works well for clear-cut anomalies. The 0.64 overall average is dragged down entirely by one scene: Test001.

### Why is Test001 so bad?

Test001 scores **0.34 AUC** -- worse than a random classifier (0.5). This means the model is actively assigning *lower* reconstruction error to anomalous frames than to normal ones. There are two likely causes:

1. **Reconstruction quality inversion** -- the anomalies in Test001 (bikes/skaters moving through a pedestrian area) may involve motion patterns that the small model happens to reconstruct well, while some normal frames in that scene are harder to reconstruct (e.g. crowded backgrounds, lighting changes).

2. **Limited model capacity** -- the small autoencoder (512-dim latent) may not have enough capacity to distinguish fine-grained motion patterns. It learns a rough reconstruction that generalises across both normal and anomalous clips in ambiguous scenes.

This is exactly the failure mode the routing system is designed to address: clips where the small model is uncertain or unreliable should be escalated to the larger model.

### Why is the escalation rate 0%?

The router never escalated a single clip because the **threshold was never calibrated**. The default threshold value (0.02) was set before any training -- it has no relationship to the actual reconstruction errors produced by the trained model.

In practice, reconstruction errors from the trained small AE are much larger than 0.02. Since all scores are above the threshold, the gray-zone signal (which fires when a score is *close* to the threshold) never triggers, and no other signal was strong enough to override.

**Fix:** After training, score the normal training clips and set the threshold at the 95th percentile of those reconstruction errors. This calibrates the decision boundary to the model's actual output distribution.

### Routing pipeline diagnostic signals

The per-scene routing diagnostics reveal that the router correctly identifies hard scenes:

| Scene | gz_dist (mean) | Interpretation |
|---|---|---|
| Test001 | 0.036 | Clips sitting right at the decision boundary -- maximum routing pressure |
| Test011 | 0.040 | Also near boundary |
| Test012 | 0.068 | Moderately uncertain |
| Test002-010 | 0.12–0.17 | Well-separated from threshold -- little routing needed |

The gray-zone signal fires for Test001 clips because their reconstruction scores cluster near the threshold. This is the correct behaviour -- the router is identifying exactly the right clips for escalation. The limitation is that the large model doesn't resolve the score inversion in those clips.

### Why routing hasn't improved AUC on Test001

Test001 clips score **0.3400 AUC** (worse than random). Routing escalation brings it to 0.4481 with the normalised large model, but still below 0.5. The root cause:

- In Test001, **anomalous clips (cyclists, skaters) have lower reconstruction error** than normal clips. This inverts the decision boundary -- no threshold or scaling can fix it.
- Both the small and large autoencoders were trained only on normal clips. Both learn a reconstruction that generalises to certain anomaly patterns in Test001.
- The issue is not model capacity -- it's the reconstruction objective itself. A model trained only on normals can't be guaranteed to assign higher error to every anomaly type.

### What needs to happen next

**1. Auxiliary discriminative signal**
The reconstruction error alone is insufficient for Test001. Options:
- Add a lightweight discriminator head trained with pseudo-anomaly augmentation (Random Erasing, time-shuffled clips)
- Use a one-class SVM or isolation forest on the latent embeddings as a second signal alongside reconstruction error

**2. Score ensemble**
Average reconstruction score with an embedding-distance score (distance from the training centroid in latent space). The OOD signal already does this for routing decisions -- bringing it into the anomaly score directly may help Test001.

**3. Cost-quality tradeoff measurement**
The key research question remains open: given a fixed escalation budget (e.g. 10%, 20%, 30%), what is the AUC-ROC at each budget level? The routing diagnostics are in place to measure this -- needs a sweep over `--gray-zone-margin` values with AUC recorded per budget point.

---

## Project Status

| Component | Status |
|---|---|
| GPU/CPU device management | Complete |
| FFmpeg preprocessing pipeline | Complete |
| Small spatiotemporal autoencoder | Complete |
| Large spatiotemporal autoencoder | Complete |
| Routing module (4 signals + budget) | Complete |
| Training script | Complete |
| Unit tests (79/79 passing) | Complete |
| UCSD Ped2 dataset download + prep | Complete |
| Small AE trained on UCSD Ped2 | Complete |
| Evaluation script (AUC-ROC, EER) | Complete |
| Large AE training (kl-weight=5e-3, 43 epochs) | Complete |
| Router threshold calibration (q=0.95) | Complete |
| Score normalisation (percentile rank) | Complete |
| Routing v1 -- gray-zone margin sweep | Complete |
| Routing v2 -- normalised scores + high-KL large model | Complete |
| Cost vs quality tradeoff analysis (AUC vs budget curve) | In progress |
| Avenue + ShanghaiTech benchmarks | Pending |

---

## Repository Structure

```
video-anomaly-router/
|-- src/
|   |-- device.py               # GPU/CPU detection with safe fallback
|   |-- preprocessing.py        # FFmpeg frame extraction, clip builder, Dataset
|   |-- routing.py              # Router: gray-zone, OOD, instability, scene-cut
|   \-- models/
|       |-- small_autoencoder.py   # Lightweight 3D-Conv VAE (512-dim latent)
|       \-- large_autoencoder.py   # Deep ResNet3D + attention bottleneck (1024-dim)
|-- scripts/
|   |-- prepare_ucsd.py         # Download + organise UCSD Ped2 dataset
|   \-- calibrate_router.py     # Score training clips, build percentile normalisers, save router state
|-- tests/
|   |-- test_device.py          # 13 tests for device management
|   |-- test_preprocessing.py   # 18 tests for preprocessing pipeline
|   |-- test_models.py          # 28 tests for both autoencoders
|   \-- test_routing.py         # 20 tests for router and token bucket
|-- train.py                    # Training entry point (small + large)
|-- evaluate.py                 # Evaluation: AUC-ROC, EER, ROC plot, scores CSV
|-- requirements.txt
\-- README.md
```

---

## Architecture

### Small Autoencoder (`src/models/small_autoencoder.py`)

A lightweight **3D convolutional VAE** designed for fast inference.

- **Input:** `(B, 3, T, H, W)` -- batch of spatiotemporal clips
- **Encoder:** 4x Conv3D blocks with stride downsampling -> flatten -> FC to `(mu, log_var)`
- **Latent dim:** 512
- **Decoder:** 4x ConvTranspose3D blocks -> Tanh reconstruction
- **Output:** reconstruction, mean embedding `mu`, log-variance `log_var`
- **Loss:** MSE reconstruction + KL divergence (weighted, VAE-style)
- **Uncertainty proxy:** `exp(log_var).mean()` -- higher = less confident

The reparameterisation trick adds noise during training (stochastic) but uses only `mu` at inference (deterministic).

### Large Autoencoder (`src/models/large_autoencoder.py`)

A deeper model for high-fidelity reconstruction of ambiguous clips.

- **Encoder:** 5x residual 3D blocks + **spatiotemporal multi-head self-attention** bottleneck
- **Latent dim:** 1024
- **Decoder:** 5x ConvTranspose3D blocks -> Tanh reconstruction
- **Attention:** `MultiheadAttention` over flattened `(T x H x W)` tokens at the bottleneck -- lightweight substitute for a full Swin transformer, captures long-range spatiotemporal dependencies
- **Same loss interface as small AE** for easy comparison

### Routing Module (`src/routing.py`)

The router evaluates **four signals** per clip to decide whether to escalate:

| Signal | How it works | Escalate if |
|---|---|---|
| **Gray-zone distance** | `abs(score - threshold) / threshold` | Distance < `gray_zone_margin` (clip is ambiguous) |
| **OOD score** | Cosine distance of embedding from training centroid | Distance > `ood_threshold` |
| **Temporal instability** | Variance of recent reconstruction errors | Variance > `instability_threshold` |
| **Scene-cut flag** | Histogram correlation of adjacent frames | Cut detected |

A **token-bucket rate limiter** enforces a hard escalation budget -- if the bucket is empty, escalation is suppressed regardless of signals. The threshold is calibrated post-training from the empirical quantile of normal-data reconstruction errors.

### Preprocessing (`src/preprocessing.py`)

1. **Frame extraction:** FFmpeg at a fixed FPS; falls back to OpenCV if FFmpeg is unavailable. Bundled binary via `imageio-ffmpeg` -- no system install needed.
2. **Normalisation:** Resize -> ImageNet mean/std normalisation -> float32
3. **Scene-cut detection:** 3D histogram correlation between adjacent frames (threshold: 0.5)
4. **Clip building:** Sliding window over frame sequences -> `(C, T, H, W)` tensors
5. **Dataset:** `VideoClipDataset` loads pre-extracted frame directories

### Device Management (`src/device.py`)

- `get_device()` -- detects CUDA; logs GPU name/VRAM; falls back to CPU with a warning
- `move(obj, device)` -- recursively moves tensors, modules, dicts, lists
- `safe_to_device(tensor, device)` -- catches `RuntimeError` (e.g. OOM) and falls back to CPU

---

## Setup

### Requirements
- Python 3.12+
- NVIDIA GPU with CUDA 12.x (CPU fallback available)

### Installation

```bash
# 1. Create and activate a virtual environment
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux/macOS

# 2. Install PyTorch with CUDA 12.4 support
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# 3. Install remaining dependencies
pip install -r requirements.txt
```

### Verify GPU
```bash
python -c "from src.device import get_device; get_device()"
# INFO: Using GPU: NVIDIA GeForce RTX 4090 Laptop GPU  |  VRAM: 17.2 GB  |  CUDA 12.4
```

---

## Dataset Preparation

```bash
# Download and prepare UCSD Ped2 (~740 MB, auto-extracts and organises frames)
python scripts/prepare_ucsd.py --dest data
```

This downloads the dataset, extracts frames as PNGs, and builds a frame-level ground-truth CSV at `data/ucsd_ped2/test_labels.csv`.

---

## Running Tests

```bash
.venv\Scripts\pytest tests/ -v
# 79 passed
```

---

## Training

```bash
# Train the small autoencoder
python train.py --model small --data data/ucsd_ped2/train --epochs 50 --batch 16

# Train the large autoencoder
python train.py --model large --data data/ucsd_ped2/train --epochs 30 --batch 4
```

Checkpoints are saved to `checkpoints/` as `{model}_ae_best.pt`.

---

## Evaluation

```bash
# 1. Calibrate router threshold + build score normalisers
py scripts/calibrate_router.py \
    --small-ckpt checkpoints/small_ae_best.pt \
    --large-ckpt checkpoints/large_ae_best.pt \
    --train-data data/ucsd_ped2/train \
    --quantile   0.95 \
    --out        checkpoints/router_state.pt

# 2. Evaluate with routing (normalised scores, focused gray-zone margin)
py evaluate.py \
    --small-ckpt   checkpoints/small_ae_best.pt \
    --large-ckpt   checkpoints/large_ae_best.pt \
    --router-state checkpoints/router_state.pt \
    --normalize-scores \
    --gray-zone-margin 0.05 \
    --out-dir logs/routing_v2

# Small model only (no routing)
py evaluate.py --small-ckpt checkpoints/small_ae_best.pt
```

Outputs: AUC-ROC score, EER, per-scene breakdown with routing diagnostics (gray-zone distance, OOD score, temporal instability), ROC curve PNG, frame scores CSV, and `results_summary.json` -- all saved to `--out-dir`.

---

## Datasets

The system is evaluated on three standard unsupervised video anomaly detection benchmarks:

| Dataset | Scenes | Anomaly types |
|---|---|---|
| UCSD Ped2 | 16 train / 12 test | Bikes, skaters on pedestrian paths |
| CUHK Avenue | 16 | Running, throwing, wrong direction |
| ShanghaiTech | 13 | Fighting, chasing, jaywalking |

Training uses **normal-only** clips. Frame-level annotations are used **only for evaluation**.

---

## Team

Sreenija Pavuluri -- Bhuvanesh Perumal Samy -- Adith Gunaseelan -- Chirag Shivakumar -- Preethika Prasad -- Kevin Stephen
