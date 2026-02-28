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

> Small autoencoder only (50 epochs, no routing yet). Large model + routing coming next.

| Metric | Value |
|---|---|
| Frame-level AUC-ROC | **0.6399** |
| Equal Error Rate | 0.4587 |
| Escalation rate | 0% (large model not yet trained) |

**Per-scene AUC-ROC:**

| Scene | AUC |
|---|---|
| Test001 | 0.3400 |
| Test002 | 1.0000 |
| Test003 | 0.9247 |
| Test004 | 1.0000 |
| Test005 | 0.9184 |
| Test006 | 0.8320 |
| Test007 | 1.0000 |
| Test012 | 0.7460 |

The overall AUC is expected to improve significantly once the large model is trained and the router is activated -- the routing system is specifically designed to handle the hard scenes (like Test001) where the small model struggles.

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
| Large AE training | In progress |
| Router threshold calibration | Pending |
| Cost vs quality tradeoff analysis | Pending |
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
|   \-- prepare_ucsd.py         # Download + organise UCSD Ped2 dataset
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
# Small model only
python evaluate.py \
    --small-ckpt checkpoints/small_ae_best.pt \
    --test-data  data/ucsd_ped2/test \
    --labels     data/ucsd_ped2/test_labels.csv

# With large model + routing
python evaluate.py \
    --small-ckpt checkpoints/small_ae_best.pt \
    --large-ckpt checkpoints/large_ae_best.pt \
    --test-data  data/ucsd_ped2/test \
    --labels     data/ucsd_ped2/test_labels.csv
```

Outputs: AUC-ROC score, EER, per-scene breakdown, ROC curve PNG, and a frame scores CSV -- all saved to `logs/`.

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
