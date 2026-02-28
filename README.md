# video-anomaly-router

**Cost-Aware Small–Large Model Routing for Unsupervised Video Anomaly Detection**

CSCI 566 Project — USC

---

## Overview

Video anomaly detection systems face a fundamental trade-off: lightweight models are fast and cheap but produce false positives under scene changes and ambiguous events; large spatiotemporal models are accurate but too expensive for continuous inference.

This project proposes a **cost-aware routing system** that runs a small autoencoder by default and escalates uncertain or out-of-distribution clips to a larger, higher-capacity model — only when needed. The system is **fully unsupervised**: trained only on normal video, with no labeled anomalies required.

```
Video Stream
    └─► FFmpeg frame extraction
            └─► Small 3D-Conv Autoencoder
                    ├─► [Confident] Anomaly score → Decision
                    └─► [Uncertain / OOD / Scene-cut]
                                └─► Large ResNet3D + Attention AE
                                            └─► High-confidence score → Decision
```

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
| Dataset download & preparation | In progress |
| Model training on real data | Pending |
| Evaluation (AUC-ROC, AUROC) | Pending |
| End-to-end inference pipeline | Pending |

---

## Repository Structure

```
video-anomaly-router/
├── src/
│   ├── device.py               # GPU/CPU detection with safe fallback
│   ├── preprocessing.py        # FFmpeg frame extraction, clip builder, Dataset
│   ├── routing.py              # Router: gray-zone, OOD, instability, scene-cut
│   └── models/
│       ├── small_autoencoder.py   # Lightweight 3D-Conv VAE (512-dim latent)
│       └── large_autoencoder.py   # Deep ResNet3D + attention bottleneck (1024-dim)
├── tests/
│   ├── test_device.py          # 13 tests for device management
│   ├── test_preprocessing.py   # 18 tests for preprocessing pipeline
│   ├── test_models.py          # 28 tests for both autoencoders
│   └── test_routing.py         # 20 tests for router and token bucket
├── train.py                    # Training entry point
├── requirements.txt
└── README.md
```

---

## Architecture

### Small Autoencoder (`src/models/small_autoencoder.py`)

A lightweight **3D convolutional VAE** designed for fast inference.

- **Input:** `(B, 3, T, H, W)` — batch of spatiotemporal clips
- **Encoder:** 4× Conv3D blocks with stride downsampling → flatten → FC to `(mu, log_var)`
- **Latent dim:** 512
- **Decoder:** 4× ConvTranspose3D blocks → Tanh reconstruction
- **Output:** reconstruction, mean embedding `mu`, log-variance `log_var`
- **Loss:** MSE reconstruction + KL divergence (weighted, VAE-style)
- **Uncertainty proxy:** `exp(log_var).mean()` — higher = less confident

The reparameterisation trick adds noise during training (stochastic) but uses only `mu` at inference (deterministic).

### Large Autoencoder (`src/models/large_autoencoder.py`)

A deeper model for high-fidelity reconstruction of ambiguous clips.

- **Encoder:** 5× residual 3D blocks + **spatiotemporal multi-head self-attention** bottleneck
- **Latent dim:** 1024
- **Decoder:** 5× ConvTranspose3D blocks → Tanh reconstruction
- **Attention:** `MultiheadAttention` over flattened `(T×H×W)` tokens at the bottleneck — lightweight substitute for a full Swin transformer, captures long-range spatiotemporal dependencies
- **Same loss interface as small AE** for easy comparison

### Routing Module (`src/routing.py`)

The router evaluates **four signals** per clip to decide whether to escalate:

| Signal | How it works | Escalate if |
|---|---|---|
| **Gray-zone distance** | `\|score - threshold\| / threshold` | Distance < `gray_zone_margin` (clip is ambiguous) |
| **OOD score** | Cosine distance of embedding from training centroid | Distance > `ood_threshold` |
| **Temporal instability** | Variance of recent reconstruction errors | Variance > `instability_threshold` |
| **Scene-cut flag** | Histogram correlation of adjacent frames | Cut detected |

A **token-bucket rate limiter** enforces a hard escalation budget — if the bucket is empty, escalation is suppressed regardless of signals. The threshold is calibrated post-training from the empirical quantile of normal-data reconstruction errors.

### Preprocessing (`src/preprocessing.py`)

1. **Frame extraction:** FFmpeg at a fixed FPS; falls back to OpenCV if FFmpeg is unavailable
2. **Normalisation:** Resize → ImageNet mean/std normalisation → float32
3. **Scene-cut detection:** 3D histogram correlation between adjacent frames (threshold: 0.5)
4. **Clip building:** Sliding window over frame sequences → `(C, T, H, W)` tensors
5. **Dataset:** `VideoClipDataset` loads pre-extracted frame directories

### Device Management (`src/device.py`)

- `get_device()` — detects CUDA; logs GPU name/VRAM; falls back to CPU with a warning
- `move(obj, device)` — recursively moves tensors, modules, dicts, lists
- `safe_to_device(tensor, device)` — catches `RuntimeError` (e.g. OOM) and falls back to CPU

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

## Running Tests

```bash
.venv\Scripts\pytest tests/ -v
# 79 passed in ~27s
```

---

## Training

```bash
# Train the small autoencoder
python train.py --model small --data data/train --epochs 50 --batch 8

# Train the large autoencoder
python train.py --model large --data data/train --epochs 30 --batch 4
```

**Expected data layout:**
```
data/train/
  scene_001/
    frame_000001.png
    frame_000002.png
    ...
  scene_002/
    ...
```

Checkpoints are saved to `checkpoints/` as `{model}_ae_best.pt`.

---

## Datasets

The system is evaluated on three standard unsupervised video anomaly detection benchmarks:

| Dataset | Scenes | Resolution | Anomaly types |
|---|---|---|---|
| UCSD Ped2 | 2 | 240×360 | Bikes, skaters on pedestrian paths |
| CUHK Avenue | 16 | 360×640 | Running, throwing, wrong direction |
| ShanghaiTech | 13 | 480×856 | Fighting, chasing, jaywalking |

Training uses **normal-only** clips. Frame-level annotations are used **only for evaluation**.

---

## Team

Sreenija Pavuluri · Bhuvanesh Perumal Samy · Adith Gunaseelan · Chirag Shivakumar · Preethika Prasad · Kevin Stephen
