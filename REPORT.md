# Midterm Report: Cost-Aware Small-Large Model Routing for Unsupervised Video Anomaly Detection

**CSCI 566 -- Deep Learning and Its Applications**
University of Southern California

**Team:** Sreenija Pavuluri · Bhuvanesh Perumal Samy · Adith Gunaseelan · Chirag Shivakumar · Preethika Prasad · Kevin Stephen

---

## Abstract

We present a cost-aware routing system for unsupervised video anomaly detection. The system runs a lightweight 3D-convolutional autoencoder (small model) by default and escalates uncertain or out-of-distribution clips to a larger, higher-capacity model only when needed. Routing decisions are driven by four signals: reconstruction gray-zone proximity, embedding OOD distance, temporal instability, and scene-cut detection. A token-bucket rate limiter enforces a hard escalation budget. The system is fully unsupervised -- trained entirely on normal video with no labeled anomalies. On the UCSD Ped2 benchmark, the small model alone achieves 0.6399 AUC. Routing with the large model (29.9% escalation rate) yields 0.6150 AUC in the current configuration. We identify a fundamental score-inversion failure mode in one test scene, analyze the root causes, and outline the path to competitive performance.

---

## 1. Introduction

Real-world video anomaly detection systems face a practical tension: lightweight models run efficiently at scale but produce false positives in ambiguous scenes; large spatiotemporal models are more accurate but too expensive for continuous inference across hundreds of camera feeds.

The standard approach -- running one model on all clips -- ignores this tension. Heavy models are deployed uniformly, wasting compute on the 80--90% of clips that are unambiguously normal. Lightweight models are deployed everywhere, producing unreliable results when the scene is difficult.

**Our approach:** Route each clip to the appropriate model based on its difficulty. Easy, confident clips are scored by the small model at low cost. Only clips that are ambiguous, out-of-distribution, or temporally unstable are escalated to the large model. This amortises the cost of the large model over the small fraction of hard clips.

**Contributions to date:**

- A four-signal routing module with a token-bucket budget constraint
- A lightweight 3D-conv VAE (small model) and a ResNet3D + attention AE (large model), both trained on normal video only
- A calibration pipeline that sets the anomaly threshold from the empirical 95th percentile of normal training scores
- Percentile-rank score normalisation enabling fair comparison between the two models' reconstruction errors
- Empirical routing experiments on UCSD Ped2 with a full per-scene diagnostic breakdown

---

## 2. Related Work

**Reconstruction-based anomaly detection.** The dominant paradigm trains an autoencoder on normal video; anomalies produce higher reconstruction error at inference [1]. Key works include ConvAE [2], 3D-ConvAE with temporal pooling, and MemAE [3], which adds a memory module to prevent the AE from generalising to anomalies.

**Frame-prediction baselines.** Alternatively, a model predicts future frames; anomalies are detected as high prediction error. Liu et al. [4] achieve 95.4% AUC on UCSD Ped2 with an optical-flow-guided prediction network.

**One-class classification.** SVDD [5] and its deep variant learn a hypersphere around normal embeddings; anomalies fall outside it. This approach decouples the embedding from the reconstruction objective and can be combined with our routing pipeline as an auxiliary signal.

**Routing and mixture-of-experts.** Adaptive computation [6] and mixture-of-experts [7] dynamically allocate compute per sample. Our routing system adapts these ideas to video anomaly detection: rather than selecting an expert, we select a quality tier based on sample difficulty signals.

---

## 3. System Architecture

```
Video Stream
    --> Frame extraction (FFmpeg / OpenCV)
            --> Small 3D-Conv VAE
                    |--> [Confident normal/anomaly] --> Reconstruction score --> Decision
                    |--> [Gray zone / OOD / Unstable / Scene-cut]
                                --> Router (4 signals + token bucket)
                                        --> Large ResNet3D + Attention VAE
                                                    --> High-confidence score --> Decision
```

### 3.1 Small Autoencoder

A lightweight **3D-convolutional VAE** designed for fast inference.

- **Input:** `(B, 3, T, H, W)` clips (T=16 frames, H=W=64)
- **Encoder:** 4 × Conv3D blocks with stride downsampling → FC to `(μ, log σ²)`
- **Latent dim:** 512
- **Decoder:** 4 × ConvTranspose3D → Tanh output
- **Loss:** MSE reconstruction + KL divergence (VAE-style)
- **Anomaly score:** per-clip MSE between input and reconstruction
- **Uncertainty proxy:** `exp(log σ²).mean()` -- used as an auxiliary routing signal

The reparameterisation trick adds noise during training (stochastic) but inference uses only `μ` (deterministic).

### 3.2 Large Autoencoder

A deeper model for high-fidelity reconstruction of ambiguous clips.

- **Encoder:** 5 × residual 3D blocks (`ResBlock3D`) with progressive downsampling + **spatiotemporal multi-head self-attention** at the bottleneck
- **Latent dim:** 1024 (2× the small model)
- **Decoder:** 5 × ConvTranspose3D → Tanh output
- **Attention:** `MultiheadAttention` over flattened `(T × H × W)` tokens at the bottleneck -- captures long-range spatiotemporal dependencies with O(n²) cost limited to the compressed bottleneck

The attention mechanism allows the large model to relate distant spatial positions (e.g. a cyclist at the top of frame to a pedestrian at the bottom), which is beyond the local receptive field of pure convolutions.

### 3.3 Routing Module

The router evaluates **four signals** per clip to decide whether to escalate:

| Signal | Computation | Escalate if |
|---|---|---|
| **Gray-zone distance** | `|score - threshold| / threshold` | distance < `gray_zone_margin` |
| **OOD score** | Cosine distance of embedding from training centroid | distance > `ood_threshold` |
| **Temporal instability** | Variance of recent reconstruction errors (sliding window) | variance > `instability_threshold` |
| **Scene-cut flag** | 3D histogram correlation of adjacent frames | cut detected |

A **token-bucket rate limiter** (`capacity=200`, `refill_rate=0.2`) enforces a hard escalation budget: if the bucket is empty, escalation is suppressed regardless of signal values. This gives direct control over the cost-quality tradeoff.

### 3.4 Score Calibration and Normalisation

The anomaly threshold is set post-training from the 95th percentile of reconstruction errors on the normal training set (294 clips from UCSD Ped2 Train001--Train016). Since the small and large models produce different raw MSE scales, we apply **percentile-rank normalisation**: each clip's raw MSE is mapped to its rank within the model's training-score distribution, producing a comparable [0, 1] scale for both models. In percentile space, the calibrated threshold is exactly 0.95 by definition.

---

## 4. Implementation

### 4.1 Data Pipeline

**Dataset:** UCSD Pedestrian 2 (Ped2) -- 16 training scenes (normal-only) and 12 test scenes with frame-level anomaly annotations. Anomalies are non-pedestrian objects: cyclists, skateboarders, and carts.

**Preprocessing:** Frames are extracted at the native frame rate using FFmpeg (falling back to OpenCV). Each frame is resized to 64 × 64 and normalised with ImageNet mean/std. Clips are built by a sliding window of 16 frames with stride 8, yielding 1830 test clips across 12 scenes.

**Scene-cut detection:** Adjacent frames are compared using 3D histogram correlation; a correlation below 0.5 triggers a scene-cut flag.

### 4.2 Training

Both models are trained on normal-only clips (no labels used during training).

| Hyperparameter | Small AE | Large AE |
|---|---|---|
| Epochs | 50 | 43 (best checkpoint) |
| Batch size | 16 | 4 |
| Learning rate | 1e-3 | 5e-4 |
| KL weight | 1e-4 | 5e-3 |
| Optimiser | Adam | Adam |
| Scheduler | ReduceLROnPlateau | ReduceLROnPlateau |
| val_recon (best) | 0.04044 | 0.04087 |

The large model uses 50× higher KL weight to enforce a tighter, more regularised latent space -- the intent is to reduce the volume of the latent space that the model maps normal patterns to, making anomalous patterns more distinguishable.

### 4.3 Evaluation

Evaluation computes frame-level AUC-ROC and Equal Error Rate (EER). Clip-level reconstruction scores are assigned to all frames in the clip; the per-frame score is the mean over all clips that contain that frame. The final AUC is computed against binary frame-level labels.

---

## 5. Experiments and Results

### 5.1 Baseline: Small Model Only

| Metric | Value |
|---|---|
| Frame-level AUC-ROC | **0.6399** |
| Equal Error Rate | 0.4587 |
| Escalation rate | 0% |

Per-scene breakdown reveals the headline number is misleading: 7 of 8 evaluable scenes score ≥ 0.75 AUC. The overall average is pulled down by a single scene (Test001, AUC = 0.34).

| Scene | AUC | Note |
|---|---|---|
| Test001 | 0.3400 | Score inversion -- see Section 6 |
| Test002 | 1.0000 | Perfect |
| Test003 | 0.9247 | Strong |
| Test004 | 1.0000 | Perfect |
| Test005 | 0.9184 | Strong |
| Test006 | 0.8320 | Good |
| Test007 | 1.0000 | Perfect |
| Test012 | 0.7460 | Reasonable |

### 5.2 Routing Experiments

All routing runs use a calibrated threshold (q = 0.95, raw MSE = 0.0889).

**Gray-zone margin sweep** (original large model, kl-weight=1e-4, 30 epochs):

| Margin | AUC | Escalation rate |
|---|---|---|
| 0.05 | 0.6370 | 26.9% |
| 0.10 | 0.6339 | 39.7% |
| 0.15 | 0.6342 | 47.0% |
| 0.30 | 0.6330 | 80.7% |

**Routing v2**: high-KL large model (kl-weight=5e-3, 43 epochs) + percentile-rank score normalisation, margin=0.05:

| Config | AUC | Escalation rate |
|---|---|---|
| Routing v2 | 0.6150 | 29.9% |

Full per-scene comparison:

| Scene | Small only | Routing v1 (0.05) | Routing v2 |
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

### 5.3 Routing Signal Diagnostics

A key output of the evaluation pipeline is per-scene routing signal means, which confirm the router correctly identifies hard scenes:

| Scene | gz_dist (mean) | Interpretation |
|---|---|---|
| Test001 | 0.036 | Clips sitting at the decision boundary -- maximum escalation pressure |
| Test011 | 0.040 | Also near boundary |
| Test012 | 0.068 | Moderately uncertain |
| Test002--Test010 | 0.12--0.17 | Well-separated from threshold; little escalation needed |

The gray-zone signal fires correctly for Test001 because those clips' reconstruction scores cluster near the calibrated threshold. The router is identifying the right clips. The problem is that the large model does not improve discrimination on those clips (see Section 6).

---

## 6. Analysis and Discussion

### 6.1 Why Test001 Has Inverted Scores

Test001 achieves AUC = 0.34 -- worse than a random classifier. This means the model assigns **lower** reconstruction error to anomalous frames than to normal ones. Two contributing factors:

1. **Background complexity.** Test001's normal clips contain dense pedestrian crowds with complex overlapping motion that is hard to reconstruct. Anomalous clips (a lone cyclist) have simpler, more structured motion that the autoencoder happens to reconstruct well.

2. **Reconstruction objective generalisation.** The VAE trained only on normal data learns a general motion prior. Some anomaly motion patterns (smooth, predictable bicycle motion) lie close to this prior; some normal patterns (crowds) lie far from it. There is no guarantee that unseen anomaly types will fall outside the learned distribution.

Score inversion of this kind is a known failure mode of pure reconstruction-based methods. It is not specific to model size -- both the small and large AEs exhibit it.

### 6.2 Why the Large Model Does Not Help

The large model improves Test001 incrementally (0.34 → 0.39 → 0.45 across configurations) but cannot resolve the fundamental inversion. Both models share the same training objective (MSE + KL on normal clips) and produce correlated score distributions. A larger model with more capacity learns a more faithful reconstruction of normal patterns, but this does not guarantee it will fail more on the specific anomaly patterns in Test001.

Additionally, with a wide gray-zone margin (0.30), 80.7% of clips are escalated to the large model. At this escalation rate, the system effectively runs the large model on everything -- negating the cost advantage of routing -- while achieving lower AUC than the small model alone.

### 6.3 Routing Efficiency

With a narrow margin (0.05), the router escalates ~27--30% of clips. The routing signal correctly concentrates escalations on the scenes that need them: Test001, Test011, and Test012. This is the intended behaviour and demonstrates that the routing pipeline itself is functioning correctly. The limitation is not routing accuracy -- it is the large model's anomaly discrimination.

---

## 7. Individual Contributions

| Team Member | Primary Responsibilities |
|---|---|
| **Adith Gunaseelan** | Project lead. System integration, training script (`train.py`), evaluation framework design (`evaluate.py`), routing diagnostic signals, experiment orchestration |
| **Bhuvanesh Perumal Samy** | Large autoencoder architecture (`large_autoencoder.py`): ResBlock3D, spatiotemporal multi-head attention bottleneck, VAE loss interface |
| **Chirag Shivakumar** | Routing module (`routing.py`): gray-zone signal, OOD cosine distance, temporal instability, token-bucket budget; router calibration script (`calibrate_router.py`) |
| **Sreenija Pavuluri** | Small autoencoder architecture (`small_autoencoder.py`): 3D-Conv VAE, reparameterisation trick, uncertainty proxy; training hyperparameter tuning |
| **Preethika Prasad** | Data pipeline (`preprocessing.py`, `scripts/prepare_ucsd.py`): FFmpeg frame extraction, clip builder, scene-cut detection, UCSD Ped2 dataset preparation and label CSV |
| **Kevin Stephen** | Testing suite (`tests/`, 79 tests passing): unit tests for device management, preprocessing, both model architectures, and routing module; evaluation metrics (AUC-ROC, EER, per-scene breakdown) |

---

## 8. Planned Work for the Final Report

The midterm results establish the routing pipeline infrastructure and surface the core challenge: pure reconstruction-based scoring is insufficient to detect all anomaly types (specifically, the score-inversion problem in Test001). The second half of the project will address this on three fronts.

### 8.1 Fixing the Score-Inversion Problem

The root cause is that reconstruction error alone cannot distinguish some anomaly types from normal clips. We will pursue two complementary fixes:

**Pseudo-anomaly augmentation.** We will generate synthetic negatives at training time using transformations that approximate anomalous appearance without requiring real labels -- Random Erasing, time-shuffled clips (temporal disorder), and spatial jitter (simulating sudden position shifts). A lightweight auxiliary classification head attached to the encoder will be trained to separate real-normal from pseudo-anomalous clips. At inference, the anomaly score will be a weighted combination of the reconstruction MSE and the auxiliary head's output. This approach is fully unsupervised in the sense that no real anomaly labels are used.

**Embedding-distance score ensemble.** The OOD signal (cosine distance from the training centroid in latent space) is currently used only for routing decisions, not for the final anomaly score. We will add it as a second score component: `final_score = α * recon_score + (1 - α) * ood_score`, with α tuned on a held-out validation split. In Test001, the centroid distance may still discriminate between normal and anomalous clips even when reconstruction error does not, because cyclists and skateboarders produce embeddings that are geometrically distant from the pedestrian-only training distribution.

### 8.2 Cost-Quality Tradeoff Curve

The central research question of this project is: **at a given escalation budget, what is the best achievable AUC?** We will generate a full cost-quality curve by sweeping `--gray-zone-margin` from 0.01 (nearly no escalation) to 1.0 (escalate everything) in increments of 0.02, recording `(escalation_rate, AUC)` at each point. This will be plotted as a Pareto curve showing the tradeoff between compute cost and detection quality, and will be the primary result figure in the final report.

We will also compare this curve against two baselines: running the small model only at every point (a flat line at 0.6399 AUC) and running the large model only (a flat line at the large-model AUC). The routing system is only useful if its Pareto curve dominates both baselines.

### 8.3 Cross-Dataset Generalisation

To assess whether the system generalises beyond UCSD Ped2, we will evaluate on two additional standard benchmarks:

- **CUHK Avenue** (16 train / 21 test scenes): anomalies include running, throwing objects, and wrong-direction walking -- different appearance statistics from UCSD Ped2.
- **ShanghaiTech** (13 scenes, 130 training clips, 107 test clips): large-scale dataset with fighting, chasing, and jaywalking. Significantly more diverse than UCSD Ped2.

The same trained models and router will be applied to these datasets without retraining, measuring zero-shot transfer. If performance degrades substantially, we will investigate domain-specific threshold recalibration (re-running `calibrate_router.py` on a small held-out normal set from the new domain) as a lightweight adaptation mechanism.

### 8.4 Timeline

| Week | Task |
|---|---|
| Week 9 | Pseudo-anomaly augmentation: implement, retrain small AE with auxiliary head |
| Week 9 | Embedding-distance score ensemble: tune α, re-evaluate on UCSD Ped2 |
| Week 10 | Cost-quality Pareto curve: full margin sweep, generate figure |
| Week 10 | CUHK Avenue evaluation (zero-shot + recalibrated) |
| Week 11 | ShanghaiTech evaluation |
| Week 11 | Final analysis: per-dataset results, Pareto curves, ablation table |
| Week 12 | Final report writing and presentation preparation |

---

## References

[1] Chalapathy, R. & Chawla, S. (2019). Deep learning for anomaly detection: A survey. *arXiv:1901.03407*.

[2] Hasan, M. et al. (2016). Learning temporal regularity in video sequences. *CVPR 2016*.

[3] Gong, D. et al. (2019). Memorizing normality to detect anomaly: Memory-augmented deep autoencoder for unsupervised anomaly detection. *ICCV 2019*.

[4] Liu, W. et al. (2018). Future frame prediction for anomaly detection -- a new baseline. *CVPR 2018*.

[5] Tax, D. & Duin, R. (2004). Support vector data description. *Machine Learning, 54*(1).

[6] Graves, A. (2016). Adaptive computation time for recurrent neural networks. *arXiv:1603.08983*.

[7] Shazeer, N. et al. (2017). Outrageously large neural networks: The sparsely-gated mixture-of-experts layer. *ICLR 2017*.
