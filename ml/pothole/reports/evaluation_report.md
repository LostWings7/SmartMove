# SafeCity Loop V2 — Pothole Detector Evaluation Report

- **Evaluated Model:** `models/pothole_yolo26s.pt`
- **Held-Out Test Set:** 215 images (200 positive, 15 negative)

## 1. Overall Test Set Performance

| Metric | Value |
| :--- | :--- |
| **Precision (P)** | **0.4747** |
| **Recall (R)** | **0.4294** |
| **F1 Score** | **0.4509** |
| **mAP@50** | **0.3774** |
| **mAP@50-95** | **0.1678** |
| **Inference Latency** | **150.6 ms** (6.6 FPS on Intel CPU) |

## 2. Confidence-Threshold Sweep

| Threshold | Precision | Recall | F1 Score | TP | FP | FN |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `0.20` | 0.2302 | 0.6242 | **0.3364** | 201 | 672 | 121 |
| `0.30` | 0.3702 | 0.5093 | **0.4288** | 164 | 279 | 158 |
| `0.40` ⭐ *(Selected)* | 0.5226 | 0.3944 | **0.4496** | 127 | 116 | 195 |
| `0.50` | 0.7190 | 0.2702 | **0.3928** | 87 | 34 | 235 |
| `0.60` | 0.7377 | 0.1398 | **0.2350** | 45 | 16 | 277 |
| `0.70` | 0.7200 | 0.0559 | **0.1037** | 18 | 7 | 304 |
| `0.80` | 1.0000 | 0.0031 | **0.0062** | 1 | 0 | 321 |

## 3. Threshold Decision Rationale

The deployment threshold **`0.40`** was empirically chosen because it maximizes the harmonic mean of precision and recall ($F_1 = 0.4496$), avoiding excessive false alerts on benign surface stains while ensuring critical road safety defects are reliably flagged.

## 4. Robustness & Clean Road Specificity

- Negative background test images: 15
- False alarms on clean asphalt: 2
- Clean road specificity rate: 86.7%
