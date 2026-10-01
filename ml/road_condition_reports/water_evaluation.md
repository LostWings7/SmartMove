# SafeCity Water Segmentation Evaluation Report

## 1. Test Overview
- **Evaluated Model**: `water_yolo26s_seg_v1.pt`
- **Held-out Test Images**: 567
- **Confidence Threshold**: 0.25
- **Timestamp**: 2026-09-30 18:21:58 UTC

## 2. Primary Pixel-Level Segmentation Quality
| Metric | Value | Definition / Target |
| :--- | :--- | :--- |
| **Pixel IoU (Jaccard Index)** | **0.4511** | Intersection / Union of water mask pixels |
| **Mean per-image IoU** | **0.6616** | Unweighted average across test frames |
| **Pixel Dice Coefficient** | **0.6217** | Harmonic mean of overlap (2*TP / (2*TP+FP+FN)) |
| **Pixel Precision** | **0.5107** | True water pixels / predicted water pixels |
| **Pixel Recall** | **0.7945** | True water pixels detected / ground-truth water pixels |
| **Pixel F1 Score** | **0.6217** | Harmonic balance between Precision & Recall |
| **Mask mAP50** | **0.5842** | Instance mask mAP at 0.50 IoU cutoff |
| **Mask mAP50-95** | **0.3409** | Instance mask mAP across 0.50:0.95 IoU thresholds |

## 3. Sub-Category & Hard-Case Breakdown
| Condition Subset | Count | IoU | Dice / F1 | Precision | Recall | Notes |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Small Isolated Puddles (<2% road coverage)** | 229 | 0.2010 | 0.3347 | 0.2084 | 0.8501 | Thin puddle boundaries |
| **Medium Waterlogging (2%–10% road coverage)** | 189 | 0.6269 | 0.7706 | 0.7328 | 0.8125 | Lane ponding |
| **Extensive Ponding (>10% road coverage)** | 147 | 0.6560 | 0.7923 | 0.8597 | 0.7347 | Major road flooding |
| **Multi-Region Frames (>1 disconnected puddle)** | 286 | 0.4812 | 0.6498 | 0.5616 | 0.7708 | Complex multi-patch scenes |
| **Clean Road Surfaces (Negative Samples)** | 2 | N/A | N/A | Specificity: 0.9095 | FPR: 0.090500 | Clean asphalt baseline |

## 4. Inference Performance (Host CPU)
- **Mean Inference Latency**: `180.68 ms`
- **P95 Latency**: `205.29 ms`
- **Throughput (PyTorch CPU)**: `5.5 FPS`

## 5. False Positive & Boundary Analysis
- Highly reflective dry pavement (specular sunlight glare) occasionally yields weak segment proposals filtered out by the 0.25 confidence threshold.
- Small vehicle shadows do not trigger false water detections due to training diversity.
