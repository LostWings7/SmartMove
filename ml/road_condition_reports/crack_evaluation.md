# SafeCity Road Crack Detection Evaluation Report

## 1. Test Overview
- **Evaluated Model**: `road_crack_yolo26s_v1.pt`
- **Evaluation Dataset**: RDD2022 Held-Out Multi-Country Test Split
- **Confidence Threshold**: 0.25
- **Timestamp**: 2026-09-30 18:28:13 UTC

## 2. Overall Detection Metrics
| Metric | Value | Target Benchmark |
| :--- | :--- | :--- |
| **Precision** | **0.4694** | Visual true positive rate |
| **Recall** | **0.2933** | Defect capture rate |
| **F1 Score** | **0.3610** | Harmonic mean |
| **mAP50** | **0.2022** | Mean Average Precision at IoU 0.50 |
| **mAP50-95** | **0.0795** | COCO-standard across IoU 0.50:0.95 |

## 3. Per-Class Performance Breakdown
| Road Crack Category | Precision | Recall | F1 Score | mAP50 | mAP50-95 | Notes |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`longitudinal_crack` (D00)** | 0.5006 | 0.4041 | 0.4472 | 0.2906 | 0.1116 | Parallel to road direction |
| **`transverse_crack` (D10)** | 0.5153 | 0.3976 | 0.4489 | 0.2725 | 0.1074 | Perpendicular across roadway |
| **`alligator_crack` (D20)** | 0.3922 | 0.0781 | 0.1303 | 0.0435 | 0.0196 | Interconnected fatigue cracking |

## 4. Class Confusion & Boundary Analysis
- **Longitudinal vs Transverse**: Minor confusion on diagonal cracks, resolved by orientation aspect ratio.
- **Alligator vs Pavement Texture**: The inclusion of hard negative asphalt texture samples suppresses false alligator cracking alarms on coarse chip-seal surfaces.
- **Potholes (D40)**: Potholes are deliberately not trained in this model, maintaining separation from the protected pothole subsystem.

## 5. Hard Negative Stress Test (Clean Road & Surface Markings)
- **Negative Test Images Evaluated**: 18
- **False Alarms Recorded**: 3
- **Pavement Specificity**: `83.3%`
- **Challenging Artifacts Tested**:
  - Bitumen sealant lines
  - Worn yellow/white painted lane dividers
  - Harsh tree canopy shadows
  - Wet specular surface glare

## 6. Runtime Performance (Host CPU)
- **Mean Inference Latency**: `141.6 ms`
- **Throughput**: `7.1 FPS`
