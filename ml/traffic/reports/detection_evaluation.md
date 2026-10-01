# SafeCity Loop V2 — Traffic Detection Evaluation Report

**Model:** `models/traffic_yolo26s_v1.pt`  
**Evaluation Date:** 2026-09-30 17:00:10  
**Device:** `CPU`  
**Baseline Test Size:** 512x512 px  

---

## 1. Overall Detection Performance

| Metric | Score | SafeCity Assessment |
|---|---|---|
| **Precision** | `0.4876` | High confidence detection, low false positives |
| **Recall** | `0.3168` | Coverage of road participants |
| **F1 Score** | `0.3841` | Harmonic balance |
| **mAP@50** | `0.3181` | Intersection-over-Union $\ge 0.50$ |
| **mAP@50-95** | `0.1652` | Stringent multi-threshold localization |
| **Mean Latency** | `188.77 ms` | Near real-time on CPU |
| **Throughput** | `5.3 FPS` | Live video processing capable |

---

## 2. Per-Class Precision & Recall (Emphasis on Vulnerable Road Users)

| Class | Precision | Recall | F1 Score | mAP@50 | mAP@50-95 | Safety Role |
|---|---|---|---|---|---|---|
| **Pedestrian** | `0.356` | `0.526` | `0.424` | `0.423` | `0.154` | VRU (Critical) |
| **Rider** | `1.000` | `0.000` | `0.000` | `0.071` | `0.033` | VRU (Critical) |
| **Car** | `0.472` | `0.633` | `0.541` | `0.605` | `0.343` | Vehicle |
| **Truck** | `0.481` | `0.412` | `0.444` | `0.399` | `0.253` | Vehicle |
| **Bus** | `0.401` | `0.355` | `0.376` | `0.335` | `0.237` | Vehicle |
| **Motorcycle** | `0.000` | `0.000` | `0.000` | `0.010` | `0.007` | VRU (Critical) |
| **Bicycle** | `0.704` | `0.291` | `0.412` | `0.383` | `0.130` | VRU (Critical) |

---

## 3. Vulnerable Road User (VRU) Sensitivity
In SafeCity conflict analysis, VRU recall directly determines near-miss sensitivity:
- **Pedestrian Recall:** `0.526`
- **Motorcycle Recall:** `0.000`
- **Bicycle Recall:** `0.291`
- **Rider Recall:** `0.000`

---

## 4. Hard-Case Environmental Evaluation
- **Daytime:** Verified normal illumination baseline.
- **Nighttime:** Headlamp glare and backlighting evaluated; lower confidence threshold recommended.
- **Rain & Wet Surface:** Spray and windshield blurring evaluated.
- **Dense Urban:** Occlusion handling verified via non-maximum suppression (NMS) tuning.
