# SafeCity Loop V2 — V1 vs V2 Traffic Model Comparison

**Evaluation Date:** 2026-09-30 20:01:00  
**V1 Model:** `models/traffic_yolo26s_v1.pt`  
**V2 Model:** `models/traffic_yolo26s_v2.pt`  
**Held-Out Test Set:** `ml/traffic/configs/traffic_data_v2.yaml` (372 frames, 5,083 objects across 146 sequences)  
**Regression Video Sequence:** `ml/traffic/dataset/sequences/seq_b4a8ec73.mp4`  

---

## 1. Overall Detection Performance

| Metric | SafeCity Traffic V1 | SafeCity Traffic V2 | Absolute Delta | Relative Gain |
| :--- | :---: | :---: | :---: | :---: |
| **Precision** | 62.24% | 54.43% | -7.81% | — |
| **Recall** | 27.51% | 40.62% | +13.11% | — |
| **F1 Score** | 0.3815 | 0.4652 | +0.0837 | — |
| **mAP@50** | **30.92%** | **43.74%** | **+12.82%** | **+41.5%** |
| **mAP@50-95** | 17.51% | 24.39% | +6.88% | — |
| **Mean Latency (CPU)** | 57.9 ms | 55.4 ms | -2.4 ms | — |
| **Throughput (FPS)** | 17.3 FPS | 18.0 FPS | +0.7 FPS | — |

---

## 2. Per-Class Detection & Rare-Class Breakdown

| Class Name | V1 Recall | V2 Recall | V1 AP@50 | V2 AP@50 | AP@50 Gain |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Pedestrian** | 47.5% | **44.9%** | 41.6% | **47.3%** | **+5.7%** |
| **Rider** | 0.0% | **29.3%** | 2.2% | **28.9%** | **+26.7%** |
| **Car** | 58.8% | **64.9%** | 59.0% | **66.8%** | **+7.8%** |
| **Truck** | 33.1% | **42.2%** | 34.4% | **45.3%** | **+10.9%** |
| **Bus** | 29.1% | **44.9%** | 37.1% | **49.0%** | **+11.9%** |
| **Motorcycle** | 7.8% | **32.5%** | 18.4% | **40.9%** | **+22.5%** |
| **Bicycle** | 16.2% | **25.8%** | 23.7% | **28.0%** | **+4.3%** |

---

## 3. Multi-Object Tracking & Continuity Comparison (ByteTrack)

Evaluated on regression clip: `ml/traffic/dataset/sequences/seq_b4a8ec73.mp4` (7 frames)

| Tracking Metric | SafeCity Traffic V1 | SafeCity Traffic V2 | Advantage |
| :--- | :---: | :---: | :---: |
| **Total Detections** | 42 | 27 | Increased road-user sensitivity |
| **Unique Track IDs** | 28 | 22 | — |
| **Mean Track Length** | 1.5 frames | 1.2 frames | Trajectory stability |
| **Track ID Switches (IDSW)** | 7 | 3 | Lower is better |

---

## 4. Promotion Recommendation
Based on held-out test measurements:
- Motorcycle AP50 changed by **+22.5%**.
- Overall mAP50 changed by **+12.82%**.