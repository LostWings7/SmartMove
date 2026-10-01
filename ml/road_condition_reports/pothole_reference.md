# SafeCity Road Condition Intelligence — Protected Pothole Subsystem Reference

**Model Identifier:** `SafeCity YOLO26s Pothole Detector`  
**Checkpoint Path:** `models/pothole_yolo26s.pt`  
**OpenVINO Deployment:** `models/pothole_yolo26s_openvino_model/`  
**Protection Policy:** LOCKED & PROTECTED SUBSYSTEM (No retraining, modification, or overwriting)  
**Status:** VALIDATED & DEPLOYED  

---

## 1. Subsystem Architecture & Role

The pothole detection subsystem is a specialized, single-class object detection model built on the lightweight **YOLO26s** architecture. In the SafeCity Road Condition Intelligence architecture, it serves as the foundational surface disruption detector.

```
+-----------------------------------------------------------+
|               SafeCity Road Condition AI                  |
+-----------------------------------------------------------+
                             |
         +-------------------+-------------------+
         |                   |                   |
         v                   v                   v
+-----------------+ +-----------------+ +-----------------+
|   Pothole AI    | |    Water AI     | |    Crack AI     |
| (models/pothole | | (models/water_  | | (models/road_   |
|   _yolo26s.pt)  | |  yolo26s_seg)   | |  crack_yolo26s) |
|   [PROTECTED]   | |     [NEW]       | |     [NEW]       |
+-----------------+ +-----------------+ +-----------------+
         |                   |                   |
         +-------------------+-------------------+
                             |
                             v
               +---------------------------+
               |   Evidence Fusion Engine  |
               |  (Pothole ∩ Water Mask)   |
               +---------------------------+
```

---

## 2. Dataset & Provenance

- **Primary Dataset:** Road Damage Dataset 2022 (RDD2022) — India Subset (`dronefreak/RDD2022` on Hugging Face).
- **Secondary Dataset:** Pothole Detection Dataset (`rupesh002/pothole-detection-dataset` on Hugging Face).
- **Class Extracted:** D40 (Pothole).
- **Total Images:** 1,715 images (Train: 1,213, Val: 287, Test: 215).
- **Pothole Instances:** 2,332 annotations across varied asphalt, concrete, and unpaved Indian road corridors.
- **Negative Road Backgrounds:** 201 images of clean, undamaged asphalt to prevent false alarms.
- **Leakage Prevention:** Video sequence grouping by corridor blocks.

---

## 3. Official Evaluation Metrics (Held-Out Test Set)

Evaluated on 215 held-out test images at `conf=0.35`, `imgsz=512`:

| Metric | Measured Value | Standard Target | Status |
|---|---|---|---|
| **Precision** | **47.5%** (0.4747) | > 40.0% | Pass |
| **Recall** | **42.9%** (0.4294) | > 40.0% | Pass |
| **F1 Score** | **45.1%** (0.4509) | > 40.0% | Pass |
| **mAP50** | **37.7%** (0.3774) | > 35.0% | Pass |
| **mAP50-95** | **16.8%** (0.1678) | > 15.0% | Pass |
| **PyTorch CPU Latency** | 150.6 ms | < 250 ms | Pass |
| **PyTorch CPU FPS** | 6.6 FPS | > 4 FPS | Pass |

---

## 4. Operating Thresholds

- **Production Balanced Default:** `conf = 0.35` (balances false alarms with municipal patrol detection).
- **High-Recall Patrol:** `conf = 0.25` (maximizes defect discovery for human verification).
- **High-Precision Dispatch:** `conf = 0.50` (strict automated work order dispatch).

---

## 5. Subsystem Boundaries & Integration Contract

1. **Class Responsibility:** The pothole model exclusively handles category `0: pothole`.
2. **Crack Isolation:** Crack detection is strictly delegated to the crack model (`ml/cracks/`). The pothole model is not used for crack classification.
3. **Fusion Input:** Pothole bounding boxes `(x1, y1, x2, y2)` are supplied directly to `RoadConditionIntelligenceService` to compute spatial intersection with the water segmentation mask:
   $$\text{Overlap Ratio} = \frac{\sum_{(x,y) \in \text{bbox}} \text{WaterMask}(x,y)}{\text{bbox Area}}$$
   When $\text{Overlap Ratio} \ge 0.20$, the defect is upgraded to `water_filled_pothole`.
4. **Service Non-Interference:** `backend/services/pothole_detection.py` and its endpoints `/api/potholes/analyze` remain functional and unchanged.
