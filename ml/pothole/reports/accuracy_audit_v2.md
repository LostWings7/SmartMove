# SafeCity Pothole Subsystem — Accuracy & Dataset Audit V2

## 1. Root Cause Investigation: Hypotheses A–L (Section 4)

Before retraining any weights, we performed controlled diagnostics on `ROAD_CONDITION_REGRESSION_001` (`270x148` rural road with extensive cracking, banana plant in top-right sky, bushes, and grass) and secondary failure cases (`REGRESSION_002` through `REGRESSION_004`).

| Hypothesis | Investigated? | Empirical Finding | Severity |
| :--- | :---: | :--- | :---: |
| **A. Insufficient Negative Data** | Yes | `processed_v2` included 726 road negatives, but lacked explicit non-road object negatives (foliage/banana leaves, buildings, sky/clouds, utility poles, manholes, vehicles) and dense forward-facing alligator crack networks without potholes. | **HIGH** |
| **B. Annotation Errors** | Yes | Found **207** annotations in the upper 35% of the frame (background/horizon), **145** elongated thin boxes (`aspect > 4.2` or `< 0.24`) where linear cracks/shadows were labeled as potholes, and **133** oversized boxes (`>30%` frame area) covering surrounding road texture and crack networks. | **HIGH** |
| **C. Wrong Dataset Conversion** | Yes | `RDD2022` class `3` (`D40`) was mapped to `0: pothole`. However, **338** positive training/val/test images in `RDD2022_India` contain *both* `D40` (pothole/spalling) and `D00/D10/D20` (cracks), where RDD2022 `D40` often annotates shallow surface spalling inside alligator crack networks. | **HIGH** |
| **D. Class Contamination** | Yes | In `Roboflow_rupesh002` and `RDD2022_India` (`D40`), weathered crack intersections and dark tar patches are frequently annotated as `pothole`, teaching the detector to fire on dark crack junctions. | **HIGH** |
| **E. Image Preprocessing / Fallback Bug** | **CRITICAL** | **SMOKING GUN #1**: In `backend/services/pothole_detection.py` (lines 480–482), whenever the neural `pothole_yolo26s.pt` model returns `0` detections on an uploaded image, the service automatically executed `_detect_optical_surface_anomalies()`, an unmasked OpenCV grayscale threshold (`blurred < mean - std`) that boxed the top 5 darkest contours in the image—including the banana leaf (`[225,0,270,39]`, fake conf `0.75`), right bushes (`0.74`), left grass (`0.73`), and road cracks (`0.84`, `0.75`) on `ROAD_CONDITION_REGRESSION_001`! | **CRITICAL** |
| **F. Confidence Threshold** | Yes | `PotholeDetectionService` defaulted to `conf_threshold = 0.20`, where the held-out False Discovery Rate is **36.93%** (`113 FP` vs `193 TP`). On `REGRESSION_003` (alligator cracks) and `REGRESSION_004` (overhead SUV), the neural model fired false positives at `conf=0.369` and `0.316`. | **MODERATE** |
| **G. NMS / Duplicate Boxes** | Yes | Standard IoU NMS (`iou=0.45`) fails when a small pothole box is 85–100% contained inside a larger pothole box (`IoU < 0.45` due to area difference, but `IoMin > 0.85`), causing duplicate concentric boxes on `REGRESSION_005` (`WATER-FILLED POTHOLE (59%)` drawn twice). | **MODERATE** |
| **H. Training Configuration** | Yes | V2 was trained for 15 epochs on CPU with mosaic=0.5, which improved over V1, but included the contaminated crack-as-pothole boxes from `rupesh002` and `RDD2022`. | **MODERATE** |
| **I. Model Overfitting** | Yes | Overfit to dark high-contrast blobs regardless of whether they sit inside drivable asphalt or in upper-frame trees/foliage/vehicles. | **MODERATE** |
| **J. No Road-Region Constraint** | **CRITICAL** | **SMOKING GUN #2**: Pothole detection had **zero road-region gating**. Detections in the upper sky (`y < 0.35`), green vegetation/foliage, or vertical building facades were accepted without penalty. | **CRITICAL** |
| **K. Background/Context Mismatch** | Yes | No cross-model disambiguation existed between the Pothole Detector and the Crack Detector when a candidate box lay directly over an elongated crack network. | **HIGH** |

---

## 2. Dataset Split & Source Distribution Audit (`processed_v2`)

| Split | Total Images | Positive Images | Negative Images | Pothole Instances | RDD2022 India (Pos) | Roboflow rupesh002 | RDD2022 Clean Neg | RDD2022 Crack Neg |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Train** | 1753 | 1027 | 726 | 1584 | 713 | 500 | 0 | 0 |
| **Val** | 361 | 258 | 103 | 394 | 191 | 96 | 0 | 0 |
| **Test** | 215 | 193 | 22 | 315 | 96 | 119 | 0 | 0 |

---

## 3. Duplicate & Leakage Audit
- **Exact MD5 Duplicate Groups**: 101 (Cross-split exact duplicates: **0**)
- **Perceptual dHash Near-Duplicate Groups**: 102 (Cross-split near-duplicates: **0**)
- **Crack + Pothole Co-occurrence in RDD2022 India**: **338** images contain both crack labels (`D00/D10/D20`) and pothole labels (`D40`), creating boundary ambiguity where weathered cracks meet surface spalling.

---

## 4. Suspicious Annotation Audit Findings
- **Upper-Frame / Sky / Background Boxes (`cy < 0.35`)**: **207 instances**
  - Example: `RDD2022_India_Japan_000311.jpg` (`box=[0.8392, 0.1708, 0.085, 0.0183]`)
- **Elongated Linear Crack/Shadow Boxes (`aspect > 4.2` or `< 0.24`)**: **145 instances**
  - Example: `RDD2022_India_India_001890.jpg` (`box=[0.3875, 0.8222, 0.0778, 0.3278]`)
- **Oversized Road Texture / Crack Network Boxes (`area > 30%`)**: **133 instances**
  - Example: `RDD2022_India_Japan_000218.jpg` (`box=[0.5067, 0.8258, 0.9667, 0.3383]`)
- Visual audit samples saved to: `ml/pothole/reports/audit_v2_samples/`

---

## 5. Corrective Action Plan for V3 (`models/pothole_yolo26s_vNEXT.pt`)
1. **Remove Rogue Optical Fallback**: Eliminate `_detect_optical_surface_anomalies()` fallback when the neural model is loaded and predicts 0 potholes.
2. **Sanitize Contaminated Pothole Labels**: Filter out upper-sky/background boxes (`cy < 0.32`), extreme linear crack-like boxes (`aspect > 4.2` or `< 0.24`), and oversized road-texture boxes (`area > 0.30`).
3. **Expand Hard-Negative Training & Categorized Stress-Test Suite**: Add verified hard negatives across all 7 categories (`non_road`, `cracks`, `shadows`, `manholes`, `road_patches`, `wet_roads`, `normal_roads`).
4. **Road-Region Context Gating (`backend/services/road_context.py`)**: Enforce road-surface overlap scoring (`road_overlap_ratio` & `road_context_score`) and containment-aware NMS (`IoMin` suppression).
5. **Pothole vs. Crack Disambiguation**: Suppress pothole candidates that are explained by elongated crack networks without a distinct closed depression.
