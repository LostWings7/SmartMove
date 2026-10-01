# SafeCity Road Condition Intelligence Deployment Summary

## Executive Overview
The SafeCity Road Condition Intelligence subsystem expands the SafeCity Loop V2 / SmartMove platform with multimodal road hazard perception:
1. **Protected Pothole Subsystem**: Preserved intact without modifications (`models/pothole_yolo26s.pt`).
2. **Waterlogging / Ponding Segmentation Subsystem**: Pixel-accurate puddle & ponding detection with visible-road ROI coverage % and operational severity assessment (`models/water_yolo26s_seg_v1.pt`, OpenVINO IR at `models/water_yolo26s_seg_v1_openvino_model/`).
3. **Road Crack Detection Subsystem**: Dedicated detector for longitudinal, transverse, and alligator cracking trained on RDD2022 (`models/road_crack_yolo26s_v1.pt`, OpenVINO IR at `models/road_crack_yolo26s_v1_openvino_model/`).
4. **Multimodal Evidence Fusion Engine**: Fuses spatial overlap between detected potholes and water masks (`Pothole ∩ Water >= 20%`) to identify critical submerged/water-filled potholes, sub-additively calculating compound hazard risk scores without double counting.
5. **Production Backend & Dynamic Discovery**: Extensible adapter architecture in `backend/services/road_condition_intelligence.py` querying hardware capabilities (CPU/GPU/NPU) and model availability via `GET /api/ai/status` and `POST /api/road-condition/analyze`.
6. **Unified Operations Dashboard**: Interactive frontend in `frontend/src/pages/AIVision.tsx` and `RoadConditionInspectionViewport.tsx` featuring layer toggling (Fused, Potholes, Water Mask, Cracks), side-by-side comparison, defect KPI cards, and evidence fusion audit trail inspector.

---

## 1. Subsystem Architecture & Integrity

```
                              ROAD CONDITION AI
                                      │
            ┌─────────────────────────┼─────────────────────────┐
            ↓                         ↓                         ↓
     POTHOLE DETECTOR           WATER SEGMENTER          CRACK DETECTOR
   (YOLO26s - Protected)       (YOLO26s-seg New)        (YOLO26s New)
            │                         │                         │
            └─────────────────────────┼─────────────────────────┘
                                      ↓
                            EVIDENCE FUSION ENGINE
                                      │
             ┌────────────────────────┴────────────────────────┐
             ↓                                                 ↓
   HAZARD RISK ESTIMATOR                             DEFECT VISUALIZER
 (Sub-additive non-linear)                       (Multi-layer RGBA overlay)
             │                                                 │
             └────────────────────────┬────────────────────────┘
                                      ↓
                            API ROUTER & DASHBOARD
```

### Protection of Pothole Subsystem
- **Model Path**: `models/pothole_yolo26s.pt` (SHA256 intact, verified unmodified).
- **Backend Service**: `backend/services/pothole_detection.py` and existing endpoints `/api/detect-potholes` continue functioning without alterations.
- **Reference Doc**: `ml/road_condition_reports/pothole_reference.md`.

---

## 2. Model Performance Benchmarks

### Waterlogging Segmentation (`water_yolo26s_seg_v1`)
- **Evaluation Set**: 567 held-out road frames from Puddle-1000 with sequence-aware grouping to prevent leakage.
- **Pixel IoU**: 0.4511 (Mean per-image: 0.6616)
- **Pixel Dice / F1**: 0.6217 (Mean per-image: 0.7491)
- **Pixel Recall**: 0.7945 (79.45% of real water surface detected)
- **Pixel Precision**: 0.5107
- **Mask mAP50**: 0.5842 (58.42%)
- **Mask mAP50-95**: 0.3409 (34.09%)

### Road Crack Detection (`road_crack_yolo26s_v1`)
- **Evaluation Set**: 1,319 held-out road frames from RDD2022.
- **Overall Precision**: 0.4694
- **Overall Recall**: 0.2933
- **Overall F1**: 0.3610
- **Overall mAP50**: 0.2022
- **Class Breakdown**:
  - `longitudinal_crack` (D00): P=0.5006, R=0.4041, F1=0.4472, mAP50=0.2906
  - `transverse_crack` (D10): P=0.5153, R=0.3976, F1=0.4489, mAP50=0.2725
  - `alligator_crack` (D20): P=0.3922, R=0.0781, F1=0.1303, mAP50=0.0435
- **Clean Road Specificity**: 83.33% (rejection of crack false alarms on smooth asphalt).

### OpenVINO Hardware Acceleration
Exported to OpenVINO Intermediate Representation (IR FP16/FP32):
- **Water Model**:
  - OpenVINO Latency: **78.64 ms** (12.7 FPS) on Intel Core Ultra 5 225H.
  - PyTorch Latency: 180.70 ms (OpenVINO speedup: **2.30x**).
  - Numerical Mask Agreement (IoU): **0.9997 (99.97%)**.
- **Crack Model**:
  - OpenVINO Latency: **69.95 ms** (14.3 FPS) on Intel Core Ultra 5 225H.
  - PyTorch Latency: 145.20 ms (OpenVINO speedup: **2.08x**).
  - Numerical Box Agreement: **100.0%**.

---

## 3. Multimodal Evidence Fusion & Risk Calculation

### Fusion Logic
1. **Spatial Intersection**:
   $$\text{IoU}_{\text{pothole}} = \frac{\text{Area}(\text{Pothole BBox} \cap \text{Water Mask})}{\text{Area}(\text{Pothole BBox})}$$
2. **Threshold**:
   - If $\text{IoU}_{\text{pothole}} \ge 0.20$ (20% overlap), the pothole is classified as `water_filled_pothole` with status `"submerged"` or `"partially_submerged"`.
   - If $\text{IoU}_{\text{pothole}} < 0.20$, the pothole is classified as `"dry"` (perimeter specular reflections are rejected).
3. **Compound Risk Formulation**:
   $$R_{\text{fused}} = \min\left(100.0, R_{\max} + 0.12 \times \sum_{i \ne \max} R_i\right)$$
   This guarantees sub-additivity ($R_{\text{fused}} < \sum R_i$) and prevents double-counting correlated visual evidence.

---

## 4. REST API Endpoints

### `POST /api/road-condition/analyze`
Accepts `multipart/form-data` with `file`, optional `confidence_threshold` (default 0.25), `pothole_water_overlap_thresh` (default 0.20), and `save_evidence` (default true).

Returns `RoadConditionAnalysisOut`:
- `summary`: counts of potholes, water hazards, cracks, and water-filled potholes.
- `water_analysis`: overall visible-road coverage %, operational severity (`LOW`, `MODERATE`, `HIGH`, `CRITICAL`), and individual `regions`.
- `potholes`: list of detected potholes with spatial coordinates and confidence.
- `water_filled_potholes`: fused intersections including `overlap_ratio`, `pothole_confidence`, `water_confidence`, and compound severity.
- `cracks`: list of verified crack classifications (`longitudinal_crack`, `transverse_crack`, `alligator_crack`).
- `overall_hazard_level` and `compound_risk_score` (0–100).
- `evidence_overlays`: URLs to rendered visualization layers (`fused`, `water_mask`, `potholes`, `cracks`).
- `execution_profile`: per-subsystem inference latency (ms) and accelerator type (`openvino` / `pytorch-cpu`).

### `GET /api/ai/status`
Returns real-time deployment status:
- Supported features: `pothole_detection`, `water_segmentation`, `road_crack_detection`, `multimodal_fusion`.
- Model availability and paths.
- Active accelerator backend.
