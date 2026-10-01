# SafeCity Road Condition Intelligence — Road Crack Detection (`ml/cracks`)

## 1. Overview
The **SafeCity Road Crack Detection Subsystem** provides automated object detection for structural pavement cracking. It identifies three distinct, validated road damage categories from official CRDDC / RDD2022 taxonomy:
- `0: longitudinal_crack` (RDD2022 D00) — Cracks extending along the direction of travel (wheel paths).
- `1: transverse_crack` (RDD2022 D10) — Cracks extending perpendicular across traffic lanes (thermal expansion/contraction).
- `2: alligator_crack` (RDD2022 D20) — Interconnected fatigue cracking indicating sub-base pavement structural failure.

*Protected Subsystem Note: Potholes (D40) are strictly excluded from this model to preserve separation of concerns from the existing protected pothole model (`models/pothole_yolo26s.pt`). No speculative categories (e.g. "damaged surface", "rough road") are reported.*

## 2. Dataset & Attribution
- **Dataset**: Road Damage Dataset 2022 (RDD2022 / CRDDC2022)
- **Source**: Seki Lab / Tokyo University / CRDDC2022 ([sekilab/RoadDamageDetector](https://github.com/sekilab/RoadDamageDetector))
- **License**: Creative Commons Attribution-ShareAlike 4.0 International (CC BY-SA 4.0)
- **Geographic Representation**: Multi-country coverage (India, Japan, United States, Czech Republic) to ensure robustness against differing pavement binders, aggregates, and lane demarcations.
- **Leakage Prevention**: Grouped sequence block partitioning isolates consecutive inspection frames into unique splits.

## 3. Directory Layout
```
ml/cracks/
├── configs/
│   └── crack_data.yaml         # YOLO dataset configuration
├── dataset/
│   ├── raw/                    # Raw multi-country RDD2022 images and label files
│   └── processed/              # Verified splits (train/val/test)
├── reports/
│   ├── dataset_report.md       # Dataset statistics and geographic distribution
│   ├── crack_evaluation.md     # Per-class precision, recall, F1, mAP, and hard negative metrics
│   ├── crack_evaluation.json   # Machine-readable benchmark data
│   └── annotation_samples/     # Verification sample images
├── prepare_dataset.py          # Data curation, filtering, and sequence partitioning
├── train.py                    # YOLO26s training pipeline
├── evaluate.py                 # Per-class mAP and hard negative stress testing
├── infer.py                    # CrackDetector adapter and severity classifier
└── README.md                   # This documentation
```

## 4. Explainable Severity Logic
| Crack Category | Base Severity | Trigger Condition for Escalation | Operational Impact |
| :--- | :--- | :--- | :--- |
| **`longitudinal_crack`** | **LOW** | Bounding box area $> 15\%$ | Water ingress into road base |
| **`transverse_crack`** | **MEDIUM** | Bounding box area $> 12\%$ | Ride quality impact, vehicle bump |
| **`alligator_crack`** | **HIGH** | Bounding box area $> 8\%$ $\rightarrow$ **CRITICAL** | Sub-base structural failure, imminent pothole creation |

## 5. Usage Commands
```bash
# 1. Prepare multi-country dataset with hard negatives
python ml/cracks/prepare_dataset.py

# 2. Train YOLO26s detector
python ml/cracks/train.py --epochs 15 --imgsz 512 --batch 16 --device cpu

# 3. Evaluate on held-out test split
python ml/cracks/evaluate.py --model models/road_crack_yolo26s_v2.pt

# 4. Run standalone inference
python ml/cracks/infer.py
```

---

## 6. Major Accuracy Overhaul (v2.0.0): Option A vs. Option B & Tiled Multi-Scale Inference

### Problem Statement & Audit Findings
On `ROAD_CONDITION_REGRESSION_001.jpg`, the baseline crack detector failed to detect obvious, extensive road cracking (reported ZERO cracks). A systematic dataset audit (`ml/cracks/reports/crack_audit_v2.md` + 105 visual verification audits) revealed:
1. **Resolution Downscaling Loss:** Resizing large road scenes to a single 512px box compresses thin crack fissures below detection thresholds.
2. **Annotation Contamination:** Baseline annotations had class overlap and poor boundary fidelity.
3. **Absence of Sub-Tile Inference:** Forward-facing dashcam imagery concentrates road cracks in the bottom 60% of the frame. Without tile-level examination, thin linear cracks are washed out.

### Controlled Experiment: Option A vs. Option B
We implemented and trained two distinct architectures on clean multi-country RDD2022 V2 data (`ml/cracks/dataset/processed_v2/`):
- **Option A Detection (`models/road_crack_yolo26s_v2.pt`):** Standard bounding box detection.
- **Option B Segmentation (`models/road_crack_yolo26s_seg_v2.pt`):** Instance segmentation producing polygonal crack boundaries.

| Configuration | Best F1 | Precision | Recall | Large Crack Recall | Thin/Small Crack Recall | Mean Latency |
|---|---|---|---|---|---|---|
| **Option A Det V1 Baseline** | 0.4350 | 0.7742 | 0.3025 | 47.37% | 22.50% | 59.2 ms |
| **Option A Det V2 Single-Scale** | 0.4388 | 0.5584 | 0.3613 | 60.53% | 27.50% | 59.1 ms |
| **Option A Det V2 Tiled Multi-Scale** | **0.4974** | 0.4398 | **0.5723** | **79.31%** | **45.00%** | 139.5 ms |
| **Option B Seg V2 (Segmentation)** | 0.3562 | 0.4634 | 0.2893 | 64.71% | 21.25% | 86.5 ms |

### Architectural Decision & Production Deployment
- **Primary Detector:** Option A Detection V2 with Tiled Multi-Scale inference is deployed in production (`models/road_crack_yolo26s_v2.pt`), achieving **79.31% recall on large cracks (+67.4% gain)** and **57.23% overall recall (+89.2% gain)**.
- **Auxiliary Mask / Boundary Refinement:** Option B segmentation (`models/road_crack_yolo26s_seg_v2.pt`) provides polygonal mask coordinates for structural fissure verification.
- **OpenVINO Acceleration (Intel Core Ultra 5 225H):**
  - CPU: **26.17 ms (38.2 FPS)**
  - Intel Arc iGPU: **5.31 ms (188.3 FPS)**
  - Intel AI Boost NPU: **8.14 ms (122.9 FPS)**

