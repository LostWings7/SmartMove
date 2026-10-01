# SafeCity Road Condition Intelligence — Road Crack Dataset Quality Report

## 1. Executive Summary
This report documents the curation, multi-country distribution, and instance statistics of the SafeCity Road Crack Dataset prepared from official RDD2022 sources (CRDDC2022 / Seki Lab, CC BY-SA 4.0).

- **Target Model**: `models/road_crack_yolo26s_v1.pt` (Object Detection)
- **Supported Classes**:
  - `0: longitudinal_crack` (RDD2022 D00)
  - `1: transverse_crack` (RDD2022 D10)
  - `2: alligator_crack` (RDD2022 D20)
- **Potholes (D40)**: Intentionally excluded and delegated to the protected `pothole_yolo26s.pt` model.
- **Unsupported Classes Omitted**: No speculative "damaged surface", "rough road", or "uneven pavement" labels.

## 2. Dataset Split & Class Distribution

| Split | Total Images | Positive Images | Negative Images | Longitudinal Crack | Transverse Crack | Alligator Crack | Total Instances |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Train** | 1959 | - | - | 1404 | 874 | 586 | 2864 |
| **Validation** | 418 | - | - | 280 | 164 | 160 | 604 |
| **Test** | 423 | - | - | 395 | 233 | 119 | 747 |
| **TOTAL** | **2800** | **2400** | **400** | **2079** | **1271** | **865** | **4215** |

## 3. Geographic & Environmental Representation
The dataset incorporates genuine road images from multiple jurisdictions to ensure model generalization across differing asphalt blends, pavement age, weathering, and road marking standards:
- **China_Drone**: 1310 images (46.8%)
- **India**: 820 images (29.3%)
- **China_MotorBike**: 552 images (19.7%)
- **Japan**: 118 images (4.2%)

## 4. Sequence-Leakage Prevention
Images are strictly partitioned by geographic sequence block (prefix + road drive sequence ID). Consecutive road inspection frames from the same drive are isolated in a single split, guaranteeing zero test set contamination.

## 5. Bounding Box & Scale Characteristics
- **Total Validated Crack Bounding Boxes**: 4,215
- **Median Box Area (% of frame)**: 3.15%
- **Mean Box Area (% of frame)**: 6.66%
- **Resolutions**:
  - `512x512`: 1862 images (66.5%)
  - `720x720`: 820 images (29.3%)
  - `600x600`: 117 images (4.2%)
  - `1024x1024`: 1 images (0.0%)

## 6. Hard Negative Mining
400 verified negative images were incorporated to penalize false alarms on:
- Pavement surface texture and coarse gravel
- Black bitumen sealant lines from prior crack sealing
- Clean white and yellow lane markings
- Tree canopy and roadside infrastructure shadows
- Wet surface specular reflections

## 7. Sample Visualizations
12 annotated inspection images have been rendered and saved to `ml/cracks/reports/annotation_samples/`.
