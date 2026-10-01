# SafeCity Loop V2 — Pothole Dataset Verification Report

Generated from verified dataset in `ml/pothole/dataset/processed/`.

## 1. Split & Instance Breakdown

| Split | Images | Positive Images | Negative (Background) Images | Pothole Instances |
| :--- | :--- | :--- | :--- | :--- |
| **Train** | 1213 | 1053 | 160 | 1612 |
| **Val** | 287 | 261 | 26 | 398 |
| **Test** | 215 | 200 | 15 | 322 |
| **Total** | **1715** | **1514** | **201** | **2332** |

## 2. Dataset Sources & Licensing

- **RDD2022 (India Subset):** 1000 images (58.3%)
- **Secondary Pothole Dataset (Roboflow/rupesh002):** 715 images (41.7%)
- **Duplicate Images Detected:** 101

## 3. Bounding Box Geometry Statistics

| Metric | Min | Max | Mean | Median | Std Dev |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Normalized Width | 0.0234 | 0.9967 | 0.282 | 0.197 | 0.228 |
| Normalized Height | 0.01 | 0.9906 | 0.2052 | 0.15 | 0.1752 |
| Normalized Area | 0.0004 | 0.8424 | 0.0851 | 0.0269 | 0.1319 |
| Normalized Aspect_ratio | 0.1605 | 25.8498 | 1.7431 | 1.308 | 1.4316 |

## 4. Top Image Resolutions

- `720x720`: 820 images
- `640x640`: 715 images
- `600x600`: 179 images
- `1024x1024`: 1 images

## 5. Visual Samples

Generated ground truth annotation visualizers in `C:\Users\dvasa\Projects\SmartMove\ml\pothole\reports\visual_samples`.
