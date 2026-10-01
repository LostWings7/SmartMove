# SafeCity Road Condition Intelligence — Water Segmentation Dataset Quality Report

## 1. Executive Summary
This report documents the preparation, validation, and statistical distribution of the SafeCity Water / Ponding Segmentation Dataset used to train the YOLO26s-seg water hazard segmentation model.

- **Primary Hazard Class**: `water_puddle` (Standing water, puddles, waterlogged road surfaces)
- **Annotation Type**: Pixel-level closed polygon segmentation masks (YOLO polygon format)
- **Target Deployment**: SafeCity Loop V2 Edge Inference on Intel Core Ultra 5 225H / OpenVINO

## 2. Dataset Composition & Splits

| Split | Total Images | Positive Images (With Water) | Negative / Background Images | Total Water Polygons |
| :--- | :--- | :--- | :--- | :--- |
| **Train** | 2641 | 2632 | 9 | 5688 |
| **Validation** | 566 | 559 | 7 | 1127 |
| **Test** | 567 | 565 | 2 | 1163 |
| **TOTAL** | **3774** | **3756** | **18** | **7978** |

## 3. Data Leakage Prevention
- **Sequence Grouping**: Frames extracted from video sequences (e.g. `01_mp4`, `yt-SVR2JpnvH4k`, `222_mp4`, KakaoTalk vehicle runs) were grouped by scene identifier.
- **Leakage Integrity**: All consecutive or correlated frames from the same video/road run were assigned exclusively to either Train, Validation, or Test. No near-identical sequence frames appear across splits.

## 4. Water Region & Coverage Statistics
- **Total Validated Water Polygons**: 7,978
- **Median Water Region Area (% of image)**: 0.94%
- **Mean Water Region Area (% of image)**: 5.48%
- **25th Percentile Area (Small Puddles)**: 0.06%
- **75th Percentile Area (Medium Ponding)**: 5.14%
- **95th Percentile Area (Severe Waterlogging)**: 27.48%

## 5. Image Resolutions & Quality Verification
- **Image Integrity Checks**: 100% of images verified with Pillow and OpenCV.
- **Corrupt / Unreadable Files Filtered**: 0
- **Degenerate Polygons Sanitized**: 3
- **Resolution Distribution**:
  - `256x256`: 2253 images (59.7%)
  - `1280x1280`: 1355 images (35.9%)
  - `416x416`: 166 images (4.4%)

## 6. Environmental Diversity
The dataset encompasses:
- Asphalt road surfaces (urban, suburban, highway)
- Variable lighting (overcast, wet reflection, direct sunlight glare, dusk/dawn)
- Wet road sections with and without standing water (hard negatives included)
- Multiple disconnected puddle instances per frame
- Vehicle splash and tire tracks around water boundaries

## 7. Visual Inspection Samples
12 annotated ground-truth verification samples were rendered and saved to `ml/water/reports/annotation_samples/` with transparent masks.
