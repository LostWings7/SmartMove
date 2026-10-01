# SafeCity Road Condition Intelligence — Water Segmentation (`ml/water`)

## 1. Overview
The **SafeCity Water Segmentation Subsystem** delivers pixel-level segmentation of standing water, road puddles, and waterlogged corridors. It operates as an independent detector alongside the protected Pothole Detector (`models/pothole_yolo26s.pt`), feeding into the Unified Evidence Fusion Engine to calculate road hazard severity and identify **Water-Filled Potholes**.

## 2. Dataset Attribution & Integrity
- **Primary Source**: Puddle-1000 & Curated Road Water Segmentation Benchmark.
- **Annotation Format**: YOLO segmentation polygon masks (`class 0: water_puddle`).
- **Total Images**: 3,774 images across 2,303 distinct scenes/sequences.
- **Polygons**: 7,978 validated water region masks.
- **Leakage Prevention**: Video sequence grouping ensures that all frames from the same driving sequence are partitioned exclusively into Train, Validation, or Held-Out Test.

## 3. Directory Structure
```
ml/water/
├── configs/
│   └── water_data.yaml         # Dataset path and class definition
├── dataset/
│   ├── raw/                    # Downloaded raw archives
│   └── processed/              # Verified images and polygon labels (train/val/test)
├── reports/
│   ├── dataset_report.md       # Dataset statistics and quality report
│   ├── water_evaluation.md     # Pixel IoU, Dice, Precision, Recall benchmark
│   ├── water_evaluation.json   # Machine-readable evaluation results
│   └── annotation_samples/     # Rendered ground-truth verification samples
├── prepare_dataset.py          # Sequence-aware parsing & sanitization pipeline
├── train.py                    # Model training with Intel XPU / CPU fallback
├── evaluate.py                 # Pixel-level evaluation (IoU, Dice, mAP50, mAP50-95)
├── infer.py                    # WaterSegmenter adapter & temporal video tracker
└── README.md                   # This documentation
```

## 4. Key Methodologies

### A. Visible-Road Coverage Calculation
Water coverage is evaluated specifically against the relevant drivable roadway rather than the entire frame area:
$$\text{Water Coverage \%} = \frac{\text{Water Mask Pixels} \cap \text{Road ROI Pixels}}{\text{Relevant Visible Road ROI Pixels}} \times 100$$
The Road ROI is defined using a calibrated perspective drivable corridor trapezoid, avoiding distortion from skies, hoods, or off-road background.

### B. Transparent Operational Severity
| Severity Level | Roadway Water Coverage | Connected Component Area | Operational Impact |
| :--- | :--- | :--- | :--- |
| **LOW** | $< 3.0\%$ | Isolated small puddle | Minimal steering impact |
| **MODERATE** | $3.0\% - 10.0\%$ | Medium water patch | Hydroplaning risk at speed |
| **HIGH** | $10.0\% - 25.0\%$ | Multi-patch or large puddle | Lane impediment, active hazard |
| **CRITICAL** | $> 25.0\%$ | Extensive waterlogging | Road impassable / vehicle stall risk |

*Note: Thresholds are configurable operational indicators until municipal sensor depth calibration is integrated.*

### C. Temporal Video Confirmation
Consecutive frame detections of water are grouped across a sliding window to generate a single consolidated water hazard event with start frame, representative frame, end frame, and maximum coverage.

## 5. Usage Commands

```bash
# 1. Prepare dataset and generate data quality report
python ml/water/prepare_dataset.py

# 2. Train YOLO26s-seg model
python ml/water/train.py --epochs 20 --imgsz 512 --batch 8 --device cpu

# 3. Evaluate on held-out test split
python ml/water/evaluate.py --model models/water_yolo26s_seg_v1.pt

# 4. Run standalone inference
python ml/water/infer.py
```
