# SafeCity Road Crack Subsystem — Dataset & Annotation Visual Audit (Sections 14–16)

## 1. Why the V1 Crack Detector Missed Obvious Large Cracks (Section 14)

Our root-cause investigation into `models/road_crack_yolo26s_v1.pt` identified **five compounding technical failures**:

1. **Severe Under-Training & Frozen Backbone (`models/road_crack_model.json`)**:
   - Trained for only **3 epochs** (`epochs_requested: 3`).
   - Froze all **10 backbone layers** (`freeze: 10`), preventing the early convolutional layers from learning fine-grained dark fissure edges on gray asphalt.
   - Trained at a heavily downsampled resolution of **`384x384`**, blurring out thin crack structures, while running inference at `512x512`.
2. **Shard Starvation & Top-Down Drone Viewpoint Bias (`ml/cracks/prepare_dataset.py`)**:
   - In `prepare_dataset.py` (line 170), the candidate collection loop broke early inside `shard_000` (`China_Drone` and `China_MotorBike`), downloading only **1 image** from `shard_004` (Japan forward-facing road cameras) and **1 image** from `shard_005` (Japan/Norway forward-facing road cameras).
   - As a result, **46.8% of the dataset was `China_Drone`** (top-down aerial views) and **19.7% was `China_MotorBike`**, starving the model of forward-facing perspective views where crack networks span the road width (`ROAD_CONDITION_REGRESSION_001`).
3. **Directory Non-Clean Split Leakage**:
   - Because `PROCESSED_DIR` was not wiped before re-running `prepare_dataset.py`, **985 images** overlapped between `train` and `test`, and **284 images** overlapped between `train` and `val`.
4. **Severe Class Imbalance & Scarcity of Large Alligator Cracks**:
   - `alligator_crack` had the fewest instances in training and achieved only **7.81% recall (`mAP50 = 0.0435`)** on the held-out evaluation.
5. **Bounding-Box Representation Limit on Branching Crack Networks**:
   - On `ROAD_CONDITION_REGRESSION_001`, V1 predicted 3 boxes covering the alligator/branching crack network at confidences `0.2076`, `0.1688`, and `0.1326`—all rejected by the hardcoded `0.25` threshold.

---

## 2. Per-Class Quantitative Dataset Audit (Section 15)

| Crack Class | Images | Instances | Mean Box Area | Median Box Area | Min Area | Max Area | Aspect Ratio (P10 / Med / P90) | Thin/Long (`AR>=3` or `<=0.33`) | Large Cracks (`Area>=15%`) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `longitudinal_crack` | 2106 | 3182 | 5.15% | 2.69% | 0.0610% | 67.1% | 0.11 / 0.28 / 1.23 | 1922 (60.4%) | 244 (7.7%) |
| `transverse_crack` | 1570 | 2060 | 4.24% | 2.10% | 0.0973% | 67.2% | 2.46 / 4.82 / 8.66 | 1720 (83.5%) | 110 (5.3%) |
| `alligator_crack` | 1014 | 1219 | 13.39% | 9.03% | 0.2136% | 88.3% | 0.66 / 1.43 / 3.16 | 175 (14.4%) | 404 (33.1%) |

---

## 3. Annotation Visual Audit of 105 Training Images (Section 16)

We rendered **105 random training annotations** to `ml/cracks/reports/annotation_audit/` and ran automated quality checks across all splits:

- **Loose / Oversized Bounding Boxes (`area > 40%`)**: **77 instances**
  - When a diagonal or branching crack crosses a lane, a single axis-aligned bounding box encloses 70–90% clean pavement and only 10–30% crack pixels, diluting feature supervision for object detectors.
- **Tiny Fragment Boxes (`area < 0.15%`)**: **23 instances**
- **Fragmented Long Cracks ($\ge 4$ tiny boxes on a single crack)**: **10 images**
  - Several `China_Drone` annotations split one continuous longitudinal crack into 5–12 tiny disconnected boxes, confusing NMS and penalizing single-box predictions.
- **Upper-Frame Background Boxes (`y_min < 0.25`)**: **776 instances**

---

## 4. Architectural & Dataset Remediation Plan (Sections 17–24)

1. **Clean, Leakage-Free Multi-Country Forward-Facing Dataset (`ml/cracks/dataset/processed_v2`)**:
   - Wipe stale split directories completely before populating splits.
   - Pull balanced forward-facing perspective road imagery from `shard_002` & `shard_003` (India/Japan), `shard_004` & `shard_005` (Japan/Norway), and `shard_001`/`shard_008` (Czech/USA), capping `China_Drone` so forward-facing road views make up >80% of the dataset.
   - Ensure strong representation of **large crack networks (`alligator_crack`, long `longitudinal_crack`, wide `transverse_crack`)** + **crack-specific hard negatives** (tar lines, lane markings, shadows, pavement seams, gravel, pothole edges).
2. **Controlled Architecture Experiment (Section 17)**:
   - Compare **Option A (YOLO26s Detection)** vs **Option B (YOLO26s-seg Segmentation)** where crack fissure masks are extracted within ground-truth crack regions, evaluating both box recall and pixel/mask localization on large branching crack networks.
3. **Multi-Scale / Tiled Inference & Resolution Sweep (Sections 22–24)**:
   - Compare `512`, `640`, and `768` resolutions and tiled/multi-scale inference for thin/elongated crack structures.
