# SafeCity Loop V2 — Traffic AI Dataset Quality Report

**Generated:** 2026-09-30 16:20:51  
**Primary Dataset:** Berkeley DeepDrive (BDD100K) — Official Public Release  
**License:** BSD 3-Clause Open-Access License  
**Attribution:** Fisher Yu et al., UC Berkeley DeepDrive Consortium (https://bdd-data.berkeley.edu)  
**Geographic Coverage:** United States Urban / Suburban Roadways (San Francisco, New York, Bay Area)  
**Camera / Viewpoint:** Windshield-mounted forward-facing driving camera (720p HD)  

---

## 1. Dataset Overview & Temporal Leakage Prevention

| Metric | Value | Design Context |
|---|---|---|
| **Total Verified Frames** | `704` | Filtered for valid dimensions and non-corrupt status |
| **Corrupted Files Rejected** | `0` | Excluded from pipeline |
| **Total Sequences (Videos)** | `103` | Video-level source identity preserved |
| **Total Road User Annotations** | `10839` | Focused strictly on dynamic traffic participants |
| **Average Objects / Frame** | `15.40` | Dense mixed urban traffic |
| **Multi-Frame Sequences Preserved** | `100` | Dedicated MOT/TTC tracking validation sets |

### Leakage Prevention Guarantee
- **Split Strategy:** Strict **Video Sequence Hashing** (by video UUID prefix before `-`).
- Consecutive frames from the same video sequence are **never distributed across multiple splits**.
- Train sequences (`72`), Validation sequences (`15`), and Test sequences (`16`) are 100% mutually exclusive.

---

## 2. Train / Validation / Test Breakdown

| Split | Sequences | Frames | Total Objects | Mean Objects/Frame |
|---|---|---|---|---|
| **Train (70%)** | `72` | `507` | `7834` | `15.45` |
| **Val (15%)** | `15` | `104` | `1487` | `14.30` |
| **Test (15%)** | `16` | `93` | `1518` | `16.32` |

---

## 3. Road-User Class Distribution

| Class ID | SafeCity Class | BDD100K Label | Instances | Percentage | Category Role |
|---|---|---|---|---|---|
| 0 | `pedestrian` | `pedestrian` | `2563` | `23.6%` | Vulnerable Road User (VRU) |
| 1 | `rider` | `rider` | `104` | `1.0%` | VRU (Cycle/Motorcycle Rider) |
| 2 | `car` | `car` | `7404` | `68.3%` | Motor Vehicle |
| 3 | `truck` | `truck` | `350` | `3.2%` | Heavy Vehicle |
| 4 | `bus` | `bus` | `177` | `1.6%` | Public Transit Vehicle |
| 5 | `motorcycle` | `motorcycle` | `40` | `0.4%` | High-Vulnerability Powered 2-Wheeler |
| 6 | `bicycle` | `bicycle` | `201` | `1.9%` | VRU (Non-Motorized Transport) |

---

## 4. Object Size Distribution (COCO Standard Metric)

- **Small Objects** ($Area < 32^2 px$): `4437` (`40.9%`)
- **Medium Objects** ($32^2 px \le Area \le 96^2 px$): `4493` (`41.5%`)
- **Large Objects** ($Area > 96^2 px$): `1909` (`17.6%`)

---

## 5. Environmental & Hard-Case Metadata Distribution

### Weather
- **Clear**: `310` frames (44.0%)
- **Partly cloudy**: `45` frames (6.4%)
- **Undefined**: `117` frames (16.6%)
- **Rainy**: `48` frames (6.8%)
- **Overcast**: `96` frames (13.6%)
- **Snowy**: `87` frames (12.4%)
- **Foggy**: `1` frames (0.1%)

### Time of Day
- **Night**: `244` frames (34.7%)
- **Daytime**: `403` frames (57.2%)
- **Dawn/dusk**: `56` frames (8.0%)
- **Undefined**: `1` frames (0.1%)

### Road Scene
- **City street**: `511` frames (72.6%)
- **Residential**: `77` frames (10.9%)
- **Highway**: `113` frames (16.1%)
- **Parking lot**: `1` frames (0.1%)
- **Undefined**: `2` frames (0.3%)

---

## 6. Verification Samples
Visual annotation overlays were checked and saved to `ml/traffic/reports/annotation_samples/`. Bounding box coordinates were clamped within `[0, 1]`, and labels follow YOLO standard format.
