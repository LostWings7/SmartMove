# SafeCity Loop V2 — AI-Powered Urban Road-Safety Platform

> **Detect danger. Predict risk. Prioritize action. Build safer cities.**

**Core Workflow:**
`DETECT → ANALYZE → PREDICT → PRIORITIZE → PREVENT → REPAIR → MEASURE`

---

> ⚠️ **DEMO DATA DISCLAIMER**
> All road names, hazard coordinates, near-miss events, and repair recommendations present in this prototype are **realistic but fictional demo data** (`[DEMO]`). They do not represent real municipal or road safety authority records.

---

## 🏛 Architecture & Tech Stack

### Backend
- **Framework:** FastAPI (Python 3.10+, tested on Python 3.14)
- **Database:** SQLite (`safecity.db`) via SQLAlchemy ORM
- **Computer Vision & AI Inference:**
  - **Ultralytics YOLOv8** (Local CPU/GPU inference, no cloud AI dependencies)
  - **Pothole Detection Pipeline:** Configurable via `MODEL_PATH`. Includes bounding box extraction, confidence rating, severity classification, and prototype risk scoring.
  - **Demo AI Mode Architecture:** If custom pothole weights are not configured, runs uncalibrated optical surface anomaly analysis with transparent labels.
  - **Traffic Near-Miss Pipeline:** YOLOv8 + **ByteTrack** multi-object tracking (vehicles, pedestrians, motorcycles, cyclists) + **Time-To-Collision (TTC)** calculation. Flags potential near misses with TTC < 2.0s with wording: *AI-assisted near-miss detection*.
  - **Static Media Server:** Original and processed detection images and video footage served via `/uploads`.

### Frontend
- **Framework:** React 18 + TypeScript + Vite
- **Styling:** Tailwind CSS (Navy/Charcoal command center palette: `#0a0f1e`, `#111827`, `#1a2236`)
- **Mapping:** OpenStreetMap & Leaflet interactive abstraction (`MapProvider.tsx`, swappable to Google Maps via `VITE_MAP_PROVIDER=google`)
- **Data & State:** TanStack Query v5 + Axios
- **Analytics:** Recharts

---

## 👁 AI Vision Features (Part 2)

### 1. Pothole Detection (`POST /api/potholes/analyze`)
- **Input:** Roadway photo (JPG, PNG, WEBP, up to 15MB).
- **Processing:** Local YOLOv8 inference pipeline.
- **Outputs:**
  - Bounding box coordinates (`[x1, y1, x2, y2]`)
  - Detection confidence percentage
  - Severity classification: `LOW` / `MEDIUM` / `HIGH` / `CRITICAL`
  - Prototype Risk Score:
    $$\text{Risk Score} = (\text{Severity Weight} \times 0.6) + (\text{Confidence} \times 100 \times 0.4)$$
- **Persistence:** Automatically saves newly detected potholes to the `Hazard` database table linked to the road, updating the dashboard, risk score, and interactive map.
- **Visualizer:** Side-by-side view of Original Image vs. Detection Overlay with bounding boxes and defect badges.

### 2. Traffic Near-Miss Detection (`POST /api/traffic/analyze`)
- **Input:** Traffic junction video (MP4, MOV, AVI, up to 50MB).
- **Processing:** OpenCV frame extraction + YOLOv8 + ByteTrack multi-object tracking.
- **Tracked Classes:** Vehicles (cars, buses, trucks), pedestrians, cyclists, motorcycles.
- **TTC Conflict Model:**
  $$\text{TTC} = \frac{D(t)}{V_{\text{rel}}} \quad (\text{flagged when } \text{TTC} < 2.0\text{s})$$
- **Outputs:** Annotated video, keyframe conflict snapshots, tracked user counts, and chronologically indexed near-miss events with timestamps (`00:00:17`), object types (`Vehicle + Pedestrian`), TTC (`1.4 sec`), risk level (`HIGH`), and conflict zones.

### 3. Road Condition Intelligence & Evidence Fusion (`POST /api/road-condition/analyze`)
A unified multi-modal road distress perception system operating across three core hazard categories:
- **A. Potholes:** Protected deep cavity detection with depth estimation and cavity depression analysis.
- **B. Waterlogging / Ponding:** Semantic pixel-level segmentation using `water_yolo26s_seg_v1.pt`, reporting water coverage %, connected component regions, and dry-crack reflection suppression.
- **C. Road Surface Cracks:** Strictly supported CRDDC / RDD2022 categories:
  - `longitudinal_crack` (D00): Wheel path linear distress.
  - `transverse_crack` (D10): Thermal expansion perpendicular fissures.
  - `alligator_crack` (D20): Structural fatigue mesh indicating sub-base failure.
- **Shared RoadContextService Gating:** Dynamic camera-adaptive segmentation (dashcam vs. downward surface) that masks sky, clouds, roadside tree canopy, banana leaves, and vehicle bodies, preventing false positives before NMS.
- **Explicit 4-Case Pothole-vs-Crack Disambiguation:**
  - *Case A:* Elongated cracks without cavity drop assigned to Crack Intelligence.
  - *Case B:* Compact depressions with rim differential confirmed as Potholes.
  - *Case C:* Potholes co-occurring inside crack networks preserved without double-counting severity.
  - *Case D:* Potholes overlapping water masks ($\ge 20\%$) fused into high-severity **Water-Filled Potholes**.
- **Spatial Deduplication & Multi-Layer Evidence:** Overlapping multi-scale proposals merged via spatial clustering. Generates 4 downloadable visual layers: `fused_analysis`, `potholes_layer`, `water_layer`, `cracks_layer`.

---

## 🚀 Quick Start (Run Locally)

### 1. Backend Setup & Run
From the root directory:
```bash
# Install Python dependencies (or: uv pip install -r requirements.txt)
pip install -r requirements.txt

# Start FastAPI server
python -m uvicorn backend.main:app --reload --port 8000
```
- API Docs & Swagger UI: `http://localhost:8000/docs`
- Health check: `http://localhost:8000/health`

### 2. Frontend Setup & Run
In a separate terminal, from `frontend/`:
```bash
cd frontend

# Install packages
npm install   # (on Windows PowerShell: cmd.exe /c npm install)

# Start Vite dev server
npm run dev
```
- Open browser at: `http://localhost:5173`

---

## ⚙️ Model Configuration & Acceleration

Model weights and deployment artifacts are located in `models/`:
- **Protected Pothole Detector (Baseline V2):** `models/pothole_yolo26s.pt` (Ultralytics YOLO26s PyTorch / OpenVINO).
- **Sanitized Pothole Detector (VNEXT):** `models/pothole_yolo26s_vNEXT.pt` (Trained on 660 hard negatives; −55.8% false positives).
- **Waterlogging / Ponding Segmentation:** `models/water_yolo26s_seg_v1.pt` (YOLO26s-seg trained on Puddle-1000).
- **Road Crack Detector (V2 Multi-Scale):** `models/road_crack_yolo26s_v2.pt` & `models/road_crack_yolo26s_seg_v2.pt` (Trained on clean multi-country RDD2022; 79.31% large-crack recall).
- **Traffic Safety Detector (V2):** `models/traffic_yolo26s_v2.pt` (7 classes for ByteTrack surrogate conflict analysis).
- **OpenVINO Hardware Acceleration (Intel Core Ultra 5 225H / Arc iGPU / AI Boost NPU):**
  - **Pothole VNEXT:** CPU 26.65 ms (37.5 FPS) | **iGPU 5.66 ms (176.7 FPS)** | **NPU 10.82 ms (92.4 FPS)**
  - **Water Segmentation:** CPU 41.18 ms (24.3 FPS) | **iGPU 8.09 ms (123.6 FPS)** | **NPU 14.80 ms (67.6 FPS)**
  - **Road Crack Detector:** CPU 26.17 ms (38.2 FPS) | **iGPU 5.31 ms (188.3 FPS)** | **NPU 8.14 ms (122.9 FPS)**
  - **Full Combined Pipeline:** CPU 133.9 ms (7.5 FPS) | **iGPU 33.93 ms (29.5 FPS — Real-Time 30 FPS)** | **NPU 52.03 ms (19.2 FPS)**
- Complete benchmark report: [`ml/road_condition_reports/model_comparison_v2.md`](ml/road_condition_reports/model_comparison_v2.md)

---

## 🧪 Comprehensive Verification & Test Suites

Run the full end-to-end verification and regression suite:
```bash
# 1. Road Condition Accuracy Overhaul Suite (13 automated tests)
pytest tests/test_road_condition_accuracy_v2.py -v

# 2. Road Condition Multimodal Perception & Fusion Tests
pytest test_road_condition_ai.py -v

# 3. Protected Pothole Subsystem Integration Tests
pytest test_pothole_detector_integration.py -v

# 4. Traffic AI Conflict & Surrogate Safety Tests
pytest test_traffic_ai_v2.py -v
pytest test_traffic_conflict_evidence.py -v

# 5. AI Vision Layer Verification
pytest test_ai_layer_v2.py -v

# 6. Complete 12-Stage Municipal SafeCity Closed-Loop Lifecycle
python test_complete_lifecycle.py

# 7. Citizen Account + Complaint Management + Admin Portal Suite (10 security & lifecycle tests)
python -m pytest tests/test_citizen_complaint_admin_portal.py -v
```

---

## 🏛 Citizen Account + Complaint Management + Municipal Admin Portal

SafeCity Loop V2 integrates an authoritative, database-backed Citizen Account and Municipal Complaint Management system into the urban road-safety platform. All state and transitions are authoritative in SQLite (`safecity.db`) via SQLAlchemy ORM with JWT authentication, bcrypt password hashing, and strict role-based access control (RBAC).

### 👥 Pre-Configured Demo Accounts

For rapid evaluation and interactive testing, SafeCity Loop V2 seeds 5 role-specific accounts accessible via **1-click quick-fill buttons on `/login`**:

| Role | Name | Email | Password | Primary Capabilities |
| :--- | :--- | :--- | :--- | :--- |
| **`CITIZEN`** | Priya Sharma | `citizen_demo@safecity.gov` | `CitizenPass123!` | File complaints, interactive map picker, upload evidence, track lifecycle, confirm resolution or reopen, manage profile. |
| **`INSPECTOR`** | Marcus Chen | `inspector_demo@safecity.gov` | `InspectorPass123!` | Review complaints, verify/reject submissions, inspect evidence, link AI-detected hazards, publish municipal updates. |
| **`ENGINEER`** | Elena Rostova | `engineer_demo@safecity.gov` | `EngineerPass123!` | Progress repair work orders, link complaints to work orders, log engineering actions, update technical resolution summary. |
| **`ADMIN`** | David Vance | `admin_demo@safecity.gov` | `AdminPass123!` | Full complaints oversight, staff assignment, duplicate detection, CSV export, user role management, system audit trail. |
| **`SUPERUSER`**| Comm. Sarah Rao | `superuser_demo@safecity.gov` | `SuperPass123!` | Unrestricted administrative access, emergency overrides, user lifecycle management with last-superuser safety protection. |

### 🔐 Role-Based Access Control (RBAC) Matrix

| Feature / Action | `CITIZEN` | `INSPECTOR` | `ENGINEER` | `ADMIN` | `SUPERUSER` |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Register Account / Log In | ✅ | ✅ | ✅ | ✅ | ✅ |
| Manage Own Citizen Profile & Notifications | ✅ | ✅ | ✅ | ✅ | ✅ |
| Submit New Complaint & Upload Evidence | ✅ | ❌ | ❌ | ✅ | ✅ |
| View Own Complaints & Audit Timeline | ✅ | ✅* | ✅* | ✅ | ✅ |
| View All Municipal Complaints & Filters | ❌ | ✅ | ✅ | ✅ | ✅ |
| State Transitions (`UNDER_REVIEW`, `VERIFIED`, etc.) | ❌ | ✅ | ✅ | ✅ | ✅ |
| Confirm Resolution / Reopen Own Complaint | ✅ | ❌ | ❌ | ✅ | ✅ |
| Post Official Public Response to Citizen | ❌ | ✅ | ✅ | ✅ | ✅ |
| Add Confidential Staff-Only Internal Notes | ❌ | ✅ | ✅ | ✅ | ✅ |
| Link Complaint to AI Hazard / Create Hazard | ❌ | ✅ | ✅ | ✅ | ✅ |
| Link Complaint to Municipal Work Order / Repair | ❌ | ❌ | ✅ | ✅ | ✅ |
| Assign Complaint to Staff Officer | ❌ | ❌ | ❌ | ✅ | ✅ |
| Export Complaints Dataset (CSV) | ❌ | ❌ | ❌ | ✅ | ✅ |
| User Management & Role Alterations | ❌ | ❌ | ❌ | ✅ | ✅ |
| Last-Superuser Protection Guard | Enforced | Enforced | Enforced | Enforced | Enforced |

*\*Staff can view their own filed or assigned complaints.*

### 🔄 Complaint State Machine Workflow

Every complaint moves through a strictly enforced state transition graph with transactional history logging, in-app citizen notifications, and tamper-evident audit ledger entries:

```
[SUBMITTED]
    │
    ├──► [ACKNOWLEDGED]
    │        │
    │        └──► [UNDER_REVIEW]
    │                 │
    │                 ├──► [VERIFIED] ──────► [ASSIGNED] ──► [IN_PROGRESS] ──► [RESOLVED]
    │                 │                                                             │
    │                 ├──► [REJECTED]                                               ├──► [CITIZEN_CONFIRMATION] ──► [CLOSED]
    │                 │                                                             │
    │                 └──► [DUPLICATE]                                              └──► [REOPENED] ──► [UNDER_REVIEW]
```

- **Public Reference Number:** Every complaint is assigned a human-readable identifier formatted as `SC-{YYYY}-{SEQUENCE:06d}` (e.g. `SC-2026-000184`).
- **Data Privacy & Officer Protection:** Citizens can only view their own complaints, evidence, and public municipal responses. Confidential internal notes and officer identification codes are strictly scrubbed in citizen-facing schemas.
- **AI Closed-Loop Linkage:** Municipal staff can correlate citizen complaints against AI vision hazard detections within 100 meters, automatically linking reports to the road risk scoring and municipal repair priority engines.

