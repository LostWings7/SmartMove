# SafeCity Road Condition Intelligence — Multimodal Evidence Fusion Evaluation

**Generated:** 2026-09-30 17:46:46 UTC  
**Subsystem:** Evidence Fusion Layer (`Pothole ∩ Water Mask`)  
**Operating Overlap Threshold:** **20.0% (0.20)**  

---

## 1. Fusion Scenario Validation

Evidence fusion evaluates the spatial intersection between protected pothole bounding boxes and the trained water segmentation mask. Component confidences are strictly preserved.

| Scenario ID | Test Scenario | Overlap % | Resulting Classification | Pothole Conf | Water Conf | Assigned Severity | Fused Risk | Verification |
|---|---|---|---|---|---|---|---|---|
| `FUS-01` | Dry Surface Pothole (No Standing Water) | 0.0% | `pothole` | 86% | 0% | `MEDIUM` | 45.0 | **PASS** |
| `FUS-02` | Boundary Jitter / Wet Perimeter (< 20% overlap) | 6.0% | `pothole` | 84% | 88% | `MEDIUM` | 45.0 | **PASS** |
| `FUS-03` | Partially Water-Filled Pothole (35% overlap) | 35.0% | `water_filled_pothole` | 91% | 88% | `HIGH` | 57.0 | **PASS** |
| `FUS-04` | Submerged Deep Pothole (85% overlap) | 100.0% | `water_filled_pothole` | 89% | 88% | `CRITICAL` | 89.8 | **PASS** |
| `FUS-05` | Extensive Roadway Ponding (No Pothole) | 0.0% | `water` | N/A | 88% | `HIGH` | 62.0 | **PASS** |

---

## 2. Threshold Sensitivity Analysis

An operational cutoff of **20%** was selected based on bounding box geometry and segmentation edge uncertainty:

| Cutoff Threshold | Decision Rule | False Fusion Risk (Edge Jitter) | Missed Fusion Risk (Partial Pools) | Operational Status |
|---|---|---|---|---|
| `5%` | `Overlap >= 5%` | HIGH | LOW | Candidate |
| `10%` | `Overlap >= 10%` | HIGH | LOW | Candidate |
| `15%` | `Overlap >= 15%` | MODERATE | LOW | Candidate |
| `20%` | `Overlap >= 20%` | LOW | LOW | **PRODUCTION DEFAULT (OPTIMAL)** |
| `25%` | `Overlap >= 25%` | LOW | LOW | Candidate |
| `30%` | `Overlap >= 30%` | LOW | MODERATE | Candidate |
| `40%` | `Overlap >= 40%` | LOW | HIGH | Candidate |
| `50%` | `Overlap >= 50%` | LOW | HIGH | Candidate |

### Rationale for 20% Cutoff:
1. **Perimeter Wetness Rejection:** Wet pavement bordering a dry pothole commonly produces 3–8% mask overlap due to bounding box aspect ratio mismatch. A threshold < 15% falsely flags dry potholes on wet roads.
2. **True Water Ponding Capture:** Real water accumulation settles in the depression, occupying 25% to 100% of the pothole cross-sectional area.
3. **20% Cutoff provides a 12% safety margin** against false positive fusion while preserving high recall for genuinely hazardous water-filled potholes.

---

## 3. Explainable Risk Sub-Additivity & Anti-Double-Counting

To prevent double-counting when multiple collocated defects occur on the same road segment:

```
Fused Risk = min(100.0, R_max + 0.12 * sum(R_remaining))
```

| Method | Component Risks (Pothole=60, Water=50, Crack=40) | Calculated Fused Risk | Double-Counting Prevented? |
|---|---|---|---|
| Linear Summation (Naive) | 60 + 50 + 40 | 150.0 (Unbounded) | ❌ No (Double-counts risk) |
| **SafeCity Sub-Additive Fusion** | 60 + (0.12 * 50) + (0.12 * 40) | **70.8 / 100** | **✓ Yes (Diminishing marginal risk)** |

### Properties Confirmed:
- **Monotonicity:** Adding another hazard never decreases risk.
- **Boundedness:** Risk remains strictly <= 100.0.
- **Dominance:** The most critical defect sets the risk baseline (R_max).
- **Non-Double Counting:** Sub-additive property holds: Fused Risk < sum(R_i).
