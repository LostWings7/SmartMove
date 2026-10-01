# SafeCity Loop V2 — Traffic AI False Conflict Taxonomy & Failure Analysis

This document provides a systematic engineering breakdown of false conflict scenarios, optical failure modes, tracking edge cases, and the algorithmic mitigations implemented in the SafeCity Video Surrogate Safety Pipeline.

---

## 1. Taxonomy of False Conflict Scenarios

### Category A: Sensor & Environmental Artifacts
1. **Low-Light / Night Headlight Glare:**
   - *Failure Mode:* Headlight glare on wet asphalt surfaces or low-light sensor blooming expands visual bounding boxes, temporarily distorting the calculated bottom-center ground contact point.
   - *Consequence:* An artificial displacement of 1.5–3.0 meters occurs in a single frame, resulting in an instantaneous velocity spike that falsely triggers TTC convergence.
   - *Mitigation:* Bounding-box temporal smoothing with sliding-window median filtering and velocity delta clipping ($|\Delta v| < 15\text{ m/s}^2$).

2. **Rain & Surface Reflections:**
   - *Failure Mode:* In heavy rain, reflected silhouettes of vehicles or pedestrians appear on wet road surfaces.
   - *Consequence:* The detector may detect inverted duplicate instances or jitter bottom-edge coordinates.
   - *Mitigation:* Floor clamping in Inverse Perspective Mapping (IPM) ensuring longitudinal ground coordinates fall within legitimate road bounds ($1.5\text{m} \le Y \le 80\text{m}$).

---

### Category B: Multi-Object Tracking (MOT) Artifacts
3. **Track ID Switches (IDSW):**
   - *Failure Mode:* When two vehicles cross or temporarily occlude each other, the tracker may swap their persistent track IDs.
   - *Consequence:* The trajectory history abruptly jumps between two distinct spatial coordinates. When differentiated over $\Delta t$, this produces an erroneous high velocity vector projecting a near-instantaneous collision.
   - *Mitigation:* Kalman state continuity checks; if a displacement between successive frames exceeds $40\text{ px}$ (or $4.5\text{m}$ ground distance), the velocity estimator discards the transition and re-initializes kinematic state.

4. **Track Fragmentation (Short-Lived Tracks):**
   - *Failure Mode:* Transient detections lasting 1–2 frames (false positives on roadside mailboxes or lampposts).
   - *Consequence:* High variance in single-frame coordinate derivatives.
   - *Mitigation:* Minimum Track Longevity threshold: only tracks with $\ge 3$ consecutive frames are eligible for kinematic trajectory derivation and surrogate safety evaluation.

---

### Category C: Geometric & Perspective Ambiguities
5. **Parallel Lane Following vs. Merging:**
   - *Failure Mode:* Vehicles driving side-by-side in adjacent highway lanes have small lateral distances ($< 2.5\text{m}$) and similar speeds.
   - *Consequence:* A naive distance threshold flags this as a critical near-miss despite zero collision vector convergence.
   - *Mitigation:* 2D Closest Point of Approach (CPA) kinematics: a conflict is **only** registered if trajectories mathematically converge ($\vec{p}_{\text{rel}} \cdot \vec{v}_{\text{rel}} < 0$). Parallel trajectories yield $\vec{p}_{\text{rel}} \cdot \vec{v}_{\text{rel}} \approx 0$ and are classified as *"Lateral Merge / Parallel Movement"*, avoiding false alerts.

6. **Parked & Stationary Roadside Objects:**
   - *Failure Mode:* Parked cars along curbs have zero velocity while passing traffic moves past them at close lateral clearance.
   - *Consequence:* Proximity checks alone flag every parked car passed by an active vehicle.
   - *Mitigation:* Direction & Velocity thresholding: closing speed must exceed $1.0\text{ m/s}$ ($3.6\text{ km/h}$) and projected clearance at closest approach ($d_{\text{CPA}}$) must violate safety buffers.

---

## 2. Surrogate Conflict Verification Pipeline

To ensure municipal credibility and prevent false alarms from overwhelming traffic engineers:

```
Raw YOLO Detections
       ↓
ByteTrack Association (Persistent IDs)
       ↓
Longevity Filter (Tracks ≥ 3 Frames)
       ↓
Bottom-Center Contact Projection: (x1+x2)/2, y2
       ↓
Inverse Perspective Mapping (IPM) to Real-World Ground Plane (Meters)
       ↓
Sliding-Window Kinematic Smoothing (Velocity & Acceleration)
       ↓
2D Closest Point of Approach (CPA) Convergence Check: p_rel · v_rel < -0.05
       ↓
Multi-Criteria Evaluation: TTC < Threshold AND DRAC > Threshold
       ↓
Municipal Human-in-the-Loop Review Queue ('reviewed', 'actioned', 'false_positive')
```

---

## 3. Municipal Engineering Audit Protocol

1. **False Positive Labeling:** Traffic engineers can mark spurious alerts as `false_positive` directly in the Municipal Conflict Review Modal.
2. **Chain of Custody:** Every conflict candidate records:
   - Synchronized keyframe snapshot
   - Interaction geometry classification
   - 2D CPA TTC, true PET, and required deceleration (DRAC)
   - Plain-English causal justification
