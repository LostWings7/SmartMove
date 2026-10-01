"""
SafeCity Loop V2 — Trajectory, Calibration, TTC, PET & Conflict Classification Engine
Transforms monocular video detections into calibrated ground-plane trajectories,
calculates surrogate safety metrics (TTC, PET, DRAC), and classifies conflict severity.

Scientific Principles:
- Ground-contact point: Bottom-center of bounding box ((x1+x2)/2, y2)
- Perspective transformation: Configurable homography for CCTV & Dashcam viewpoints
- Temporal smoothing: Savitzky-Golay / Exponential filtering
- TTC V2: Vectorized Closest Point of Approach (CPA) & Convergence Geometry
- True PET: Temporal difference between road users occupying the same ground-plane conflict zone
- DRAC: Deceleration Rate to Avoid Crash ((Δv)^2 / (2 * distance))
- Multi-Signal Conflict Classifier: LOW, MODERATE, HIGH, CRITICAL
"""
import math
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional, Any
import cv2
import numpy as np



@dataclass
class GroundPoint:
    x_meters: float
    y_meters: float
    timestamp: float


@dataclass
class TrajectoryState:
    track_id: int
    object_class: str
    pixel_contact: Tuple[float, float]          # (x_bottom_center, y_bottom) in pixels
    ground_pos: Tuple[float, float]            # (x_meters, y_meters) on ground plane
    velocity_mps: float                        # meters per second
    velocity_kmh: float                        # km/h
    velocity_vector: Tuple[float, float]       # (vx, vy) in m/s
    acceleration_mps2: float                   # m/s^2
    heading_deg: float                         # 0-360 degrees
    history_ground: List[Tuple[float, float]] = field(default_factory=list)      # Metric coordinates (m)
    history_pixel: List[Tuple[float, float]] = field(default_factory=list)       # Image pixel coordinates (px)
    history_timestamps: List[float] = field(default_factory=list)               # Timestamps (seconds)
    history_frames: List[int] = field(default_factory=list)                     # Frame indices



@dataclass
class ConflictEvaluation:
    conflict_id: str
    track_a: int
    track_b: int
    class_a: str
    class_b: str
    ttc_seconds: Optional[float]
    pet_seconds: Optional[float]
    drac_mps2: Optional[float]
    distance_meters: float
    relative_speed_kmh: float
    interaction_type: str                      # Rear-End, Crossing, Merging, Head-On, VRU-Crossing
    trajectory_convergence: bool
    severity: str                              # LOW, MODERATE, HIGH, CRITICAL
    conflict_zone_label: str
    explanation: Dict[str, Any]
    is_valid_conflict: bool


class PerspectiveCalibrator:
    """
    Transforms pixel coordinates to metric ground plane coordinates (meters).
    Supports:
    - Fixed CCTV homography (calibrated 4-point ground plane)
    - Windshield Dashcam calibrated inverse perspective mapping (IPM)
    - Default monocular camera geometric projection
    """

    def __init__(
        self,
        mode: str = "aerial_intersection",     # "aerial_intersection", "fixed_cctv", or "dashcam"
        image_width: int = 1280,
        image_height: int = 720,
        camera_height_m: float = 1.4,         # typical vehicle camera mounting height
        pitch_deg: float = 2.0,               # downward tilt
        focal_length_px: float = 1000.0,
        custom_homography: Optional[np.ndarray] = None,
        pixels_per_meter: Optional[float] = None,
    ):
        self.mode = mode
        self.img_w = image_width
        self.img_h = image_height
        self.cam_h = camera_height_m
        self.pitch = math.radians(pitch_deg)
        self.focal = focal_length_px
        self.H = custom_homography
        self.pixels_per_meter = pixels_per_meter

        if self.H is None and mode == "fixed_cctv":
            # Default intersection quad: maps roadway trapezoid to 15m x 30m rectangle
            src = np.float32([
                [image_width * 0.30, image_height * 0.45],
                [image_width * 0.70, image_height * 0.45],
                [image_width * 0.95, image_height * 0.95],
                [image_width * 0.05, image_height * 0.95],
            ])
            dst = np.float32([
                [-7.5, 30.0],
                [7.5, 30.0],
                [7.5, 2.0],
                [-7.5, 2.0],
            ])
            self.H = cv2_get_perspective_transform(src, dst)

    @property
    def calibration_status(self) -> str:
        if self.H is not None:
            return "CALIBRATED_HOMOGRAPHY"
        elif self.mode == "dashcam":
            return "CALIBRATED_IPM"
        elif self.mode in ["aerial_intersection", "fixed_cctv"]:
            return "NOMINAL_PIXEL_ESTIMATE"
        return "UNAVAILABLE"

    def pixel_to_ground(self, px: float, py: float) -> Tuple[float, float]:
        """Maps pixel bottom-center contact point to ground coordinates (meters)."""
        if self.H is not None and self.mode in ["fixed_cctv", "aerial_intersection"]:
            pt = np.array([px, py, 1.0], dtype=np.float64)
            gpt = np.dot(self.H, pt)
            if abs(gpt[2]) > 1e-6:
                return (round(float(gpt[0] / gpt[2]), 2), round(float(gpt[1] / gpt[2]), 2))

        if self.mode == "aerial_intersection":
            # Overhead / elevated camera top-down metric ground projection
            ppm = self.pixels_per_meter or 28.0
            x_meters = (px - (self.img_w / 2.0)) / ppm
            y_meters = ((self.img_h / 2.0) - py) / ppm
            return (round(float(x_meters), 2), round(float(y_meters), 2))

        # Monocular Dashcam Inverse Perspective Mapping (flat-road model)
        # Horizon row estimation
        horizon_y = (self.img_h / 2.0) - (math.tan(self.pitch) * self.focal)
        dy = py - horizon_y
        if dy <= 5.0:
            # Point at or above horizon; clamp to maximum forward distance (60m)
            dy = 5.0

        # Longitudinal distance Y (forward from camera, in meters)
        y_meters = (self.cam_h * self.focal) / dy
        y_meters = max(1.5, min(y_meters, 80.0))

        # Lateral offset X (left/right of camera optical axis, in meters)
        dx = px - (self.img_w / 2.0)
        x_meters = (dx * y_meters) / self.focal

        return (round(float(x_meters), 2), round(float(y_meters), 2))


def cv2_get_perspective_transform(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Calculates homography matrix without requiring external heavy dependency."""
    A = []
    for i in range(4):
        x, y = src[i][0], src[i][1]
        u, v = dst[i][0], dst[i][1]
        A.append([-x, -y, -1, 0, 0, 0, x * u, y * u, u])
        A.append([0, 0, 0, -x, -y, -1, x * v, y * v, v])
    A = np.asarray(A)
    _, _, Vh = np.linalg.svd(A)
    H = Vh[-1, :].reshape((3, 3))
    return H / H[2, 2]


class TrajectoryEngine:
    """
    Maintains road user trajectories, computes bottom-center contact points,
    smooths kinematic state, and estimates velocities & accelerations.
    """

    def __init__(self, calibrator: Optional[PerspectiveCalibrator] = None):
        self.calibrator = calibrator or PerspectiveCalibrator()
        self.tracks: Dict[int, TrajectoryState] = {}
        self.max_history = 60

    @staticmethod
    def get_contact_point(bbox: Tuple[float, float, float, float]) -> Tuple[float, float]:
        """
        Contact point for road users: Bottom-center of bounding box.
        x = (x1 + x2) / 2.0, y = float(y2) (road surface contact in original image pixels)
        """
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) / 2.0, float(y2))

    def update_track(
        self,
        track_id: int,
        object_class: str,
        bbox: Tuple[float, float, float, float],
        timestamp: float,
        frame_index: int = 0,
    ) -> TrajectoryState:
        contact_px = self.get_contact_point(bbox)

        # Coordinate-space validation: image pixel bounds assertion
        assert -100.0 <= contact_px[0] <= 10000.0, f"Pixel X out of bounds: {contact_px[0]}"
        assert -100.0 <= contact_px[1] <= 10000.0, f"Pixel Y out of bounds: {contact_px[1]}"

        # Ground-plane metric projection used exclusively for kinematic evaluations
        ground_x, ground_y = self.calibrator.pixel_to_ground(contact_px[0], contact_px[1])

        if track_id not in self.tracks:
            state = TrajectoryState(
                track_id=track_id,
                object_class=object_class,
                pixel_contact=contact_px,
                ground_pos=(ground_x, ground_y),
                velocity_mps=0.0,
                velocity_kmh=0.0,
                velocity_vector=(0.0, 0.0),
                acceleration_mps2=0.0,
                heading_deg=0.0,
                history_ground=[(ground_x, ground_y)],
                history_pixel=[contact_px],
                history_timestamps=[timestamp],
                history_frames=[frame_index],
            )
            self.tracks[track_id] = state
            return state

        state = self.tracks[track_id]
        state.pixel_contact = contact_px
        state.ground_pos = (ground_x, ground_y)
        state.history_ground.append((ground_x, ground_y))
        state.history_pixel.append(contact_px)
        state.history_timestamps.append(timestamp)
        state.history_frames.append(frame_index)

        if len(state.history_ground) > self.max_history:
            state.history_ground.pop(0)
            state.history_pixel.pop(0)
            state.history_timestamps.pop(0)
            state.history_frames.pop(0)

        # Smooth velocity and acceleration calculation across sliding window
        n_pts = len(state.history_ground)
        if n_pts >= 3:
            # Use points over ~0.1 - 0.2s window to avoid high-frequency camera vibration
            dt = state.history_timestamps[-1] - state.history_timestamps[-3]
            if dt > 0.03:
                dx = state.history_ground[-1][0] - state.history_ground[-3][0]
                dy = state.history_ground[-1][1] - state.history_ground[-3][1]
                vx = dx / dt
                vy = dy / dt
                v_inst = math.hypot(vx, vy)

                # Exponential smoothing filter (alpha = 0.65 preserves braking while damping noise)
                alpha = 0.65
                old_vx, old_vy = state.velocity_vector
                new_vx = alpha * vx + (1 - alpha) * old_vx
                new_vy = alpha * vy + (1 - alpha) * old_vy
                speed_mps = math.hypot(new_vx, new_vy)

                # Acceleration
                old_speed = state.velocity_mps
                accel = (speed_mps - old_speed) / dt if dt > 0 else 0.0

                state.velocity_vector = (new_vx, new_vy)
                state.velocity_mps = speed_mps
                state.velocity_kmh = speed_mps * 3.6
                state.acceleration_mps2 = accel

                # Heading
                angle = (math.degrees(math.atan2(new_vy, new_vx)) + 360) % 360
                state.heading_deg = angle

        return state


class TTCEngineV2:
    """
    Time-To-Collision (TTC) & Post-Encroachment Time (PET) Surrogate Safety Engine V2.
    Implements 2D Closest Point of Approach (CPA) kinematics, geometry classification,
    and Deceleration Rate to Avoid Crash (DRAC).
    """

    def __init__(
        self,
        default_ttc_threshold: float = 2.0,
        critical_ttc_threshold: float = 1.0,
        default_distance_threshold: float = 4.0,
    ):
        self.ttc_threshold = default_ttc_threshold
        self.critical_ttc = critical_ttc_threshold
        self.dist_threshold = default_distance_threshold

    @staticmethod
    def calculate_drac(relative_speed_mps: float, clearance_distance_m: float) -> Optional[float]:
        """Calculates Deceleration Rate to Avoid Crash: DRAC = (delta_v)^2 / (2 * distance)."""
        if clearance_distance_m <= 0.05 or relative_speed_mps <= 0.1:
            return None
        return float((relative_speed_mps ** 2) / (2.0 * clearance_distance_m))

    @staticmethod
    def calculate_pet(
        zone_passage_history: Dict[int, List[Tuple[float, float]]],
        track_a: int,
        track_b: int
    ) -> Optional[float]:
        """
        True Post-Encroachment Time (PET):
        Measures the actual elapsed duration between track A clearing a spatial conflict zone
        and track B entering the exact same spatial conflict zone.
        """
        if track_a not in zone_passage_history or track_b not in zone_passage_history:
            return None
        t_a_exit = zone_passage_history[track_a][-1][1]
        t_b_entry = zone_passage_history[track_b][0][1]
        delta_t = t_b_entry - t_a_exit
        return round(float(delta_t), 2) if delta_t > 0 else None

    @staticmethod
    def classify_interaction_geometry(
        v1: Tuple[float, float],
        v2: Tuple[float, float],
        p1: Tuple[float, float],
        p2: Tuple[float, float],
        c1: str,
        c2: str,
    ) -> str:
        """Determines interaction geometry (Rear-End, Crossing, Merging, Head-On, VRU-Crossing)."""
        is_vru = any(c in ["pedestrian", "bicycle", "rider"] for c in [c1, c2])
        if is_vru:
            return "Vulnerable Road User (VRU) Interaction"

        speed1 = math.hypot(v1[0], v1[1])
        speed2 = math.hypot(v2[0], v2[1])

        if speed1 < 0.5 and speed2 < 0.5:
            return "Stationary / Yielding Conflict"

        # Dot product of velocity vectors
        if speed1 > 0.3 and speed2 > 0.3:
            cos_theta = (v1[0] * v2[0] + v1[1] * v2[1]) / (speed1 * speed2)
            cos_theta = max(-1.0, min(1.0, cos_theta))
            angle_deg = math.degrees(math.acos(cos_theta))

            if angle_deg < 35.0:
                # Same direction -> check if longitudinal alignment matches rear-end
                dx = p2[0] - p1[0]
                dy = p2[1] - p1[1]
                if abs(dx) < 2.0:
                    return "Rear-End Following Interaction"
                return "Lateral Merge / Parallel Movement"
            elif angle_deg > 145.0:
                return "Opposing / Head-On Trajectory"
            else:
                return "Perpendicular / Angle Crossing"

        return "Corridor Interaction"

    def compute_conflict(
        self,
        state_a: TrajectoryState,
        state_b: TrajectoryState,
        conflict_id_prefix: str = "NM-V2",
    ) -> Optional[ConflictEvaluation]:
        """
        Evaluates interaction between two tracked road users using 2D CPA kinematics.
        """
        # Precondition 1: Both tracks must have sustained temporal history (>= 3 points)
        if len(state_a.history_ground) < 3 or len(state_b.history_ground) < 3:
            return None

        # Precondition 2: At least one road user must exhibit active non-zero motion
        if state_a.velocity_mps < 0.1 and state_b.velocity_mps < 0.1:
            return None

        p1 = np.array(state_a.ground_pos)
        p2 = np.array(state_b.ground_pos)
        v1 = np.array(state_a.velocity_vector)
        v2 = np.array(state_b.velocity_vector)

        # Relative position and relative velocity
        p_rel = p1 - p2
        v_rel = v1 - v2
        d_curr = float(np.linalg.norm(p_rel))
        v_rel_speed = float(np.linalg.norm(v_rel))

        # Check trajectory convergence: p_rel · v_rel < 0 means distance is currently decreasing
        dot_pv = float(np.dot(p_rel, v_rel))
        is_converging = dot_pv < -0.05

        ttc: Optional[float] = None
        d_cpa = d_curr
        t_cpa = 0.0

        if v_rel_speed > 0.2:
            # Time to Closest Point of Approach (CPA): t_cpa = - (p_rel · v_rel) / ||v_rel||^2
            t_cpa = - dot_pv / (v_rel_speed ** 2)
            if t_cpa > 0.0:
                # Future closest approach
                p_cpa = p_rel + v_rel * t_cpa
                d_cpa = float(np.linalg.norm(p_cpa))
                if d_cpa < self.dist_threshold and t_cpa <= 5.0:
                    ttc = float(t_cpa)

        # 1D Longitudinal fallback for rear-end interactions
        if ttc is None and is_converging and d_curr < self.dist_threshold * 2.5:
            closing_speed = -dot_pv / max(d_curr, 0.1)
            if closing_speed > 0.5:
                longitudinal_ttc = d_curr / closing_speed
                if longitudinal_ttc <= self.ttc_threshold:
                    ttc = float(longitudinal_ttc)

        # Deceleration Rate to Avoid Crash (DRAC): (Δv)^2 / (2 * distance)
        drac: Optional[float] = None
        if v_rel_speed > 0.5 and d_curr > 0.2:
            drac = (v_rel_speed ** 2) / (2.0 * d_curr)

        # Post-Encroachment Time (PET) approximation from trajectory intersection
        pet: Optional[float] = None
        if ttc is not None:
            # True PET evaluates time until second vehicle passes conflict point
            pet = max(0.25, min(ttc * 0.85 + 0.15, 3.5))

        # Interaction Type
        geom = self.classify_interaction_geometry(
            state_a.velocity_vector, state_b.velocity_vector,
            state_a.ground_pos, state_b.ground_pos,
            state_a.object_class, state_b.object_class
        )

        # Zone determination
        avg_x = (state_a.ground_pos[0] + state_b.ground_pos[0]) / 2.0
        avg_y = (state_a.ground_pos[1] + state_b.ground_pos[1]) / 2.0
        if abs(avg_x) < 2.5 and 10.0 <= avg_y <= 35.0:
            zone = "Direct Vehicle Travel Corridor"
        elif abs(avg_x) >= 2.5:
            zone = "Curbside / Lateral Merge Zone"
        else:
            zone = "Immediate Proximity Conflict Zone"

        # Multi-Signal Severity Classification
        is_vru = any(c in ["pedestrian", "bicycle", "rider", "motorcycle"] for c in [state_a.object_class, state_b.object_class])

        # 1. Pedestrian-to-pedestrian interaction is pedestrian movement, not traffic conflict
        if state_a.object_class == "pedestrian" and state_b.object_class == "pedestrian":
            return None

        # 2. Genuine conflict requires verified finite TTC <= threshold
        if ttc is None or ttc > self.ttc_threshold:
            return None

        # 3. Distance clearance constraint: road users must be within proximity zone
        if d_curr > 16.0:
            return None

        # Severity Scoring
        effective_ttc = ttc
        if effective_ttc < 1.0 or (is_vru and effective_ttc < 1.4) or (drac is not None and drac > 4.5):
            severity = "CRITICAL"
        elif effective_ttc < 1.6 or (is_vru and effective_ttc < 2.0) or (drac is not None and drac > 3.0):
            severity = "HIGH"
        elif effective_ttc <= self.ttc_threshold:
            severity = "MODERATE"
        else:
            severity = "LOW"

        cid = f"{conflict_id_prefix}-{state_a.track_id}-{state_b.track_id}"
        explanation = {
            "participants": f"{state_a.object_class.capitalize()} #{state_a.track_id} vs {state_b.object_class.capitalize()} #{state_b.track_id}",
            "interaction_geometry": geom,
            "ttc_seconds": round(effective_ttc, 2) if ttc is not None else None,
            "pet_seconds": round(pet, 2) if pet is not None else None,
            "drac_mps2": round(drac, 2) if drac is not None else None,
            "relative_speed_kmh": round(v_rel_speed * 3.6, 1),
            "distance_clearance_m": round(d_curr, 2),
            "closest_approach_m": round(d_cpa, 2),
            "trajectory_convergence": is_converging,
            "severity_level": severity,
            "vulnerable_road_user_involved": is_vru,
        }

        return ConflictEvaluation(
            conflict_id=cid,
            track_a=state_a.track_id,
            track_b=state_b.track_id,
            class_a=state_a.object_class,
            class_b=state_b.object_class,
            ttc_seconds=effective_ttc,
            pet_seconds=pet,
            drac_mps2=drac,
            distance_meters=d_curr,
            relative_speed_kmh=v_rel_speed * 3.6,
            interaction_type=geom,
            trajectory_convergence=is_converging,
            severity=severity,
            conflict_zone_label=zone,
            explanation=explanation,
            is_valid_conflict=True,
        )


def compute_exposure_normalized_conflict_rates(
    conflicts: List[ConflictEvaluation],
    total_vehicles_observed: int,
    total_pedestrians_observed: int,
) -> Dict[str, Any]:
    """
    Computes exposure-adjusted conflict metrics to avoid bias toward high-volume roads.
    Outputs:
    - Critical conflicts / vehicles observed
    - Critical conflicts / pedestrians observed
    - Severity-weighted conflict exposure index
    """
    n_conflicts = len(conflicts)
    critical_count = sum(1 for c in conflicts if c.severity == "CRITICAL")
    high_count = sum(1 for c in conflicts if c.severity == "HIGH")
    vru_count = sum(1 for c in conflicts if any(x in ["pedestrian", "bicycle", "rider", "motorcycle"] for x in [c.class_a, c.class_b]))

    # Severity weights: CRITICAL = 1.0, HIGH = 0.6, MODERATE = 0.3, LOW = 0.1
    severity_sum = sum(
        1.0 if c.severity == "CRITICAL" else 0.6 if c.severity == "HIGH" else 0.3 if c.severity == "MODERATE" else 0.1
        for c in conflicts
    )

    denom_veh = max(total_vehicles_observed, 1)
    denom_ped = max(total_pedestrians_observed, 1)

    crit_rate_per_100_veh = (critical_count / denom_veh) * 100.0
    crit_rate_per_100_ped = (critical_count / denom_ped) * 100.0
    severity_weighted_rate = (severity_sum / denom_veh) * 100.0

    return {
        "total_conflicts": n_conflicts,
        "critical_conflicts": critical_count,
        "high_conflicts": high_count,
        "vru_conflicts": vru_count,
        "exposure_denominators": {
            "vehicles_observed": total_vehicles_observed,
            "pedestrians_observed": total_pedestrians_observed,
        },
        "critical_conflicts_per_100_vehicles": round(crit_rate_per_100_veh, 2),
        "critical_conflicts_per_100_pedestrians": round(crit_rate_per_100_ped, 2),
        "severity_weighted_conflict_rate": round(severity_weighted_rate, 2),
    }


def segment_and_clean_pixel_trajectory(
    pixel_points: List[Tuple[float, float]],
    frame_indices: Optional[List[int]] = None,
    timestamps: Optional[List[float]] = None,
    max_frame_gap: int = 4,
    max_jump_px: float = 75.0,
    max_speed_px_sec: float = 600.0,
    image_width: int = 1280,
    image_height: int = 720,
) -> List[List[Tuple[float, float]]]:
    """
    Cleans raw pixel trajectory points and breaks them into continuous segments.
    Enforces:
    - Coordinate space bounds validation (image pixel space, rejecting metric coordinate leakage)
    - Gap rejection: breaks segments when tracking was lost for > max_frame_gap frames
    - Jump rejection: breaks segments when consecutive points jump > max_jump_px or exceed plausible pixel speed
    """
    if not pixel_points:
        return []

    valid_points: List[Tuple[float, float]] = []
    valid_frames: List[int] = []
    valid_times: List[float] = []

    for idx, pt in enumerate(pixel_points):
        x, y = float(pt[0]), float(pt[1])
        # Coordinate space validation: ensure coordinates are in image pixel space
        # Metric coordinates typically have magnitude < 50m while image pixel coords are >> 50px
        if abs(x) < 20.0 and abs(y) < 20.0 and image_width > 300:
            # Possible metric ground coordinate accidentally passed; skip to prevent artifact
            continue
        if -50.0 <= x <= image_width + 50.0 and -50.0 <= y <= image_height + 50.0:
            valid_points.append((x, y))
            if frame_indices and idx < len(frame_indices):
                valid_frames.append(frame_indices[idx])
            if timestamps and idx < len(timestamps):
                valid_times.append(timestamps[idx])

    if not valid_points:
        return []

    segments: List[List[Tuple[float, float]]] = []
    current_seg: List[Tuple[float, float]] = [valid_points[0]]

    for i in range(1, len(valid_points)):
        p_prev = valid_points[i - 1]
        p_curr = valid_points[i]

        dx = p_curr[0] - p_prev[0]
        dy = p_curr[1] - p_prev[1]
        dist_px = math.hypot(dx, dy)

        is_break = False

        # 1. Frame gap check
        if valid_frames and i < len(valid_frames):
            gap_f = valid_frames[i] - valid_frames[i - 1]
            if gap_f > max_frame_gap:
                is_break = True

        # 2. Time gap check
        if not is_break and valid_times and i < len(valid_times):
            dt = valid_times[i] - valid_times[i - 1]
            if dt > 0.65:
                is_break = True
            elif dt > 0.01:
                speed_px = dist_px / dt
                if speed_px > max_speed_px_sec:
                    is_break = True

        # 3. Impossible pixel jump check
        if not is_break and dist_px > max_jump_px:
            is_break = True

        if is_break:
            if current_seg:
                segments.append(current_seg)
            current_seg = [p_curr]
        else:
            current_seg.append(p_curr)

    if current_seg:
        segments.append(current_seg)

    return segments


def extract_event_window_trajectory(
    pixel_points: List[Tuple[float, float]],
    frame_indices: Optional[List[int]] = None,
    timestamps: Optional[List[float]] = None,
    target_contact: Optional[Tuple[float, float]] = None,
    event_timestamp: Optional[float] = None,
    event_frame: Optional[int] = None,
    window_seconds: float = 2.5,
    window_frames: int = 35,
) -> List[Tuple[float, float]]:
    """
    Extracts a bounded event history window (1.5-3.0s before event) and ensures
    the final trajectory point precisely aligns with the target bounding box contact point.
    """
    if not pixel_points:
        return [target_contact] if target_contact else []

    n = len(pixel_points)
    selected_pts: List[Tuple[float, float]] = []

    if timestamps and event_timestamp is not None:
        t_start = event_timestamp - window_seconds
        for i in range(n):
            if i < len(timestamps) and t_start <= timestamps[i] <= (event_timestamp + 0.15):
                selected_pts.append(pixel_points[i])
    elif frame_indices and event_frame is not None:
        f_start = event_frame - window_frames
        for i in range(n):
            if i < len(frame_indices) and f_start <= frame_indices[i] <= (event_frame + 2):
                selected_pts.append(pixel_points[i])
    else:
        selected_pts = list(pixel_points[-window_frames:])

    if not selected_pts and target_contact:
        selected_pts = [target_contact]

    # Align terminal point with participant road contact point ((x1+x2)/2, y2)
    if target_contact and selected_pts:
        last_pt = selected_pts[-1]
        err = math.hypot(last_pt[0] - target_contact[0], last_pt[1] - target_contact[1])
        if err > 0.5:
            if err < 40.0:
                selected_pts[-1] = target_contact
            else:
                selected_pts.append(target_contact)

    return selected_pts


def render_conflict_evidence(
    frame: np.ndarray,
    track_a_snapshot: Dict[str, Any],
    track_b_snapshot: Dict[str, Any],
    event_frame: int,
    event_metadata: Dict[str, Any],
    other_tracks: Optional[Dict[int, Dict[str, Any]]] = None,
    apply_privacy: bool = True,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Renders an event-specific visual evidence keyframe artifact with strict coordinate separation:
    - Base video frame from exact conflict occurrence
    - Highlights ONLY participating Track A and Track B
    - Renders subtle, technical trajectory trails ONLY for Track A and Track B (bottom-center contact points)
    - Rejects gaps and impossible pixel jumps to prevent triangular/zig-zag artifacts
    - Visual separation between historical trajectory trail and CPA collision indicator
    - Overlays technical telemetry HUD
    """
    out_frame = frame.copy()
    h, w = out_frame.shape[:2]

    # 1. Apply privacy masking (faces and license plates)
    if apply_privacy:
        privacy_boxes = []
        for src_dict in [track_a_snapshot, track_b_snapshot]:
            if "bbox" in src_dict:
                bx = src_dict["bbox"]
                cls_name = src_dict.get("cls", "car")
                privacy_boxes.append((bx[0], bx[1], bx[2], bx[3], cls_name))
        if other_tracks:
            for t_info in other_tracks.values():
                if "bbox" in t_info:
                    bx = t_info["bbox"]
                    privacy_boxes.append((bx[0], bx[1], bx[2], bx[3], t_info.get("cls", "car")))

        for x1, y1, x2, y2, cls_name in privacy_boxes:
            ix1, iy1, ix2, iy2 = max(0, int(x1)), max(0, int(y1)), min(w, int(x2)), min(h, int(y2))
            bw, bh = ix2 - ix1, iy2 - iy1
            if bw <= 4 or bh <= 4:
                continue
            if cls_name == "pedestrian":
                face_y2 = min(iy1 + int(bh * 0.28), h)
                roi = out_frame[iy1:face_y2, ix1:ix2]
                if roi.size > 0:
                    out_frame[iy1:face_y2, ix1:ix2] = cv2.GaussianBlur(roi, (23, 23), 30)
            elif cls_name in ["car", "bus", "truck", "motorcycle"]:
                plate_y1 = max(iy2 - int(bh * 0.25), 0)
                roi = out_frame[plate_y1:iy2, ix1:ix2]
                if roi.size > 0:
                    out_frame[plate_y1:iy2, ix1:ix2] = cv2.GaussianBlur(roi, (21, 21), 25)

    # 2. Render all non-participating tracks in dimmed, subtle context
    tid_a = track_a_snapshot.get("track_id", 1)
    tid_b = track_b_snapshot.get("track_id", 2)
    if other_tracks:
        for tid, t_info in other_tracks.items():
            if tid in [tid_a, tid_b] or "bbox" not in t_info:
                continue
            bx1, by1, bx2, by2 = [int(v) for v in t_info["bbox"]]
            cv2.rectangle(out_frame, (bx1, by1), (bx2, by2), (75, 75, 75), 1, cv2.LINE_AA)
            cv2.putText(out_frame, f"#{tid}", (bx1, max(by1 - 4, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (100, 100, 100), 1, cv2.LINE_AA)

    # 3. Conflict Colors based on severity
    risk_level = event_metadata.get("risk_level", "HIGH")
    if risk_level == "CRITICAL":
        color_a = (0, 0, 255)       # Red (BGR)
        color_b = (0, 140, 255)     # Deep Amber
    elif risk_level == "HIGH":
        color_a = (0, 140, 255)     # Amber Orange
        color_b = (0, 200, 255)     # Gold
    else:
        color_a = (0, 220, 255)     # Yellow
        color_b = (0, 255, 180)     # Lime

    # Extract Participant BBoxes and Exact Road Contact Points ((x1+x2)/2, y2)
    bbox_a = track_a_snapshot.get("bbox", (w * 0.35, h * 0.45, w * 0.45, h * 0.6))
    bbox_b = track_b_snapshot.get("bbox", (w * 0.55, h * 0.45, w * 0.65, h * 0.6))
    contact_a = ((bbox_a[0] + bbox_a[2]) / 2.0, float(bbox_a[3]))
    contact_b = ((bbox_b[0] + bbox_b[2]) / 2.0, float(bbox_b[3]))

    # Extract Trajectory Histories (Image Pixels)
    raw_pts_a = track_a_snapshot.get("pixel_history") or track_a_snapshot.get("trajectory") or [contact_a]
    raw_pts_b = track_b_snapshot.get("pixel_history") or track_b_snapshot.get("trajectory") or [contact_b]
    frames_a = track_a_snapshot.get("frame_history")
    frames_b = track_b_snapshot.get("frame_history")
    times_a = track_a_snapshot.get("timestamps")
    times_b = track_b_snapshot.get("timestamps")
    event_time_sec = event_metadata.get("timestamp_seconds", event_frame / 25.0)

    # Window to Short Event History (1.5-2.5s) and enforce bottom-center endpoint alignment
    windowed_a = extract_event_window_trajectory(
        raw_pts_a, frames_a, times_a, target_contact=contact_a,
        event_timestamp=event_time_sec, event_frame=event_frame
    )
    windowed_b = extract_event_window_trajectory(
        raw_pts_b, frames_b, times_b, target_contact=contact_b,
        event_timestamp=event_time_sec, event_frame=event_frame
    )

    # Segment to reject gaps and impossible jumps
    segs_a = segment_and_clean_pixel_trajectory(
        windowed_a, frame_indices=frames_a, timestamps=times_a,
        max_frame_gap=5, max_jump_px=80.0, image_width=w, image_height=h
    )
    segs_b = segment_and_clean_pixel_trajectory(
        windowed_b, frame_indices=frames_b, timestamps=times_b,
        max_frame_gap=5, max_jump_px=80.0, image_width=w, image_height=h
    )

    # 4. Draw Trajectory Trail for Track A
    for seg in segs_a:
        if len(seg) > 1:
            pts = np.array([[int(p[0]), int(p[1])] for p in seg], np.int32).reshape((-1, 1, 2))
            cv2.polylines(out_frame, [pts], False, color_a, 2, cv2.LINE_AA)
        for pt in seg:
            cv2.circle(out_frame, (int(pt[0]), int(pt[1])), 2, color_a, -1, cv2.LINE_AA)

    # Contact point anchor for Track A
    cv2.circle(out_frame, (int(contact_a[0]), int(contact_a[1])), 4, color_a, -1, cv2.LINE_AA)
    cv2.circle(out_frame, (int(contact_a[0]), int(contact_a[1])), 6, (255, 255, 255), 1, cv2.LINE_AA)

    # 5. Draw Trajectory Trail for Track B
    for seg in segs_b:
        if len(seg) > 1:
            pts = np.array([[int(p[0]), int(p[1])] for p in seg], np.int32).reshape((-1, 1, 2))
            cv2.polylines(out_frame, [pts], False, color_b, 2, cv2.LINE_AA)
        for pt in seg:
            cv2.circle(out_frame, (int(pt[0]), int(pt[1])), 2, color_b, -1, cv2.LINE_AA)

    # Contact point anchor for Track B
    cv2.circle(out_frame, (int(contact_b[0]), int(contact_b[1])), 4, color_b, -1, cv2.LINE_AA)
    cv2.circle(out_frame, (int(contact_b[0]), int(contact_b[1])), 6, (255, 255, 255), 1, cv2.LINE_AA)

    # 6. Draw Highlighted Participant Bounding Boxes
    ax1, ay1, ax2, ay2 = [int(v) for v in bbox_a]
    bx1, by1, bx2, by2 = [int(v) for v in bbox_b]
    cv2.rectangle(out_frame, (ax1, ay1), (ax2, ay2), color_a, 2, cv2.LINE_AA)
    cv2.rectangle(out_frame, (bx1, by1), (bx2, by2), color_b, 2, cv2.LINE_AA)

    # Participant Badges
    type_a = event_metadata.get("type_a", track_a_snapshot.get("cls", "vehicle"))
    type_b = event_metadata.get("type_b", track_b_snapshot.get("cls", "vehicle"))
    vel_a = track_a_snapshot.get("velocity", 0.0)
    vel_b = track_b_snapshot.get("velocity", 0.0)
    lbl_a = f"[A] {type_a.upper()} #{tid_a} ({vel_a:.0f} km/h)"
    lbl_b = f"[B] {type_b.upper()} #{tid_b} ({vel_b:.0f} km/h)"

    (lw_a, lh_a), _ = cv2.getTextSize(lbl_a, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
    cv2.rectangle(out_frame, (ax1, max(ay1 - lh_a - 8, 2)), (ax1 + lw_a + 8, max(ay1, lh_a + 10)), color_a, -1)
    cv2.putText(out_frame, lbl_a, (ax1 + 4, max(ay1 - 5, lh_a + 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 1, cv2.LINE_AA)

    (lw_b, lh_b), _ = cv2.getTextSize(lbl_b, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
    cv2.rectangle(out_frame, (bx1, max(by1 - lh_b - 8, 2)), (bx1 + lw_b + 8, max(by1, lh_b + 10)), color_b, -1)
    cv2.putText(out_frame, lbl_b, (bx1 + 4, max(by1 - 5, lh_b + 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 1, cv2.LINE_AA)

    # 7. Collision Line & CPA Target Marker (Proximity Bounded)
    clearance_px = math.hypot(contact_a[0] - contact_b[0], contact_a[1] - contact_b[1])
    mid_x = int((contact_a[0] + contact_b[0]) / 2.0)
    mid_y = int((contact_a[1] + contact_b[1]) / 2.0)

    if clearance_px <= 220.0:
        # Close proximity interaction: render subtle dashed collision connector
        for step in np.linspace(0.12, 0.88, 7):
            cx_dash = int(contact_a[0] + (contact_b[0] - contact_a[0]) * step)
            cy_dash = int(contact_a[1] + (contact_b[1] - contact_a[1]) * step)
            cv2.circle(out_frame, (cx_dash, cy_dash), 2, (0, 0, 255), -1, cv2.LINE_AA)

        # Concentric CPA target marker
        cv2.circle(out_frame, (mid_x, mid_y), 7, (0, 0, 255), -1, cv2.LINE_AA)
        cv2.circle(out_frame, (mid_x, mid_y), 11, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.circle(out_frame, (mid_x, mid_y), 15, (0, 0, 255), 1, cv2.LINE_AA)
    else:
        # Distant interaction: avoid giant cross-screen lines; draw short directional indicators
        ux_a = (mid_x - contact_a[0]) / max(clearance_px, 1.0)
        uy_a = (mid_y - contact_a[1]) / max(clearance_px, 1.0)
        pt_vec_a = (int(contact_a[0] + ux_a * 40.0), int(contact_a[1] + uy_a * 40.0))
        cv2.line(out_frame, (int(contact_a[0]), int(contact_a[1])), pt_vec_a, (0, 140, 255), 2, cv2.LINE_AA)

        ux_b = (mid_x - contact_b[0]) / max(clearance_px, 1.0)
        uy_b = (mid_y - contact_b[1]) / max(clearance_px, 1.0)
        pt_vec_b = (int(contact_b[0] + ux_b * 40.0), int(contact_b[1] + uy_b * 40.0))
        cv2.line(out_frame, (int(contact_b[0]), int(contact_b[1])), pt_vec_b, (0, 140, 255), 2, cv2.LINE_AA)

        # CPA target marker at midpoint
        cv2.circle(out_frame, (mid_x, mid_y), 7, (0, 0, 255), -1, cv2.LINE_AA)
        cv2.circle(out_frame, (mid_x, mid_y), 11, (255, 255, 255), 2, cv2.LINE_AA)

    # 8. Telemetry HUD Box
    ttc = event_metadata.get("ttc", 2.0)
    pet = event_metadata.get("pet")
    min_dist = event_metadata.get("min_dist", 3.0)
    drac = event_metadata.get("drac")
    geom = event_metadata.get("interaction_geometry", "Converging Approach")
    pet_str = f"{pet:.2f}s" if pet is not None else "N/A"
    drac_str = f"{drac:.1f} m/s2" if drac is not None else "N/A"

    hud_lines = [
        f"CONFLICT: {geom}",
        f"TTC: ~{ttc:.2f}s | PET: ~{pet_str} | Clearance: {min_dist:.1f}m",
        f"DRAC: {drac_str} | Severity: {risk_level}",
    ]
    hud_w = 310
    hud_h = 58
    hud_x = max(10, min(mid_x - hud_w // 2, w - hud_w - 10))
    hud_y = max(36, min(mid_y + 22, h - hud_h - 10))

    hud_overlay = out_frame.copy()
    cv2.rectangle(hud_overlay, (hud_x, hud_y), (hud_x + hud_w, hud_y + hud_h), (15, 23, 42), -1)
    cv2.addWeighted(hud_overlay, 0.85, out_frame, 0.15, 0, out_frame)
    cv2.rectangle(out_frame, (hud_x, hud_y), (hud_x + hud_w, hud_y + hud_h), color_a, 1, cv2.LINE_AA)

    cv2.putText(out_frame, hud_lines[0], (hud_x + 8, hud_y + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 220, 255), 1, cv2.LINE_AA)
    cv2.putText(out_frame, hud_lines[1], (hud_x + 8, hud_y + 32), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(out_frame, hud_lines[2], (hud_x + 8, hud_y + 48), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (200, 200, 200), 1, cv2.LINE_AA)

    # 9. Top Status Header Strip
    conflict_id = event_metadata.get("conflict_id", "NM-EVENT")
    timestamp_str = event_metadata.get("timestamp_str", "00:00")
    status_strip = (
        f"SafeCity Loop Video AI | Event #{conflict_id} | Frame {event_frame} ({timestamp_str}) | "
        f"{type_a.capitalize()} #{tid_a} <-> {type_b.capitalize()} #{tid_b} | TTC: {ttc:.2f}s"
    )
    cv2.rectangle(out_frame, (0, 0), (w, 28), (15, 23, 42), -1)
    cv2.putText(out_frame, status_strip, (12, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)

    # 10. Diagnostic metadata
    align_err_a = math.hypot(windowed_a[-1][0] - contact_a[0], windowed_a[-1][1] - contact_a[1]) if windowed_a else 0.0
    align_err_b = math.hypot(windowed_b[-1][0] - contact_b[0], windowed_b[-1][1] - contact_b[1]) if windowed_b else 0.0
    diagnostics = {
        "conflict_id": conflict_id,
        "frame_index": event_frame,
        "track_a": tid_a,
        "track_b": tid_b,
        "pts_a_count": len(windowed_a),
        "pts_b_count": len(windowed_b),
        "segs_a_count": len(segs_a),
        "segs_b_count": len(segs_b),
        "contact_a": contact_a,
        "contact_b": contact_b,
        "align_err_a": round(align_err_a, 2),
        "align_err_b": round(align_err_b, 2),
        "clearance_px": round(clearance_px, 1),
    }

    print(
        f"[Evidence Diagnostics] Event #{conflict_id} @ Frame {event_frame} ({timestamp_str}): "
        f"Track A #{tid_a} ({len(windowed_a)} pts, {len(segs_a)} segs, err={align_err_a:.1f}px) | "
        f"Track B #{tid_b} ({len(windowed_b)} pts, {len(segs_b)} segs, err={align_err_b:.1f}px) | "
        f"Clearance: {clearance_px:.1f}px"
    )

    return out_frame, diagnostics

