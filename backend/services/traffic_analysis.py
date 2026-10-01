"""
SafeCity Loop V2 — Traffic Video Analysis & Surrogate Safety Engine
Pipeline: VIDEO -> FRAME SAMPLING -> YOLO OBJECT DETECTION -> BYTETrack ->
          TRAJECTORIES -> CONFLICT ANALYSIS (TTC, PET, Min Distance) -> PRIVACY MASKING

Monocular Computer Vision Notice:
All speed, distance, TTC, and PET measurements derived from ordinary monocular video
are strictly approximate estimates. Labeled as 'Potential traffic conflict' / 'Near-miss candidate'.
"""
import os
import uuid
import math
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Tuple, Optional, Any
import cv2
import numpy as np


@dataclass
class TrackedObject:
    track_id: int
    object_type: str         # car, motorcycle, bus, truck, bicycle, pedestrian
    centroid: Tuple[float, float]
    bbox: Tuple[float, float, float, float]  # x1, y1, x2, y2
    pixel_contact: Tuple[float, float] = (0.0, 0.0)  # ((x1+x2)/2, y2)
    velocity: float = 0.0                    # estimated speed (px/sec or approx km/h)
    direction: str = "Stationary"            # Northbound, Southbound, Eastbound, Westbound, etc.
    heading_deg: float = 0.0
    pixel_history: List[Tuple[float, float]] = field(default_factory=list)
    ground_history: List[Tuple[float, float]] = field(default_factory=list)
    timestamps: List[float] = field(default_factory=list)
    frame_history: List[int] = field(default_factory=list)
    trajectory: List[Tuple[float, float]] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "track_id": self.track_id,
            "object_type": self.object_type,
            "centroid": [round(self.centroid[0], 1), round(self.centroid[1], 1)],
            "bbox": [round(x, 1) for x in self.bbox],
            "pixel_contact": [round(self.pixel_contact[0], 1), round(self.pixel_contact[1], 1)],
            "velocity": round(self.velocity, 1),
            "direction": self.direction,
            "trajectory_length": len(self.pixel_history),
        }


from ml.traffic.trajectory_analysis import (
    TrajectoryEngine, PerspectiveCalibrator, TTCEngineV2,
    ConflictEvaluation, compute_exposure_normalized_conflict_rates,
    render_conflict_evidence, segment_and_clean_pixel_trajectory,
    extract_event_window_trajectory
)



@dataclass
class ConflictCandidate:
    conflict_id: str                   # e.g. "NM-2026-0012"
    timestamp_str: str                 # e.g. "00:00:14"
    timestamp_seconds: float
    object_type_a: str                 # e.g. "vehicle"
    object_type_b: str                 # e.g. "pedestrian"
    object_types: List[str]            # ["vehicle", "pedestrian"]
    track_ids: List[int]               # [3, 7]
    ttc: float                         # Approximate Time-To-Collision in seconds
    pet: Optional[float]               # Approximate Post-Encroachment Time in seconds
    minimum_distance: float            # Approximate minimum clearance in meters
    risk_level: str                    # LOW / MEDIUM / HIGH / CRITICAL
    event_severity: str                # Standardized municipal severity
    conflict_zone: str                 # e.g. "Pedestrian Crossing Corridor"
    direction: str                     # e.g. "Northbound ⟷ Crosswalk"
    frame_index: int
    confidence: float = 0.88           # Model tracking confidence
    snapshot_url: Optional[str] = None
    review_status: str = "pending_review"
    disclaimer: str = "Approximate monocular surrogate safety estimate. Potential traffic conflict."
    drac: Optional[float] = None
    relative_speed_kmh: Optional[float] = None
    interaction_geometry: Optional[str] = None
    trajectory_convergence: Optional[bool] = None
    why_conflict_explanation: Optional[Dict[str, Any]] = None

    def to_dict(self) -> dict:
        return {
            "conflict_id": self.conflict_id,
            "timestamp_str": self.timestamp_str,
            "timestamp_seconds": round(self.timestamp_seconds, 2),
            "object_type_a": self.object_type_a,
            "object_type_b": self.object_type_b,
            "object_types": self.object_types,
            "track_ids": self.track_ids,
            "ttc": round(self.ttc, 2),
            "pet": round(self.pet, 2) if self.pet is not None else None,
            "minimum_distance": round(self.minimum_distance, 1),
            "risk_level": self.risk_level,
            "event_severity": self.event_severity,
            "conflict_zone": self.conflict_zone,
            "direction": self.direction,
            "frame_index": self.frame_index,
            "confidence": round(self.confidence, 3),
            "snapshot_url": self.snapshot_url,
            "review_status": self.review_status,
            "disclaimer": self.disclaimer,
            "drac": round(self.drac, 2) if self.drac is not None else None,
            "relative_speed_kmh": round(self.relative_speed_kmh, 1) if self.relative_speed_kmh is not None else None,
            "interaction_geometry": self.interaction_geometry,
            "trajectory_convergence": self.trajectory_convergence,
            "why_conflict_explanation": self.why_conflict_explanation,
        }


# Backward compatibility alias
NearMissEvent = ConflictCandidate


@dataclass
class TrafficAnalysisResult:
    is_demo_mode: bool
    model_status: str
    status_message: str
    junction_id: Optional[int]
    road_id: Optional[int]
    duration_seconds: float
    total_frames: int
    processed_frames: int
    tracked_objects_count: int
    object_class_counts: Dict[str, int]
    near_miss_count: int
    near_misses: List[ConflictCandidate]
    tracked_objects_summary: List[Dict[str, Any]]
    original_video_url: str
    processed_video_url: Optional[str]
    conflict_snapshots: List[str]
    privacy_applied: bool
    video_id: str = ""
    is_demo_video: bool = False
    processing_time_sec: float = 0.0
    timeline: List[Dict[str, Any]] = field(default_factory=list)
    exposure_normalized_rates: Optional[Dict[str, Any]] = None
    traffic_model_meta: Optional[Dict[str, Any]] = None
    tracking_health_status: str = "ANALYSIS_HEALTHY"
    metric_calibration_status: str = "NOMINAL_PIXEL_ESTIMATE"
    tracking_health_summary: Optional[Dict[str, Any]] = None
    privacy_notice: str = (
        "Video processing is intended to minimize unnecessary storage of "
        "personally identifiable visual information (faces and license plates anonymized)."
    )

    def to_dict(self) -> dict:
        return {
            "video_id": self.video_id,
            "is_demo_mode": self.is_demo_mode,
            "is_demo_video": self.is_demo_video,
            "model_status": self.model_status,
            "status_message": self.status_message,
            "junction_id": self.junction_id,
            "road_id": self.road_id,
            "duration_seconds": round(self.duration_seconds, 2),
            "total_frames": self.total_frames,
            "processed_frames": self.processed_frames,
            "processing_time_sec": round(self.processing_time_sec, 2),
            "tracked_objects_count": self.tracked_objects_count,
            "object_class_counts": self.object_class_counts,
            "near_miss_count": self.near_miss_count,
            "near_misses": [m.to_dict() for m in self.near_misses],
            "timeline": self.timeline,
            "tracked_objects_summary": self.tracked_objects_summary,
            "original_video_url": self.original_video_url,
            "processed_video_url": self.processed_video_url,
            "conflict_snapshots": self.conflict_snapshots,
            "privacy_applied": self.privacy_applied,
            "privacy_notice": self.privacy_notice,
            "exposure_normalized_rates": self.exposure_normalized_rates,
            "traffic_model_meta": self.traffic_model_meta,
            "tracking_health_status": self.tracking_health_status,
            "metric_calibration_status": self.metric_calibration_status,
            "tracking_health_summary": self.tracking_health_summary,
        }


class TrafficAnalysisService:
    """
    Traffic Conflict & Near-Miss Intelligence Service using YOLOv8 + ByteTrack.
    Configurable conflict thresholds (TTC, PET, distance).
    Integrated privacy masking (face and license plate blurring).
    """

    # COCO Class mappings to urban traffic categories
    CLASS_MAP = {
        0: "pedestrian",   # person
        1: "bicycle",      # bicycle
        2: "car",          # car
        3: "motorcycle",   # motorcycle
        5: "bus",          # bus
        7: "truck",        # truck
    }

    def __init__(
        self,
        model_path: Optional[str] = None,
        default_ttc_threshold: float = 2.0,
        default_distance_threshold_meters: float = 3.5,
        camera_mode: Optional[str] = None,
    ):
        env_model = os.getenv("YOLO_TRAFFIC_MODEL_PATH") or os.getenv("TRAFFIC_MODEL_PATH")
        # Selection hierarchy: explicit arg -> custom fine-tuned v2 -> custom fine-tuned v1 -> env -> root baseline -> yolov8n
        if model_path:
            self.model_path = model_path
            self.model_name = "SafeCity Traffic YOLO26s (Custom)"
        elif os.path.exists("models/traffic_yolo26s_v2.pt"):
            self.model_path = "models/traffic_yolo26s_v2.pt"
            self.model_name = "SafeCity Traffic YOLO26s v2"
        elif os.path.exists("models/traffic_yolo26s_v1.pt"):
            self.model_path = "models/traffic_yolo26s_v1.pt"
            self.model_name = "SafeCity Traffic YOLO26s v1"
        elif env_model and os.path.exists(env_model):
            self.model_path = env_model
            self.model_name = "SafeCity Traffic Model"
        elif os.path.exists("yolo26s.pt"):
            self.model_path = "yolo26s.pt"
            self.model_name = "SafeCity Traffic YOLO26s"
        else:
            self.model_path = env_model or "yolov8n.pt"
            self.model_name = "SafeCity Traffic Detector"

        self.tracker_name = os.getenv("TRAFFIC_TRACKER", "bytetrack.yaml")
        self.runtime_mode = os.getenv("TRAFFIC_RUNTIME", "pytorch")
        self.camera_mode = camera_mode or os.getenv("TRAFFIC_CAMERA_MODE", "aerial_intersection")
        self.ttc_threshold = default_ttc_threshold
        self.min_dist_threshold_meters = default_distance_threshold_meters
        self.model = None
        self.class_map = self.CLASS_MAP
        self._initialize_model()

    def _initialize_model(self) -> None:
        try:
            from ultralytics import YOLO
            self.model = YOLO(self.model_path)
            if hasattr(self.model, "names") and self.model.names:
                self.class_map = {int(idx): name.lower() for idx, name in self.model.names.items()}
            else:
                self.class_map = self.CLASS_MAP
            print(f"[Traffic AI] Loaded traffic model: {self.model_path} ({len(self.class_map)} classes: {list(self.class_map.values())[:7]})")
        except Exception as e:
            print(f"[Traffic AI] Error loading traffic model ({self.model_path}): {e}")
            self.model = None
            self.class_map = self.CLASS_MAP

    def _compute_direction(self, p_start: Tuple[float, float], p_end: Tuple[float, float], speed_kmh: float = 0.0) -> Tuple[str, float]:
        dx = p_end[0] - p_start[0]
        dy = p_end[1] - p_start[1]
        dist = math.hypot(dx, dy)
        if speed_kmh < 2.0 and dist < 0.4:
            return "Stationary / Yielding", 0.0

        angle_rad = math.atan2(dy, dx)
        angle_deg = (math.degrees(angle_rad) + 360) % 360

        if 45 <= angle_deg < 135:
            return "Northbound", angle_deg
        elif 135 <= angle_deg < 225:
            return "Westbound", angle_deg
        elif 225 <= angle_deg < 315:
            return "Southbound", angle_deg
        else:
            return "Eastbound", angle_deg

    def _determine_conflict_zone(self, p1: Tuple[float, float], p2: Tuple[float, float], w: int, h: int) -> str:
        cx = (p1[0] + p2[0]) / 2.0
        cy = (p1[1] + p2[1]) / 2.0

        rel_x = cx / max(w, 1)
        rel_y = cy / max(h, 1)

        if 0.3 <= rel_x <= 0.7 and 0.3 <= rel_y <= 0.7:
            return "Intersection Center Conflict"
        elif rel_y > 0.65:
            return "Pedestrian Crossing Corridor"
        elif rel_x < 0.35 or rel_x > 0.65:
            return "Lateral Merge / Turn Conflict"
        else:
            return "Approach Lane Conflict"

    def _apply_privacy_mask(self, frame: np.ndarray, bboxes: List[Tuple[float, float, float, float, str]]) -> np.ndarray:
        """
        Blurs potential face regions (upper pedestrian bounding box) and
        license plate regions (lower vehicle bounding box) for privacy compliance.
        """
        out_frame = frame.copy()
        h, w = out_frame.shape[:2]

        for x1, y1, x2, y2, cls_name in bboxes:
            ix1, iy1, ix2, iy2 = max(0, int(x1)), max(0, int(y1)), min(w, int(x2)), min(h, int(y2))
            bw, bh = ix2 - ix1, iy2 - iy1
            if bw <= 4 or bh <= 4:
                continue

            if cls_name == "pedestrian":
                # Blur upper 25% of pedestrian bounding box (head/face region)
                face_y2 = min(iy1 + int(bh * 0.28), h)
                roi = out_frame[iy1:face_y2, ix1:ix2]
                if roi.size > 0:
                    blurred = cv2.GaussianBlur(roi, (23, 23), 30)
                    out_frame[iy1:face_y2, ix1:ix2] = blurred

            elif cls_name in ["car", "bus", "truck", "motorcycle"]:
                # Blur lower 25% of vehicle bounding box (license plate area)
                plate_y1 = max(iy2 - int(bh * 0.25), 0)
                roi = out_frame[plate_y1:iy2, ix1:ix2]
                if roi.size > 0:
                    blurred = cv2.GaussianBlur(roi, (21, 21), 25)
                    out_frame[plate_y1:iy2, ix1:ix2] = blurred

        return out_frame

    def _render_conflict_evidence_image(
        self,
        base_frame_bytes: bytes,
        file_id: str,
        output_dir: str,
        conflict_id: str,
        frame_index: int,
        timestamp_str: str,
        track_id_a: int,
        track_id_b: int,
        type_a: str,
        type_b: str,
        ttc: float,
        pet: Optional[float],
        min_dist: float,
        drac: Optional[float],
        risk_level: str,
        interaction_geometry: str,
        tracks_snapshot: Dict[int, Dict[str, Any]],
        apply_privacy: bool = True,
        timestamp_seconds: Optional[float] = None,
    ) -> str:
        """
        Renders an event-specific visual evidence keyframe artifact:
        - Base video frame from exact conflict occurrence
        - Privacy masking applied (faces & license plates blurred)
        - Highlights ONLY participating Track A and Track B
        - Renders trajectory trails ONLY for Track A and Track B using clean bottom-center pixel paths
        - Rejects gaps and impossible jumps to eliminate zig-zag artifacts
        - CPA marker and proximity collision vector
        - Saves to unique path: uploads/traffic/conflict_{file_id}_{idA}_{idB}_f{frame_index}.jpg
        """
        raw_arr = np.frombuffer(base_frame_bytes, np.uint8)
        frame = cv2.imdecode(raw_arr, cv2.IMREAD_COLOR)
        if frame is None:
            frame = np.zeros((720, 1280, 3), dtype=np.uint8)

        info_a = dict(tracks_snapshot.get(track_id_a, {}))
        info_b = dict(tracks_snapshot.get(track_id_b, {}))
        info_a["track_id"] = track_id_a
        info_b["track_id"] = track_id_b

        metadata = {
            "conflict_id": conflict_id,
            "timestamp_str": timestamp_str,
            "timestamp_seconds": timestamp_seconds,
            "type_a": type_a,
            "type_b": type_b,
            "ttc": ttc,
            "pet": pet,
            "min_dist": min_dist,
            "drac": drac,
            "risk_level": risk_level,
            "interaction_geometry": interaction_geometry,
        }

        rendered_frame, _ = render_conflict_evidence(
            frame=frame,
            track_a_snapshot=info_a,
            track_b_snapshot=info_b,
            event_frame=frame_index,
            event_metadata=metadata,
            other_tracks=tracks_snapshot,
            apply_privacy=apply_privacy,
        )

        unique_filename = f"conflict_{file_id}_{track_id_a}_{track_id_b}_f{frame_index}.jpg"
        snap_path = os.path.join(output_dir, unique_filename)
        cv2.imwrite(snap_path, rendered_frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
        return f"/uploads/traffic/{unique_filename}"

    def analyze_video(
        self,
        video_bytes: bytes,
        filename: str = "traffic.mp4",
        junction_id: Optional[int] = None,
        road_id: Optional[int] = None,
        output_dir: str = "uploads/traffic",
        max_process_frames: Optional[int] = None,
        ttc_threshold: Optional[float] = None,
        apply_privacy: bool = True,
        is_demo_video: bool = False,
        camera_mode: Optional[str] = None,
    ) -> TrafficAnalysisResult:
        """
        Processes video frames with YOLOv8 + ByteTrack tracking and computes
        approximate TTC, PET, minimum distance, and trajectory vectors.
        """
        start_time = time.time()
        os.makedirs(output_dir, exist_ok=True)
        file_id = str(uuid.uuid4())[:12]
        ext = os.path.splitext(filename)[1].lower()
        if ext not in [".mp4", ".mov", ".avi"]:
            ext = ".mp4"

        orig_filename = f"{file_id}_orig{ext}"
        proc_filename = f"{file_id}_processed.mp4"
        orig_path = os.path.join(output_dir, orig_filename)
        proc_path = os.path.join(output_dir, proc_filename)

        with open(orig_path, "wb") as f:
            f.write(video_bytes)

        cap = cv2.VideoCapture(orig_path)
        if not cap.isOpened():
            raise ValueError("Unable to read traffic video. File may be corrupted or unsupported format (MP4/MOV/AVI).")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = float(total_frames) / max(fps, 1.0)

        effective_ttc_limit = ttc_threshold or self.ttc_threshold
        effective_camera_mode = camera_mode or self.camera_mode

        # Dense tracking frame step (target ~12-15 fps for reliable ByteTrack association)
        target_tracking_fps = min(fps, 15.0)
        frame_step = max(1, int(round(fps / target_tracking_fps)))
        effective_fps = fps / frame_step

        self._unconfirmed_seq = 0
        tracks_history: Dict[int, TrackedObject] = {}
        conflicts_raw: List[ConflictCandidate] = []
        conflict_snapshots: List[str] = []
        raw_frame_cache: Dict[int, bytes] = {}
        frame_tracks_snapshot: Dict[int, Dict[int, Dict[str, Any]]] = {}
        object_class_counts: Dict[str, int] = {
            "car": 0, "pedestrian": 0, "motorcycle": 0, "bus": 0, "truck": 0, "bicycle": 0
        }

        # Video writer for annotated processed output
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_writer = cv2.VideoWriter(proc_path, fourcc, effective_fps, (width, height))

        is_demo_synth = (is_demo_video or ("demo" in filename.lower())) and (self.model is None)
        is_demo = (self.model is None)
        current_frame_idx = 0
        processed_count = 0

        # Ground-Plane Perspective Calibrator and Kinematic Trajectory Engines
        calibrator = PerspectiveCalibrator(mode=effective_camera_mode, image_width=width, image_height=height)
        traj_engine = TrajectoryEngine(calibrator=calibrator)
        ttc_engine = TTCEngineV2(
            default_ttc_threshold=effective_ttc_limit,
            default_distance_threshold=self.min_dist_threshold_meters,
        )
        pixels_per_meter = 35.0  # legacy fallback

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if max_process_frames is not None and processed_count >= max_process_frames:
                break

            if current_frame_idx % frame_step != 0:
                current_frame_idx += 1
                continue

            current_time_sec = current_frame_idx / max(fps, 1.0)
            active_tracks_this_frame: List[Tuple[int, str, Tuple[float, float], Tuple[float, float, float, float]]] = []
            privacy_boxes: List[Tuple[float, float, float, float, str]] = []

            if is_demo_synth:
                # Deterministic ground-truth demo scenario tracking matching generate_demo_video.py
                car1_y = int((height + 80) - (current_frame_idx * 4.2)) % (height + 160) - 80
                car1_box = (250.0, float(car1_y), 290.0, float(car1_y + 60))
                car1_c = (270.0, float(car1_y + 30))

                car2_y = int((current_frame_idx * 5.8) - 70) % (height + 160) - 70
                car2_box = (350.0, float(car2_y), 395.0, float(car2_y + 70))
                car2_c = (372.5, float(car2_y + 35))

                moto_y = int((height + 60) - (current_frame_idx * 6.5)) % (height + 120) - 60
                moto_box = (185.0, float(moto_y), 195.0, float(moto_y + 35))
                moto_c = (190.0, float(moto_y + 17.5))

                ped_prog = (current_frame_idx % 160) / 160.0
                ped_x = 140.0 + (ped_prog * 350.0)
                ped_y = 285.0
                ped1_box = (ped_x - 10.0, ped_y - 14.0, ped_x + 10.0, ped_y + 14.0)
                ped1_c = (ped_x, ped_y)

                ped2_y = float(int(20 + (current_frame_idx * 1.8)) % height)
                ped2_box = (532.0, ped2_y - 10.0, 548.0, ped2_y + 10.0)
                ped2_c = (540.0, ped2_y)

                demo_items = [
                    (12, "car", car1_c, car1_box, 38.0, "Northbound"),
                    (15, "car", car2_c, car2_box, 42.0, "Southbound"),
                    (8, "motorcycle", moto_c, moto_box, 45.0, "Northbound"),
                    (7, "pedestrian", ped1_c, ped1_box, 4.5, "Eastbound"),
                    (9, "pedestrian", ped2_c, ped2_box, 4.0, "Southbound"),
                ]

                for tid, cname, centroid, bbox, vel, dire in demo_items:
                    contact_px = ((bbox[0] + bbox[2]) / 2.0, float(bbox[3]))
                    active_tracks_this_frame.append((tid, cname, centroid, bbox))
                    privacy_boxes.append((bbox[0], bbox[1], bbox[2], bbox[3], cname))
                    t_state = traj_engine.update_track(
                        track_id=tid,
                        object_class=cname,
                        bbox=bbox,
                        timestamp=current_time_sec,
                        frame_index=current_frame_idx,
                    )
                    if tid not in tracks_history:
                        object_class_counts[cname] = object_class_counts.get(cname, 0) + 1
                        tracks_history[tid] = TrackedObject(
                            track_id=tid,
                            object_type=cname,
                            centroid=centroid,
                            bbox=bbox,
                            pixel_contact=contact_px,
                            velocity=vel,
                            direction=dire,
                            pixel_history=[contact_px],
                            ground_history=[(t_state.ground_pos[0], t_state.ground_pos[1])],
                            frame_history=[current_frame_idx],
                            timestamps=[current_time_sec],
                            trajectory=[contact_px],
                        )
                    else:
                        obj = tracks_history[tid]
                        obj.centroid = centroid
                        obj.bbox = bbox
                        obj.pixel_contact = contact_px
                        obj.velocity = vel
                        obj.direction = dire
                        obj.pixel_history.append(contact_px)
                        obj.ground_history.append((t_state.ground_pos[0], t_state.ground_pos[1]))
                        obj.frame_history.append(current_frame_idx)
                        obj.timestamps.append(current_time_sec)
                        obj.trajectory.append(contact_px)
                        if len(obj.pixel_history) > 60:
                            obj.pixel_history.pop(0)
                            obj.ground_history.pop(0)
                            obj.frame_history.pop(0)
                            obj.timestamps.pop(0)
                            obj.trajectory.pop(0)

            elif not is_demo and self.model is not None:
                # Run YOLO tracking with configured tracker (ByteTrack / BoT-SORT)
                results = self.model.track(
                    frame,
                    persist=True,
                    classes=list(self.class_map.keys()),
                    conf=0.25,
                    tracker=self.tracker_name,
                    verbose=False,
                )

                if results and results[0].boxes is not None:
                    boxes = results[0].boxes
                    for box in boxes:
                        cls_id = int(box.cls[0])
                        class_name = self.class_map.get(cls_id, "car")

                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        cx = (x1 + x2) / 2.0
                        cy = (y1 + y2) / 2.0
                        contact_px = ((x1 + x2) / 2.0, float(y2))

                        # Apply privacy masking to all detected objects regardless of tracker confirmation
                        privacy_boxes.append((x1, y1, x2, y2, class_name))

                        # Unconfirmed ByteTrack detections have box.id is None.
                        # Do NOT allocate synthetic 900000+ IDs into tracks_history or trajectory engine!
                        if box.id is None:
                            continue

                        track_id = int(box.id[0])

                        # Update Calibrated Kinematic Trajectory State
                        t_state = traj_engine.update_track(
                            track_id=track_id,
                            object_class=class_name,
                            bbox=(x1, y1, x2, y2),
                            timestamp=current_time_sec,
                            frame_index=current_frame_idx,
                        )

                        active_tracks_this_frame.append((track_id, class_name, (cx, cy), (x1, y1, x2, y2)))

                        if track_id not in tracks_history:
                            object_class_counts[class_name] = object_class_counts.get(class_name, 0) + 1
                            tracks_history[track_id] = TrackedObject(
                                track_id=track_id,
                                object_type=class_name,
                                centroid=(cx, cy),
                                bbox=(x1, y1, x2, y2),
                                pixel_contact=contact_px,
                                velocity=round(t_state.velocity_kmh, 1),
                                direction=self._compute_direction(t_state.history_ground[0], t_state.history_ground[-1], speed_kmh=t_state.velocity_kmh)[0] if len(t_state.history_ground) >= 2 else "Stationary",
                                heading_deg=t_state.heading_deg,
                                pixel_history=[contact_px],
                                ground_history=[(t_state.ground_pos[0], t_state.ground_pos[1])],
                                frame_history=[current_frame_idx],
                                timestamps=[current_time_sec],
                                trajectory=[contact_px],
                            )
                        else:
                            obj = tracks_history[track_id]
                            obj.centroid = (cx, cy)
                            obj.bbox = (x1, y1, x2, y2)
                            obj.pixel_contact = contact_px
                            obj.velocity = round(t_state.velocity_kmh, 1)
                            obj.heading_deg = t_state.heading_deg
                            obj.pixel_history.append(contact_px)
                            obj.ground_history.append((t_state.ground_pos[0], t_state.ground_pos[1]))
                            obj.frame_history.append(current_frame_idx)
                            obj.timestamps.append(current_time_sec)
                            obj.trajectory.append(contact_px)
                            if len(obj.pixel_history) > 60:
                                obj.pixel_history.pop(0)
                                obj.ground_history.pop(0)
                                obj.frame_history.pop(0)
                                obj.timestamps.pop(0)
                                obj.trajectory.pop(0)

                            if len(t_state.history_ground) >= 2:
                                obj.direction, _ = self._compute_direction(t_state.history_ground[0], t_state.history_ground[-1], speed_kmh=obj.velocity)

            # Snapshot active tracks state on this frame for event-specific evidence rendering
            current_frame_tracks: Dict[int, Dict[str, Any]] = {}
            for tid, cname, centroid, bbox in active_tracks_this_frame:
                obj = tracks_history.get(tid)
                contact_px = obj.pixel_contact if obj else ((bbox[0] + bbox[2]) / 2.0, float(bbox[3]))
                current_frame_tracks[tid] = {
                    "cls": cname,
                    "centroid": centroid,
                    "bbox": bbox,
                    "pixel_contact": contact_px,
                    "velocity": obj.velocity if obj else 0.0,
                    "pixel_history": list(obj.pixel_history) if obj else [contact_px],
                    "ground_history": list(obj.ground_history) if obj else [],
                    "frame_history": list(obj.frame_history) if obj else [current_frame_idx],
                    "timestamps": list(obj.timestamps) if obj else [current_time_sec],
                    "trajectory": list(obj.pixel_history) if obj else [contact_px],
                }

            # Apply privacy masking if enabled
            if apply_privacy and privacy_boxes:
                annotated_frame = self._apply_privacy_mask(frame, privacy_boxes)
            else:
                annotated_frame = frame.copy()

            frame_conflict_active = False

            # Draw tracked bounding boxes, labels, and trajectory trails
            for track_id, cls_name, (cx, cy), (x1, y1, x2, y2) in active_tracks_this_frame:
                obj = tracks_history.get(track_id)
                color = (255, 120, 0) if cls_name in ["car", "bus", "truck"] else (0, 255, 100) if cls_name == "pedestrian" else (0, 200, 255)
                cv2.rectangle(annotated_frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)

                speed_str = f" {obj.velocity:.0f}km/h" if obj and obj.velocity > 1.0 else ""
                dir_str = f" [{obj.direction[:1]}]" if obj and obj.direction != "Stationary / Yielding" else ""
                label = f"{cls_name.capitalize()} #{track_id}{speed_str}{dir_str}"
                cv2.putText(annotated_frame, label, (int(x1), max(int(y1) - 6, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 2)

                # Draw movement trail on live video with gap & jump filtering
                if obj and len(obj.pixel_history) > 1:
                    segs = segment_and_clean_pixel_trajectory(
                        obj.pixel_history, frame_indices=obj.frame_history, timestamps=obj.timestamps,
                        max_frame_gap=5, max_jump_px=80.0, image_width=width, image_height=height
                    )
                    for seg in segs:
                        if len(seg) > 1:
                            pts = np.array([[int(p[0]), int(p[1])] for p in seg], np.int32).reshape((-1, 1, 2))
                            cv2.polylines(annotated_frame, [pts], False, color, 1, cv2.LINE_AA)


            # ── Conflict Engine V2: 2D CPA Kinematics & Interaction Classification ──
            n_tracks = len(active_tracks_this_frame)
            for i in range(n_tracks):
                for j in range(i + 1, n_tracks):
                    id1, type1, c1, bbox1 = active_tracks_this_frame[i]
                    id2, type2, c2, bbox2 = active_tracks_this_frame[j]

                    state1 = traj_engine.tracks.get(id1)
                    state2 = traj_engine.tracks.get(id2)
                    
                    hist1 = tracks_history.get(id1)
                    hist2 = tracks_history.get(id2)
                    if not hist1 or not hist2:
                        continue
                    if len(hist1.pixel_history) < 2 or len(hist2.pixel_history) < 2:
                        continue

                    conf_eval: Optional[ConflictEvaluation] = None
                    if state1 and state2:
                        conf_eval = ttc_engine.compute_conflict(state1, state2, conflict_id_prefix=f"NM-{file_id[:4]}")

                    # Fallback convergence check for legacy/synthetic demo sequences
                    d_px = float(math.hypot(c1[0] - c2[0], c1[1] - c2[1]))
                    d_meters = d_px / pixels_per_meter

                    if conf_eval is not None and conf_eval.is_valid_conflict:
                        frame_conflict_active = True
                        mins = int(current_time_sec // 60)
                        secs = int(current_time_sec % 60)
                        time_str = f"{mins:02d}:{secs:02d}"
                        dir_pair = f"{hist1.direction} ⟷ {hist2.direction}"

                        candidate = ConflictCandidate(
                            conflict_id=conf_eval.conflict_id,
                            timestamp_str=time_str,
                            timestamp_seconds=current_time_sec,
                            object_type_a=type1,
                            object_type_b=type2,
                            object_types=[type1, type2],
                            track_ids=[id1, id2],
                            ttc=conf_eval.ttc_seconds or effective_ttc_limit,
                            pet=conf_eval.pet_seconds,
                            minimum_distance=conf_eval.distance_meters,
                            risk_level=conf_eval.severity,
                            event_severity=conf_eval.severity,
                            conflict_zone=conf_eval.conflict_zone_label,
                            direction=dir_pair,
                            frame_index=current_frame_idx,
                            confidence=min(0.85 + (1.0 / max(conf_eval.ttc_seconds or 1.0, 0.5)) * 0.05, 0.96),
                            drac=conf_eval.drac_mps2,
                            relative_speed_kmh=conf_eval.relative_speed_kmh,
                            interaction_geometry=conf_eval.interaction_type,
                            trajectory_convergence=conf_eval.trajectory_convergence,
                            why_conflict_explanation=conf_eval.explanation,
                        )
                        conflicts_raw.append(candidate)

                        # Draw visual collision indicator & warning (proximity bounded)
                        if d_px <= 220.0:
                            cv2.line(annotated_frame, (int(c1[0]), int(c1[1])), (int(c2[0]), int(c2[1])), (0, 0, 255), 2, cv2.LINE_AA)
                        mid_x = int((c1[0] + c2[0]) / 2)
                        mid_y = int((c1[1] + c2[1]) / 2)
                        cv2.circle(annotated_frame, (mid_x, mid_y), 6, (0, 0, 255), -1, cv2.LINE_AA)
                        warn_text = f"CONFLICT: TTC ~{candidate.ttc:.1f}s | {candidate.minimum_distance:.1f}m | {candidate.risk_level}"
                        cv2.putText(annotated_frame, warn_text, (max(mid_x - 70, 10), max(mid_y - 8, 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 255), 2, cv2.LINE_AA)


                    elif is_demo_synth and len(hist1.trajectory) >= 3 and len(hist2.trajectory) >= 3:
                        idx_prev = -3
                        p1_prev = hist1.trajectory[idx_prev]
                        p2_prev = hist2.trajectory[idx_prev]
                        dt = hist1.timestamps[-1] - hist1.timestamps[idx_prev]

                        if dt > 0.04:
                            d_prev_px = float(math.hypot(p1_prev[0] - p2_prev[0], p1_prev[1] - p2_prev[1]))
                            v_rel_px = (d_prev_px - d_px) / dt

                            if v_rel_px > 12.0 and d_meters < 18.0:
                                ttc = d_px / v_rel_px
                                pet = max(0.4, min(ttc * 0.75 + 0.2, 3.0))

                                if ttc <= effective_ttc_limit and d_meters <= self.min_dist_threshold_meters:
                                    frame_conflict_active = True
                                    risk_lvl = "CRITICAL" if ttc < 1.0 else "HIGH" if ttc < 1.6 else "MEDIUM"
                                    zone = self._determine_conflict_zone(c1, c2, width, height)
                                    mins = int(current_time_sec // 60)
                                    secs = int(current_time_sec % 60)
                                    time_str = f"{mins:02d}:{secs:02d}"
                                    dir_pair = f"{hist1.direction} ⟷ {hist2.direction}"

                                    candidate = ConflictCandidate(
                                        conflict_id=f"NM-{file_id[:4]}-{id1}-{id2}",
                                        timestamp_str=time_str,
                                        timestamp_seconds=current_time_sec,
                                        object_type_a=type1,
                                        object_type_b=type2,
                                        object_types=[type1, type2],
                                        track_ids=[id1, id2],
                                        ttc=ttc,
                                        pet=pet,
                                        minimum_distance=d_meters,
                                        risk_level=risk_lvl,
                                        event_severity=risk_lvl,
                                        conflict_zone=zone,
                                        direction=dir_pair,
                                        frame_index=current_frame_idx,
                                        confidence=min(0.85 + (1.0 / max(ttc, 0.5)) * 0.05, 0.96),
                                        drac=round((v_rel_px / 35.0) ** 2 / (2.0 * max(d_meters, 0.5)), 2),
                                        relative_speed_kmh=round((v_rel_px / 35.0) * 3.6, 1),
                                        interaction_geometry="Converging Approach",
                                        trajectory_convergence=True,
                                        why_conflict_explanation={
                                            "participants": f"{type1.capitalize()} #{id1} vs {type2.capitalize()} #{id2}",
                                            "interaction_geometry": "Converging Approach",
                                            "ttc_seconds": round(ttc, 2),
                                            "pet_seconds": round(pet, 2),
                                            "distance_clearance_m": round(d_meters, 1),
                                            "severity_level": risk_lvl,
                                        },
                                    )
                                    conflicts_raw.append(candidate)

                                    if d_px <= 220.0:
                                        cv2.line(annotated_frame, (int(c1[0]), int(c1[1])), (int(c2[0]), int(c2[1])), (0, 0, 255), 2, cv2.LINE_AA)
                                    mid_x = int((c1[0] + c2[0]) / 2)
                                    mid_y = int((c1[1] + c2[1]) / 2)
                                    cv2.circle(annotated_frame, (mid_x, mid_y), 6, (0, 0, 255), -1, cv2.LINE_AA)
                                    warn_text = f"CONFLICT: TTC ~{ttc:.1f}s | {d_meters:.1f}m"
                                    cv2.putText(annotated_frame, warn_text, (max(mid_x - 70, 10), max(mid_y - 8, 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2, cv2.LINE_AA)


            # Watermark Status Banner
            status_tag = f"SafeCity Loop Video AI | Frame {current_frame_idx} | Tracks: {len(active_tracks_this_frame)} | Privacy: {'Active' if apply_privacy else 'Off'}"
            cv2.rectangle(annotated_frame, (0, 0), (width, 28), (15, 23, 42), -1)
            cv2.putText(annotated_frame, status_tag, (10, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)

            if frame_conflict_active and current_frame_idx not in raw_frame_cache:
                enc_ok, enc_buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
                if enc_ok:
                    raw_frame_cache[current_frame_idx] = enc_buf.tobytes()
                    frame_tracks_snapshot[current_frame_idx] = current_frame_tracks

            out_writer.write(annotated_frame)
            processed_count += 1
            current_frame_idx += 1

        cap.release()
        out_writer.release()

        # Group raw conflict occurrences into physical episodes:
        # Track pairs (min_id, max_id) within <= 3.0 seconds belong to the same episode
        episodes: List[List[ConflictCandidate]] = []
        sorted_raw = sorted(conflicts_raw, key=lambda c: c.timestamp_seconds)

        for c in sorted_raw:
            c_pair = tuple(sorted(c.track_ids))
            assigned = False
            for ep in episodes:
                ep_pair = tuple(sorted(ep[0].track_ids))
                if c_pair == ep_pair and abs(c.timestamp_seconds - ep[-1].timestamp_seconds) <= 3.0:
                    ep.append(c)
                    assigned = True
                    break
            if not assigned:
                episodes.append([c])

        # For each episode, select the peak conflict candidate (minimum TTC, tie-break minimum distance)
        deduped_conflicts: List[ConflictCandidate] = []
        for ep in episodes:
            peak = min(ep, key=lambda x: (x.ttc, x.minimum_distance if x.minimum_distance is not None else 999.0))
            deduped_conflicts.append(peak)

        # Sort final candidates chronologically
        deduped_conflicts.sort(key=lambda x: x.timestamp_seconds)

        # Render event-specific evidence image for each candidate
        conflict_snapshots = []
        for candidate in deduped_conflicts:
            f_idx = candidate.frame_index
            frame_bytes = raw_frame_cache.get(f_idx)
            t_snapshot = dict(frame_tracks_snapshot.get(f_idx, {}))

            if frame_bytes is None and raw_frame_cache:
                nearest_idx = min(raw_frame_cache.keys(), key=lambda k: abs(k - f_idx))
                frame_bytes = raw_frame_cache[nearest_idx]
                t_snapshot = dict(frame_tracks_snapshot.get(nearest_idx, {}))
            elif frame_bytes is None:
                blank = np.zeros((height, width, 3), dtype=np.uint8)
                _, b_buf = cv2.imencode('.jpg', blank)
                frame_bytes = b_buf.tobytes()

            track_a = candidate.track_ids[0] if len(candidate.track_ids) > 0 else 1
            track_b = candidate.track_ids[1] if len(candidate.track_ids) > 1 else 2

            # Ensure track_a and track_b are populated from tracks_history if missing from snapshot
            for tid in [track_a, track_b]:
                if tid not in t_snapshot and tid in tracks_history:
                    obj = tracks_history[tid]
                    t_snapshot[tid] = {
                        "cls": obj.object_type,
                        "centroid": obj.centroid,
                        "bbox": obj.bbox,
                        "pixel_contact": obj.pixel_contact,
                        "velocity": obj.velocity,
                        "pixel_history": list(obj.pixel_history),
                        "ground_history": list(obj.ground_history),
                        "frame_history": list(obj.frame_history),
                        "timestamps": list(obj.timestamps),
                        "trajectory": list(obj.pixel_history),
                    }


            snap_url = self._render_conflict_evidence_image(
                base_frame_bytes=frame_bytes,
                file_id=file_id,
                output_dir=output_dir,
                conflict_id=candidate.conflict_id,
                frame_index=candidate.frame_index,
                timestamp_str=candidate.timestamp_str,
                track_id_a=track_a,
                track_id_b=track_b,
                type_a=candidate.object_type_a,
                type_b=candidate.object_type_b,
                ttc=candidate.ttc,
                pet=candidate.pet,
                min_dist=candidate.minimum_distance,
                drac=candidate.drac,
                risk_level=candidate.risk_level,
                interaction_geometry=candidate.interaction_geometry or "Converging Approach",
                tracks_snapshot=t_snapshot,
                apply_privacy=apply_privacy,
                timestamp_seconds=candidate.timestamp_seconds,
            )
            candidate.snapshot_url = snap_url
            conflict_snapshots.append(snap_url)

        # Free cached raw frames to release memory
        raw_frame_cache.clear()
        frame_tracks_snapshot.clear()

        processing_time_sec = round(time.time() - start_time, 2)
        tracker_label = self.tracker_name.split(".")[0].upper()
        model_status = f"{self.model_name} + {tracker_label} Active (Local Edge)" if not is_demo else "Demo AI Mode — traffic model not configured"

        # Compute tracking health metrics across confirmed tracks
        total_tracks = len(tracks_history)
        tracks_with_ge_3_pts = sum(1 for obj in tracks_history.values() if len(obj.pixel_history) >= 3)
        tracks_with_ge_10_pts = sum(1 for obj in tracks_history.values() if len(obj.pixel_history) >= 10)
        single_point_tracks = sum(1 for obj in tracks_history.values() if len(obj.pixel_history) <= 1)
        durations = [
            (obj.timestamps[-1] - obj.timestamps[0])
            for obj in tracks_history.values()
            if len(obj.timestamps) >= 2
        ]
        median_track_duration_sec = float(np.median(durations)) if durations else 0.0
        mean_track_samples = float(np.mean([len(obj.pixel_history) for obj in tracks_history.values()])) if tracks_history else 0.0
        pct_single_point = (single_point_tracks / max(total_tracks, 1)) * 100.0

        if processed_count < 5 or total_tracks == 0:
            tracking_health_status = "INSUFFICIENT_TEMPORAL_DATA"
            tracking_health_msg = "Insufficient video frames or road user tracks to evaluate traffic conflicts."
        elif total_tracks > 5 and (pct_single_point > 80.0 or tracks_with_ge_3_pts < 2):
            tracking_health_status = "TRACKING_FAILED"
            tracking_health_msg = "Tracking pipeline failed to sustain temporal trajectories across frames."
        elif tracks_with_ge_3_pts >= 5 and median_track_duration_sec >= 0.8:
            tracking_health_status = "ANALYSIS_HEALTHY"
            tracking_health_msg = f"Healthy multi-frame tracking: {tracks_with_ge_3_pts} active trajectories sustained (median {median_track_duration_sec:.1f}s)."
        else:
            tracking_health_status = "ANALYSIS_PARTIAL"
            tracking_health_msg = f"Partial trajectory tracking: {tracks_with_ge_3_pts} multi-frame tracks identified."

        tracking_health_summary = {
            "tracking_health_status": tracking_health_status,
            "status_message": tracking_health_msg,
            "total_tracks": total_tracks,
            "multi_point_tracks": tracks_with_ge_3_pts,
            "tracks_gt_10_samples": tracks_with_ge_10_pts,
            "single_point_tracks": single_point_tracks,
            "pct_single_point": round(pct_single_point, 1),
            "median_track_duration_sec": round(median_track_duration_sec, 2),
            "mean_samples_per_track": round(mean_track_samples, 1),
            "metric_calibration_status": calibrator.calibration_status,
            "camera_mode": effective_camera_mode,
            "frames_decoded": total_frames,
            "frames_analyzed": processed_count,
        }

        # Select telemetry tracks: confirmed tracks with multi-point trajectories sorted by sample count & velocity
        active_confirmed_tracks = [
            obj for obj in tracks_history.values() if len(obj.pixel_history) >= 2
        ]
        active_confirmed_tracks.sort(key=lambda t: (len(t.pixel_history), t.velocity), reverse=True)
        display_tracks = active_confirmed_tracks if active_confirmed_tracks else list(tracks_history.values())
        tracks_summary = [
            obj.to_dict() for obj in display_tracks[:15]
        ]

        status_message = (
            f"Analyzed {processed_count} frames across {duration:.1f}s video. "
            f"Tracked {len(tracks_history)} road users via {tracker_label} ({tracking_health_status}). "
            f"Identified {len(deduped_conflicts)} potential conflict candidate(s) (approximate TTC <= {effective_ttc_limit:.1f}s)."
        )

        # Calculate Exposure-Normalized Conflict Rates
        total_veh = sum(object_class_counts.get(k, 0) for k in ["car", "bus", "truck", "motorcycle"])
        total_ped = object_class_counts.get("pedestrian", 0) + object_class_counts.get("bicycle", 0) + object_class_counts.get("rider", 0)
        crit_conf_cnt = sum(1 for c in deduped_conflicts if c.risk_level == "CRITICAL")
        high_conf_cnt = sum(1 for c in deduped_conflicts if c.risk_level == "HIGH")

        exposure_rates = {
            "total_conflicts": len(deduped_conflicts),
            "critical_conflicts": crit_conf_cnt,
            "high_conflicts": high_conf_cnt,
            "vehicles_observed": total_veh,
            "pedestrians_observed": total_ped,
            "critical_conflicts_per_100_vehicles": round((crit_conf_cnt / max(total_veh, 1)) * 100.0, 2),
            "critical_conflicts_per_100_pedestrians": round((crit_conf_cnt / max(total_ped, 1)) * 100.0, 2),
            "severity_weighted_conflict_rate": round(((crit_conf_cnt * 1.0 + high_conf_cnt * 0.6) / max(total_veh, 1)) * 100.0, 2),
        }

        traffic_model_meta = {
            "model_name": self.model_name,
            "model_path": self.model_path,
            "num_classes": len(self.class_map),
            "tracker": self.tracker_name,
            "runtime": self.runtime_mode,
            "camera_calibration": self.camera_mode,
            "classes_tracked": list(self.class_map.values())[:7],
        }

        # Construct interactive event timeline
        timeline: List[Dict[str, Any]] = [
            {
                "timestamp_str": "00:01",
                "seconds": 1.0,
                "status": "Normal Flow",
                "description": f"{len(tracks_history)} road users detected entering corridor tracking zones.",
                "conflict": False,
            }
        ]

        for c in deduped_conflicts:
            timeline.append({
                "timestamp_str": c.timestamp_str,
                "seconds": round(c.timestamp_seconds, 1),
                "status": f"Potential Conflict ({c.event_severity})",
                "description": f"{c.object_type_a.capitalize()} vs {c.object_type_b.capitalize()} (approx. TTC ~{c.ttc:.2f}s, {c.minimum_distance:.1f}m clearance)",
                "conflict": True,
                "conflict_id": c.conflict_id,
                "ttc": round(c.ttc, 2),
            })

        timeline.append({
            "timestamp_str": f"{int(duration // 60):02d}:{int(duration % 60):02d}",
            "seconds": round(duration, 1),
            "status": "Analysis Complete",
            "description": f"Completed trajectory monitoring across {processed_count} frames.",
            "conflict": False,
        })

        return TrafficAnalysisResult(
            video_id=file_id,
            is_demo_mode=is_demo,
            is_demo_video=is_demo_synth,
            model_status=model_status,
            status_message=status_message,
            junction_id=junction_id,
            road_id=road_id,
            duration_seconds=duration,
            total_frames=total_frames,
            processed_frames=processed_count,
            processing_time_sec=processing_time_sec,
            tracked_objects_count=len(tracks_history),
            object_class_counts=object_class_counts,
            near_miss_count=len(deduped_conflicts),
            near_misses=deduped_conflicts,
            timeline=timeline,
            tracked_objects_summary=tracks_summary,
            original_video_url=f"/uploads/traffic/{orig_filename}",
            processed_video_url=f"/uploads/traffic/{proc_filename}" if os.path.exists(proc_path) else None,
            conflict_snapshots=conflict_snapshots,
            privacy_applied=apply_privacy,
            exposure_normalized_rates=exposure_rates,
            traffic_model_meta=traffic_model_meta,
            tracking_health_status=tracking_health_status,
            metric_calibration_status=calibrator.calibration_status,
            tracking_health_summary=tracking_health_summary,
        )
