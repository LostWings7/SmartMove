"""
SafeCity Loop V2 — Video Inference & Surrogate Safety Telemetry Pipeline
Renders clean, readable technical annotations:
- Bounding boxes and class labels (pedestrian, rider, car, truck, bus, motorcycle, bicycle)
- Persistent track IDs
- Trajectory trails
- Real-time estimated velocity (km/h) and direction
- Collision vectors, TTC, PET, and Conflict Severity indicators
"""
import os
import sys
import time
import argparse
from typing import Dict, List, Tuple, Optional
import cv2
import numpy as np
from ultralytics import YOLO

# Import SafeCity Trajectory and Calibration Engine
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from ml.traffic.trajectory_analysis import (
    TrajectoryEngine, PerspectiveCalibrator, TTCEngineV2, ConflictEvaluation
)

CLASS_COLORS = {
    "pedestrian": (0, 255, 100),       # Mint Green
    "rider": (0, 220, 255),            # Cyan / Yellow
    "bicycle": (0, 240, 200),          # Aqua
    "motorcycle": (0, 165, 255),       # Orange
    "car": (255, 120, 0),              # Blue / Azure
    "bus": (255, 80, 0),               # Deep Blue
    "truck": (200, 80, 0),             # Navy Blue
}


def run_video_inference(
    video_path: str,
    model_path: str = "models/traffic_yolo26s_v1.pt",
    output_path: Optional[str] = None,
    conf_threshold: float = 0.25,
    tracker_config: str = "bytetrack.yaml",
    camera_mode: str = "dashcam",
    ttc_threshold: float = 2.0,
    max_frames: Optional[int] = None,
    show_preview: bool = False,
):
    print("=" * 70)
    print("SAFECITY LOOP V2 — VIDEO INFERENCE & SURROGATE SAFETY PIPELINE")
    print(f"Video: {video_path} | Model: {model_path} | Tracker: {tracker_config}")
    print("=" * 70)

    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Input video not found: {video_path}")
    if not os.path.exists(model_path):
        if os.path.exists("yolo26s.pt"):
            print(f"[Warning] Custom model {model_path} not found, using baseline yolo26s.pt")
            model_path = "yolo26s.pt"
        else:
            raise FileNotFoundError(f"Model weights not found: {model_path}")

    model = YOLO(model_path)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if output_path is None:
        base, ext = os.path.splitext(video_path)
        output_path = f"{base}_annotated.mp4"

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Initialize Trajectory & Surrogate Conflict Engines
    calibrator = PerspectiveCalibrator(mode=camera_mode, image_width=width, image_height=height)
    traj_engine = TrajectoryEngine(calibrator=calibrator)
    ttc_engine = TTCEngineV2(default_ttc_threshold=ttc_threshold)

    frame_idx = 0
    t_start = time.time()
    total_conflicts_detected = 0

    while True:
        ret, frame = cap.read()
        if not ret or (max_frames and frame_idx >= max_frames):
            break

        timestamp = frame_idx / max(fps, 1.0)
        annotated_frame = frame.copy()

        # Run tracking
        results = model.track(
            frame,
            persist=True,
            conf=conf_threshold,
            tracker=tracker_config,
            verbose=False,
        )

        active_states = []
        if results and results[0].boxes is not None and results[0].boxes.id is not None:
            boxes = results[0].boxes
            ids = boxes.id.int().cpu().tolist()
            classes = boxes.cls.int().cpu().tolist()
            xyxys = boxes.xyxy.cpu().tolist()

            for tid, cid, bbox in zip(ids, classes, xyxys):
                cname = model.names.get(cid, "vehicle").lower()
                state = traj_engine.update_track(
                    track_id=tid,
                    object_class=cname,
                    bbox=tuple(bbox),
                    timestamp=timestamp,
                )
                active_states.append(state)

                # Render compact bounding box
                color = CLASS_COLORS.get(cname, (255, 120, 0))
                x1, y1, x2, y2 = [int(v) for v in bbox]
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)

                # Compact technical label
                speed_str = f" | {state.velocity_kmh:.0f} km/h" if state.velocity_kmh > 1.0 else ""
                label = f"{cname.upper()} #{tid}{speed_str}"
                
                # Small background tag
                (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)
                cv2.rectangle(annotated_frame, (x1, max(y1 - 16, 0)), (x1 + lw + 6, max(y1, 16)), (20, 24, 33), -1)
                cv2.putText(annotated_frame, label, (x1 + 3, max(y1 - 4, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1)

                # Trajectory movement trail
                if len(state.history_ground) >= 2:
                    # Draw pixel contact trail
                    trail_pts = []
                    for t_idx in range(max(0, len(state.history_timestamps) - 10), len(state.history_timestamps)):
                        # Simple screen space trail
                        trail_pts.append(state.pixel_contact)
                    if len(trail_pts) > 1:
                        pts_arr = np.array(trail_pts, dtype=np.int32).reshape((-1, 1, 2))
                        cv2.polylines(annotated_frame, [pts_arr], False, color, 1)

        # Evaluate surrogate conflicts pairwise
        active_conflicts: List[ConflictEvaluation] = []
        n_active = len(active_states)
        for i in range(n_active):
            for j in range(i + 1, n_active):
                conf_eval = ttc_engine.compute_conflict(active_states[i], active_states[j])
                if conf_eval:
                    active_conflicts.append(conf_eval)
                    total_conflicts_detected += 1

        # Draw conflict indicators & collision vectors
        for c in active_conflicts:
            sa = traj_engine.tracks[c.track_a]
            sb = traj_engine.tracks[c.track_b]
            pa = (int(sa.pixel_contact[0]), int(sa.pixel_contact[1]))
            pb = (int(sb.pixel_contact[0]), int(sb.pixel_contact[1]))

            # Red collision line
            cv2.line(annotated_frame, pa, pb, (0, 0, 255), 2)

            # Conflict Banner
            mid_x = (pa[0] + pb[0]) // 2
            mid_y = (pa[1] + pb[1]) // 2
            ttc_disp = f"TTC {c.ttc_seconds:.2f}s" if c.ttc_seconds else "CONVERGING"
            c_tag = f"CONFLICT: {ttc_disp} | {c.severity}"
            (tw, th), _ = cv2.getTextSize(c_tag, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 2)
            cv2.rectangle(annotated_frame, (max(mid_x - tw // 2 - 4, 10), max(mid_y - 18, 10)),
                          (max(mid_x + tw // 2 + 4, tw + 10), max(mid_y + 4, 28)), (0, 0, 180), -1)
            cv2.putText(annotated_frame, c_tag, (max(mid_x - tw // 2, 12), max(mid_y - 4, 22)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 2)

        # Upper HUD telemetry banner
        mins = int(timestamp // 60)
        secs = int(timestamp % 60)
        hud_left = f"SafeCity Loop V2 | Time: {mins:02d}:{secs:02d} | Frame: {frame_idx} | Tracks: {len(active_states)}"
        hud_right = f"TTC Limit: {ttc_threshold:.1f}s | Active Conflicts: {len(active_conflicts)}"
        cv2.rectangle(annotated_frame, (0, 0), (width, 24), (15, 23, 42), -1)
        cv2.putText(annotated_frame, hud_left, (8, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (240, 240, 240), 1)
        cv2.putText(annotated_frame, hud_right, (width - 320, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 220, 255), 1)

        out_writer.write(annotated_frame)
        frame_idx += 1

        if frame_idx % 50 == 0:
            print(f"      Processed {frame_idx}/{total_frames} frames ({frame_idx/max(time.time()-t_start, 0.001):.1f} FPS)...")

    cap.release()
    out_writer.release()
    elapsed = time.time() - t_start

    print(f"\n[SUCCESS] Video inference completed:")
    print(f"          Processed Frames: {frame_idx} in {elapsed:.1f}s ({frame_idx/max(elapsed, 0.001):.1f} FPS)")
    print(f"          Output Saved: {output_path} ({os.path.getsize(output_path):,} bytes)")
    print(f"          Total Conflicts Flagged: {total_conflicts_detected}")
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, required=True, help="Input video file path")
    parser.add_argument("--model", type=str, default="models/traffic_yolo26s_v1.pt", help="Model weights path")
    parser.add_argument("--output", type=str, default=None, help="Output annotated video path")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--tracker", type=str, default="bytetrack.yaml", help="Tracker config")
    parser.add_argument("--ttc", type=float, default=2.0, help="TTC threshold in seconds")
    parser.add_argument("--max-frames", type=int, default=None, help="Max frames to process")
    args = parser.parse_args()

    run_video_inference(
        video_path=args.video,
        model_path=args.model,
        output_path=args.output,
        conf_threshold=args.conf,
        tracker_config=args.tracker,
        ttc_threshold=args.ttc,
        max_frames=args.max_frames,
    )
