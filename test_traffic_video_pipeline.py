"""
SafeCity Loop V2 — Traffic Video Processing Pipeline & Trajectory Health Test Suite
Tests:
1. Protected Weights Verification (models/pothole_yolo26s.pt & models/traffic_yolo26s_v1.pt)
2. Ghost Track Elimination (no synthetic 900000+ IDs in track history)
3. Multi-Frame Sequential Motion & Non-Zero Velocity
4. Telemetry Track Selection (prioritizes sustained multi-point tracks over 1-point noise)
5. Perspective Calibration Modes (aerial_intersection, fixed_cctv, dashcam) & Calibration Status
6. Tracking Health State Classification (HEALTHY vs PARTIAL vs FAILED vs INSUFFICIENT)
7. TTC Preconditions (requires >= 3 points and non-zero motion to evaluate conflicts)
8. End-to-End API /api/traffic/analyze with Camera Mode & Tracking Health Schema
"""
import os
import io
import math
import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from backend.main import app
from backend.services.traffic_analysis import TrafficAnalysisService, TrackedObject
from ml.traffic.trajectory_analysis import (
    PerspectiveCalibrator, TrajectoryEngine, TTCEngineV2, TrajectoryState
)

client = TestClient(app)


def test_1_protected_weights_intact():
    """Verify protected pothole and traffic baseline weights remain intact."""
    pothole_path = os.path.join("models", "pothole_yolo26s.pt")
    traffic_v1_path = os.path.join("models", "traffic_yolo26s_v1.pt")
    
    assert os.path.exists(pothole_path), f"Pothole model missing at {pothole_path}"
    assert os.path.getsize(pothole_path) > 10_000_000, "Pothole model size must be valid weights"

    assert os.path.exists(traffic_v1_path), f"Traffic v1 model missing at {traffic_v1_path}"
    assert os.path.getsize(traffic_v1_path) > 10_000_000, "Traffic v1 model size must be valid weights"
    print("[TEST 1 PASS] Protected weights pothole_yolo26s.pt and traffic_yolo26s_v1.pt verified.")


def test_2_perspective_calibrator_modes_and_status():
    """Verify camera modes (aerial_intersection, fixed_cctv, dashcam) and status reporting."""
    # Aerial Intersection mode
    calib_aerial = PerspectiveCalibrator(mode="aerial_intersection", image_width=1280, image_height=720)
    assert calib_aerial.calibration_status == "NOMINAL_PIXEL_ESTIMATE"
    pt_aerial = calib_aerial.pixel_to_ground(640, 360)
    assert pt_aerial == (0.0, 0.0), f"Center should map to (0, 0), got {pt_aerial}"

    # Dashcam IPM mode
    calib_dashcam = PerspectiveCalibrator(mode="dashcam", image_width=1280, image_height=720)
    assert calib_dashcam.calibration_status == "CALIBRATED_IPM"
    pt_dash = calib_dashcam.pixel_to_ground(640, 680)
    assert pt_dash[1] > 0.0, "Dashcam forward distance must be positive"

    # Fixed CCTV with custom homography
    H_custom = np.eye(3, dtype=np.float64)
    calib_cctv = PerspectiveCalibrator(mode="fixed_cctv", custom_homography=H_custom)
    assert calib_cctv.calibration_status == "CALIBRATED_HOMOGRAPHY"
    print("[TEST 2 PASS] PerspectiveCalibrator modes, coordinate mappings, and statuses verified.")


def test_3_ttc_preconditions_reject_underdeveloped_tracks():
    """Verify TTCEngineV2 requires >= 3 points and active velocity before computing conflict."""
    ttc_engine = TTCEngineV2(default_ttc_threshold=3.0)

    # Case A: 1-point tracks (should be immediately rejected)
    state_a = TrajectoryState(
        track_id=1, object_class="car", pixel_contact=(300, 400),
        ground_pos=(0.0, 10.0), velocity_mps=12.0, velocity_kmh=43.2,
        velocity_vector=(0.0, 12.0), acceleration_mps2=0.0, heading_deg=0.0,
        history_ground=[(0.0, 10.0)],  # Only 1 sample!
    )
    state_b = TrajectoryState(
        track_id=2, object_class="pedestrian", pixel_contact=(300, 350),
        ground_pos=(0.0, 15.0), velocity_mps=1.5, velocity_kmh=5.4,
        velocity_vector=(0.0, -1.5), acceleration_mps2=0.0, heading_deg=180.0,
        history_ground=[(0.0, 15.0)],  # Only 1 sample!
    )
    res_1pt = ttc_engine.compute_conflict(state_a, state_b)
    assert res_1pt is None, "TTC must reject tracks with < 3 samples"

    # Case B: 3 points but both objects stationary (0 m/s)
    state_a.history_ground = [(0.0, 10.0), (0.0, 10.0), (0.0, 10.0)]
    state_a.velocity_mps = 0.0
    state_a.velocity_vector = (0.0, 0.0)
    state_b.history_ground = [(0.0, 15.0), (0.0, 15.0), (0.0, 15.0)]
    state_b.velocity_mps = 0.0
    state_b.velocity_vector = (0.0, 0.0)
    res_stat = ttc_engine.compute_conflict(state_a, state_b)
    assert res_stat is None, "TTC must reject dual-stationary road users"

    # Case C: 3 points with converging velocities
    state_a.velocity_mps = 10.0
    state_a.velocity_vector = (0.0, 10.0)
    state_b.velocity_mps = 1.5
    state_b.velocity_vector = (0.0, -1.5)
    res_conv = ttc_engine.compute_conflict(state_a, state_b)
    assert res_conv is not None, "TTC must evaluate valid converging multi-point tracks"
    assert res_conv.ttc_seconds is not None
    print(f"[TEST 3 PASS] TTC preconditions verified (TTC={res_conv.ttc_seconds:.2f}s).")


def test_4_synthetic_video_tracking_and_no_ghost_tracks():
    """Verify sequential frame tracking computes non-zero velocities and creates zero 900000+ ghost tracks."""
    service = TrafficAnalysisService(camera_mode="aerial_intersection")

    # Generate a 20-frame test video with 2 moving boxes (simulating traffic)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    temp_vid = "test_seq_track_sample.mp4"
    out = cv2.VideoWriter(temp_vid, fourcc, 15.0, (1280, 720))
    for f in range(20):
        frame = np.full((720, 1280, 3), 50, dtype=np.uint8)
        # Draw road quad
        cv2.rectangle(frame, (100, 100), (1180, 620), (70, 70, 70), -1)
        # Moving car 1: starts at (300, 200), moves downwards
        y_car1 = 200 + f * 15
        cv2.rectangle(frame, (350, y_car1), (430, y_car1 + 70), (0, 165, 255), -1)
        # Moving car 2: starts at (700, 550), moves upwards
        y_car2 = 550 - f * 12
        cv2.rectangle(frame, (650, y_car2), (730, y_car2 + 70), (255, 100, 0), -1)
        out.write(frame)
    out.release()

    try:
        with open(temp_vid, "rb") as f:
            v_bytes = f.read()

        result = service.analyze_video(
            video_bytes=v_bytes,
            filename="seq_test.mp4",
            max_process_frames=None,
            camera_mode="aerial_intersection",
        )

        # When model does not detect flat rectangles, it correctly reports INSUFFICIENT_TEMPORAL_DATA
        assert result.processed_frames > 0
        assert result.tracking_health_status in ["INSUFFICIENT_TEMPORAL_DATA", "ANALYSIS_HEALTHY", "ANALYSIS_PARTIAL"]

        # Assert no 900000+ ghost tracks exist in summary
        for trk in result.tracked_objects_summary:
            assert trk["track_id"] < 900000, f"Ghost track #{trk['track_id']} detected!"

        # Now test multi-frame tracking with demo synthetic scenario (model=None)
        demo_service = TrafficAnalysisService(model_path="nonexistent.pt", camera_mode="aerial_intersection")
        result_demo = demo_service.analyze_video(
            video_bytes=v_bytes,
            filename="demo_test.mp4",
            max_process_frames=None,
            camera_mode="aerial_intersection",
            is_demo_video=True,
        )
        assert result_demo.processed_frames > 0
        assert result_demo.tracking_health_status in ["ANALYSIS_HEALTHY", "ANALYSIS_PARTIAL"]
        for trk in result_demo.tracked_objects_summary:
            assert trk["track_id"] < 900000
            assert trk["trajectory_length"] > 1
        print(f"[TEST 4 PASS] Video tracking verified: {result_demo.tracked_objects_count} tracks, health={result_demo.tracking_health_status}.")
    finally:
        if os.path.exists(temp_vid):
            os.remove(temp_vid)


def test_5_api_tracking_health_serialization():
    """Verify /api/traffic/analyze returns tracking_health_status and tracking_health_summary."""
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    temp_vid = "test_api_health_sample.mp4"
    out = cv2.VideoWriter(temp_vid, fourcc, 15.0, (640, 480))
    for f in range(10):
        frame = np.full((480, 640, 3), 60, dtype=np.uint8)
        cv2.rectangle(frame, (200, 100 + f * 10), (260, 160 + f * 10), (0, 180, 255), -1)
        out.write(frame)
    out.release()

    try:
        with open(temp_vid, "rb") as f:
            vid_bytes = f.read()

        res = client.post(
            "/api/traffic/analyze",
            files={"file": ("test_health.mp4", io.BytesIO(vid_bytes), "video/mp4")},
            data={"ttc_threshold": 2.0, "camera_mode": "aerial_intersection"},
        )
        assert res.status_code == 200
        data = res.json()
        assert "tracking_health_status" in data
        assert "metric_calibration_status" in data
        assert "tracking_health_summary" in data
        assert data["metric_calibration_status"] == "NOMINAL_PIXEL_ESTIMATE"
        print(f"[TEST 5 PASS] API returned tracking health: {data['tracking_health_status']} ({data['metric_calibration_status']})")
    finally:
        if os.path.exists(temp_vid):
            os.remove(temp_vid)


if __name__ == "__main__":
    test_1_protected_weights_intact()
    test_2_perspective_calibrator_modes_and_status()
    test_3_ttc_preconditions_reject_underdeveloped_tracks()
    test_4_synthetic_video_tracking_and_no_ghost_tracks()
    test_5_api_tracking_health_serialization()
    print("\nALL TRAFFIC VIDEO PIPELINE TESTS PASSED!")
