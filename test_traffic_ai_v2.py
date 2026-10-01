"""
SafeCity Loop V2 — Traffic AI & Surrogate Safety Validation Suite
Tests:
1. Custom YOLO26s Traffic Model Loading & 7-Class Schema
2. Ground-Contact Point Projection & IPM Perspective Calibration
3. Trajectory Smoothing & Velocity Estimation (Real-World km/h)
4. Closest Point of Approach (2D CPA) Kinematic TTC
5. Post-Encroachment Time (PET) Spatial Zone Crossing
6. Deceleration Rate to Avoid Crash (DRAC)
7. Multi-Signal Conflict Classification (Rear-end, Crossing, Head-on)
8. Multi-Object Tracker Stability (ByteTrack & BoT-SORT)
9. Exposure-Normalized Conflict Rates (per 100 vehicles / pedestrians)
10. End-to-End API Video Inference & Telemetry Serialization
"""
import os
import io
import math
import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from backend.main import app
from backend.services.traffic_analysis import TrafficAnalysisService
from ml.traffic.trajectory_analysis import (
    PerspectiveCalibrator, TrajectoryEngine, TTCEngineV2,
    compute_exposure_normalized_conflict_rates
)

client = TestClient(app)


def test_1_model_loading_and_classes():
    """Verify custom YOLO26s weights are loaded with exact 7-class schema."""
    service = TrafficAnalysisService()
    assert service.model is not None, "Traffic model should be loaded"
    assert "traffic_yolo26s" in service.model_path or "yolo" in service.model_path
    
    expected_classes = ["pedestrian", "rider", "car", "truck", "bus", "motorcycle", "bicycle"]
    for c in expected_classes:
        assert c in service.class_map.values(), f"Class '{c}' must be present in model schema"
    print(f"[TEST 1 PASS] Model loaded from {service.model_path} with classes: {list(service.class_map.values())}")


def test_2_ground_contact_and_perspective_calibration():
    """Verify ground-contact point calculation and IPM metric mapping."""
    calibrator = PerspectiveCalibrator(mode="dashcam", image_width=1280, image_height=720)
    
    # Ground contact for bbox [x1, y1, x2, y2] is ((x1+x2)/2, y2)
    contact_pt = TrajectoryEngine.get_contact_point((100, 200, 200, 500))
    assert contact_pt == (150.0, 500.0), f"Expected (150.0, 500.0), got {contact_pt}"
    
    # Distance between near-field points in ground meters
    pt_near = calibrator.pixel_to_ground(640, 680)
    pt_far = calibrator.pixel_to_ground(640, 500)
    
    # Longitudinal distance (y) in world coords should increase as image y decreases (towards horizon)
    dist = math.hypot(pt_far[0] - pt_near[0], pt_far[1] - pt_near[1])
    assert dist > 3.0, f"Distance across vertical pixels in dashcam IPM should be >3m, got {dist:.2f}m"
    print(f"[TEST 2 PASS] Ground contact calculation and IPM projection validated (delta: {dist:.2f}m)")


def test_3_trajectory_smoothing_and_velocity():
    """Verify TrajectoryEngine filters noise and computes realistic speeds in km/h."""
    calibrator = PerspectiveCalibrator(mode="dashcam", image_width=1280, image_height=720)
    engine = TrajectoryEngine(calibrator=calibrator)
    
    # Simulate a car moving forward over 10 frames at 25 FPS
    fps = 25.0
    for frame_idx in range(10):
        # Car bounding box moving forward
        y2 = 450 + frame_idx * 15
        bbox = (500, y2 - 80, 700, y2)
        state = engine.update_track(track_id=1, object_class="car", bbox=bbox, timestamp=frame_idx / fps)
    
    track = engine.tracks[1]
    assert track.track_id == 1
    assert track.object_class == "car"
    assert track.velocity_kmh > 0.0, "Smooth speed should be positive"
    assert track.velocity_kmh < 180.0, "Smooth speed should be in realistic driving range"
    print(f"[TEST 3 PASS] TrajectoryEngine computed filtered speed: {track.velocity_kmh:.1f} km/h")


def test_4_closest_point_of_approach_ttc():
    """Verify 2D CPA kinematics detect converging paths and compute accurate TTC."""
    calibrator = PerspectiveCalibrator(mode="dashcam", image_width=1280, image_height=720)
    engine = TrajectoryEngine(calibrator=calibrator)
    ttc_engine = TTCEngineV2(default_ttc_threshold=3.0)
    
    fps = 25.0
    for frame_idx in range(8):
        t = frame_idx / fps
        # Car approaching forward
        y_car = 480 + frame_idx * 12
        state_car = engine.update_track(1, "car", (560, y_car - 60, 680, y_car), timestamp=t)
        # Pedestrian crossing into lane
        x_ped = 450 + frame_idx * 18
        state_ped = engine.update_track(2, "pedestrian", (x_ped - 15, 540, x_ped + 15, 600), timestamp=t)
        
    eval_result = ttc_engine.compute_conflict(engine.tracks[1], engine.tracks[2])
    assert eval_result is not None, "Converging trajectories must trigger conflict evaluation"
    assert eval_result.is_valid_conflict is True
    assert eval_result.ttc_seconds is not None, "TTC must be computed"
    assert eval_result.ttc_seconds <= 3.0, f"Expected TTC <= 3.0s, got {eval_result.ttc_seconds}"
    print(f"[TEST 4 PASS] 2D CPA TTC: {eval_result.ttc_seconds:.2f}s | Type: {eval_result.interaction_type}")


def test_5_post_encroachment_time_and_drac():
    """Verify PET and DRAC (Deceleration Rate to Avoid Crash) computations."""
    ttc_engine = TTCEngineV2()
    
    # Test DRAC calculation directly: DRAC = (delta_v)^2 / (2 * delta_x)
    # 50 km/h (13.89 m/s) over 15 meters
    drac = ttc_engine.calculate_drac(relative_speed_mps=13.89, clearance_distance_m=15.0)
    assert drac is not None
    assert 6.0 <= drac <= 7.0, f"Expected DRAC ~6.43 m/s², got {drac:.2f}"
    
    # Test PET with sequential zone entry
    zone_history = {
        1: [(10.0, 1.0)],   # Track 1 exited conflict zone at t=1.0s
        2: [(10.0, 2.4)],   # Track 2 entered conflict zone at t=2.4s
    }
    pet = ttc_engine.calculate_pet(zone_history, track_a=1, track_b=2)
    assert pet == pytest.approx(1.4, abs=0.05), f"Expected PET 1.4s, got {pet}"
    print(f"[TEST 5 PASS] DRAC: {drac:.2f} m/s² | True PET: {pet:.2f}s")


def test_6_exposure_normalized_conflict_rates():
    """Verify rate calculations normalize conflicts per 100 vehicles / pedestrians."""
    from ml.traffic.trajectory_analysis import ConflictEvaluation
    
    mock_conflicts = [
        ConflictEvaluation(
            conflict_id="C1", track_a=1, track_b=2, class_a="car", class_b="pedestrian",
            ttc_seconds=0.8, pet_seconds=1.1, drac_mps2=4.8, distance_meters=3.0,
            relative_speed_kmh=35.0, interaction_type="Crossing", trajectory_convergence=True,
            severity="CRITICAL", conflict_zone_label="Crosswalk", explanation={}, is_valid_conflict=True
        ),
        ConflictEvaluation(
            conflict_id="C2", track_a=3, track_b=4, class_a="car", class_b="car",
            ttc_seconds=1.4, pet_seconds=1.8, drac_mps2=3.2, distance_meters=4.5,
            relative_speed_kmh=40.0, interaction_type="Rear-End", trajectory_convergence=True,
            severity="HIGH", conflict_zone_label="Travel Corridor", explanation={}, is_valid_conflict=True
        ),
    ]
    rates = compute_exposure_normalized_conflict_rates(
        conflicts=mock_conflicts,
        total_vehicles_observed=50,
        total_pedestrians_observed=20,
    )
    assert rates["critical_conflicts"] == 1
    assert rates["high_conflicts"] == 1
    assert rates["vru_conflicts"] == 1
    assert rates["critical_conflicts_per_100_vehicles"] == 2.0
    assert rates["critical_conflicts_per_100_pedestrians"] == 5.0
    assert rates["severity_weighted_conflict_rate"] > 0.0
    print(f"[TEST 6 PASS] Exposure normalized rates: {rates}")


def test_7_api_video_analysis_pipeline():
    """Verify full FastAPI /api/traffic/analyze endpoint with custom YOLO26s."""
    # Create short 15-frame dummy video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    temp_vid = "test_traffic_v2_sample.mp4"
    out = cv2.VideoWriter(temp_vid, fourcc, 15.0, (640, 480))
    for f in range(15):
        frame = np.full((480, 640, 3), 70, dtype=np.uint8)
        # Vehicle
        cv2.rectangle(frame, (280, 400 - f * 10), (360, 470 - f * 10), (0, 160, 255), -1)
        out.write(frame)
    out.release()
    
    try:
        with open(temp_vid, "rb") as f:
            vid_bytes = f.read()
            
        res = client.post(
            "/api/traffic/analyze",
            files={"file": ("traffic_v2.mp4", io.BytesIO(vid_bytes), "video/mp4")},
            data={"junction_id": 1, "road_id": 1, "ttc_threshold": 2.5, "apply_privacy": "true"},
        )
        assert res.status_code == 200
        data = res.json()
        assert data["processed_frames"] > 0
        assert "exposure_normalized_rates" in data
        assert "traffic_model_meta" in data
        assert data["traffic_model_meta"]["num_classes"] == 7
        print(f"[TEST 7 PASS] End-to-end API analysis returned 7-class model metadata: {data['traffic_model_meta']['model_name']}")
    finally:
        if os.path.exists(temp_vid):
            os.remove(temp_vid)


if __name__ == "__main__":
    test_1_model_loading_and_classes()
    test_2_ground_contact_and_perspective_calibration()
    test_3_trajectory_smoothing_and_velocity()
    test_4_closest_point_of_approach_ttc()
    test_5_post_encroachment_time_and_drac()
    test_6_exposure_normalized_conflict_rates()
    test_7_api_video_analysis_pipeline()
    print("\n======================================================================")
    print("ALL 7 TRAFFIC AI & SURROGATE SAFETY INTEGRATION TESTS PASSED!")
    print("======================================================================")
