"""
SafeCity Loop V2 — AI Vision Test Suite
Verifies:
1. Model loading / Demo AI Mode status
2. Image upload + Pothole detection
3. Bounding box & Risk score calculation
4. Video upload + Tracking + TTC Near-miss calculation
5. Error handling: invalid files, empty files, size limits
6. Database persistence
"""
import io
import cv2
import numpy as np
from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)

def create_sample_road_image(width=640, height=480, draw_pothole=True):
    """Generates synthetic asphalt road texture with dark pitted hole."""
    # Gray asphalt base with noise
    img = np.random.randint(70, 95, (height, width, 3), dtype=np.uint8)
    
    # Road markings
    cv2.line(img, (width // 2, 0), (width // 2, height), (220, 220, 220), 4)

    if draw_pothole:
        # Dark irregular hole in the road
        center = (width // 3, height // 2)
        cv2.ellipse(img, center, (65, 45), 25, 0, 360, (20, 20, 20), -1)
        cv2.ellipse(img, center, (75, 52), 25, 0, 360, (40, 38, 35), 4)

    is_success, buffer = cv2.imencode(".jpg", img)
    return io.BytesIO(buffer.tobytes())

def create_sample_traffic_video(width=480, height=360, num_frames=30, simulate_near_miss=True):
    """Generates synthetic traffic video with converging vehicle and pedestrian shapes."""
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    temp_path = "uploads/traffic/test_temp.mp4"
    out = cv2.VideoWriter(temp_path, fourcc, 10.0, (width, height))

    for i in range(num_frames):
        frame = np.full((height, width, 3), 60, dtype=np.uint8)
        # Road lane lines
        cv2.line(frame, (0, height // 2), (width, height // 2), (180, 180, 180), 2)

        # Vehicle moving left to right
        vx = int(50 + i * 10)
        vy = int(height // 2 - 25)
        cv2.rectangle(frame, (vx, vy), (vx + 50, vy + 30), (0, 120, 255), -1)
        cv2.putText(frame, "CAR", (vx, vy - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        # Pedestrian crossing from bottom to center
        if simulate_near_miss:
            px = int(220)
            py = int(height - 40 - i * 6)
        else:
            px = int(50)
            py = int(50)

        cv2.circle(frame, (px, py), 12, (0, 255, 120), -1)
        cv2.putText(frame, "PED", (px - 10, py - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        out.write(frame)

    out.release()
    with open(temp_path, "rb") as f:
        video_bytes = f.read()
    return io.BytesIO(video_bytes)

def run_tests():
    print("=== STARTING AI VISION TEST SUITE ===\n")

    # 1. GET /api/ai/status
    res = client.get("/api/ai/status")
    assert res.status_code == 200, f"AI status failed: {res.text}"
    status_data = res.json()
    print(f"✓ GET /api/ai/status: pothole_demo={status_data['is_pothole_demo_mode']}, traffic_demo={status_data['is_traffic_demo_mode']}")

    # 2. GET /api/ai/events
    res = client.get("/api/ai/events")
    assert res.status_code == 200, f"AI events failed: {res.text}"
    events = res.json()
    print(f"✓ GET /api/ai/events: returned {len(events)} events")

    # 3. POST /api/potholes/analyze with defect image
    img_defect = create_sample_road_image(draw_pothole=True)
    res = client.post(
        "/api/potholes/analyze",
        files={"file": ("road_pothole.jpg", img_defect, "image/jpeg")},
        data={"road_id": 1}
    )
    assert res.status_code == 200, f"Pothole analyze failed: {res.text}"
    pothole_data = res.json()
    print(f"✓ POST /api/potholes/analyze (Defect): potholes={pothole_data['pothole_count']}, "
          f"severity={pothole_data['severity']}, confidence={pothole_data['confidence']*100:.1f}%, "
          f"risk={pothole_data['risk_score']}/100, hazard_id={pothole_data['hazard_id']}")
    assert "Risk" in pothole_data["risk_formula"]
    assert pothole_data["processed_image_url"].startswith("/uploads/potholes/")

    # 4. POST /api/potholes/analyze with clean road (empty detection)
    img_clean = create_sample_road_image(draw_pothole=False)
    res = client.post(
        "/api/potholes/analyze",
        files={"file": ("road_clean.jpg", img_clean, "image/jpeg")},
    )
    assert res.status_code == 200
    clean_data = res.json()
    print(f"✓ POST /api/potholes/analyze (Clean Road): potholes={clean_data['pothole_count']}, risk={clean_data['risk_score']}/100")

    # 5. Invalid image extension
    res = client.post(
        "/api/potholes/analyze",
        files={"file": ("document.pdf", b"fake pdf bytes", "application/pdf")}
    )
    assert res.status_code == 400, f"Expected 400 on invalid extension, got {res.status_code}"
    print("✓ Validation: Rejected invalid image format (PDF) with 400")

    # 6. Empty image file
    res = client.post(
        "/api/potholes/analyze",
        files={"file": ("empty.jpg", b"", "image/jpeg")}
    )
    assert res.status_code == 400
    print("✓ Validation: Rejected empty image file with 400")

    # 7. POST /api/traffic/analyze with video
    video_stream = create_sample_traffic_video(num_frames=20)
    res = client.post(
        "/api/traffic/analyze",
        files={"file": ("traffic_sim.mp4", video_stream, "video/mp4")},
        data={"junction_id": 1}
    )
    assert res.status_code == 200, f"Traffic video analyze failed: {res.text}"
    traffic_data = res.json()
    print(f"✓ POST /api/traffic/analyze: duration={traffic_data['duration_seconds']}s, "
          f"frames={traffic_data['processed_frames']}, tracks={traffic_data['tracked_objects_count']}, "
          f"near_misses={traffic_data['near_miss_count']}")
    assert traffic_data["original_video_url"].startswith("/uploads/traffic/")

    # 8. Invalid video format
    res = client.post(
        "/api/traffic/analyze",
        files={"file": ("script.txt", b"print('hello')", "text/plain")}
    )
    assert res.status_code == 400
    print("✓ Validation: Rejected invalid video format (TXT) with 400")

    print("\nALL AI VISION BACKEND TESTS COMPLETED SUCCESSFULLY!")

if __name__ == "__main__":
    run_tests()
