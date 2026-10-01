"""
SafeCity Loop V2 — Pothole Detector Integration Test Suite
Validates:
1. Model files & deployment metadata existence (models/pothole_yolo26s.pt, models/pothole_model.json, OpenVINO IR)
2. AI Vision API status endpoint (/api/ai/status) metadata integrity
3. Live inference on real road test images from held-out dataset
4. Segmentation-ready dataclass fields (mask_polygon, area_sq_meters, shape_type)
5. OpenCV fallback resilience and clean road verification
6. Precomputed demonstration backwards compatibility
7. Multi-severity classification & Risk Engine integration
"""

import os
import json
from pathlib import Path
import pytest
from fastapi.testclient import TestClient
from backend.main import app
from backend.services.pothole_detection import PotholeDetectionService, pothole_service

client = TestClient(app)
PROJECT_ROOT = Path(__file__).resolve().parent


def test_01_model_files_and_metadata_exist():
    """Validates that trained model weights and metadata are present."""
    pt_path = PROJECT_ROOT / "models" / "pothole_yolo26s.pt"
    ov_dir = PROJECT_ROOT / "models" / "pothole_yolo26s_openvino_model"
    json_path = PROJECT_ROOT / "models" / "pothole_model.json"

    assert pt_path.exists(), f"PyTorch model not found at {pt_path}"
    assert pt_path.stat().st_size > 10 * 1024 * 1024, "PyTorch model checkpoint appears incomplete"

    assert ov_dir.exists(), f"OpenVINO IR directory not found at {ov_dir}"
    xml_files = list(ov_dir.glob("*.xml"))
    bin_files = list(ov_dir.glob("*.bin"))
    assert len(xml_files) > 0, "No .xml found in OpenVINO IR directory"
    assert len(bin_files) > 0, "No .bin found in OpenVINO IR directory"

    assert json_path.exists(), f"Model metadata not found at {json_path}"
    with open(json_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    assert meta["model_name"] == "SafeCity YOLO26s Pothole Detector"
    assert "test_metrics" in meta
    assert meta["test_metrics"]["map50"] > 0.20
    assert "hardware_benchmarks" in meta
    assert "Intel" in meta["hardware_benchmarks"]["host_cpu"]["name"]
    print("\n[TEST 1 PASSED] Model weights, OpenVINO IR, and metadata JSON validated.")


def test_02_ai_status_endpoint():
    """Validates GET /api/ai/status returns enriched YOLO26s metadata."""
    res = client.get("/api/ai/status")
    assert res.status_code == 200
    data = res.json()

    assert data["pothole_model_loaded"] is True
    assert "pothole_yolo26s" in data["pothole_model_path"]
    assert data["is_pothole_demo_mode"] is False
    assert data["model_name"] == "SafeCity YOLO26s Pothole Detector"
    assert data["model_version"] == "1.0.0"
    assert "classes" in data
    assert "pothole" in data["classes"]
    assert data["confidence_threshold"] >= 0.15
    print(f"\n[TEST 2 PASSED] /api/ai/status verified: {data['message']}")


def test_03_live_inference_on_real_dataset_image():
    """Tests real neural YOLO inference using an actual image from the held-out test split."""
    test_img_dir = PROJECT_ROOT / "ml" / "pothole" / "dataset" / "processed" / "images" / "test"
    test_images = list(test_img_dir.glob("*.jpg"))
    assert len(test_images) > 0, "No test images found"

    sample_img_path = test_images[0]
    with open(sample_img_path, "rb") as f:
        img_bytes = f.read()

    res = client.post(
        "/api/potholes/analyze",
        files={"file": (sample_img_path.name, img_bytes, "image/jpeg")},
        data={"road_id": 1, "latitude": 19.0760, "longitude": 72.8777},
    )
    assert res.status_code == 200, f"Inference failed: {res.text}"
    data = res.json()

    assert data["success"] is True
    assert data["is_demo_mode"] is False
    assert data["evidence_id"].startswith("SC-H-")
    assert "visual_severity" in data
    assert "contextual_severity" in data
    assert 0 <= data["risk_score"] <= 100
    assert data["latitude"] == pytest.approx(19.0760, abs=0.001)

    # Check segmentation-ready fields on detection items
    for det in data["detections"]:
        assert det["class_name"] == "pothole"
        assert "shape_type" in det
        assert det["shape_type"] in ["bounding_box", "polygon"]
        assert "area_sq_meters" in det
        assert det["area_sq_meters"] is not None
        assert "box" in det
        assert det["box"]["width"] > 0
        assert det["box"]["height"] > 0

    print(f"\n[TEST 3 PASSED] Real image inference verified: {len(data['detections'])} defect(s), Risk: {data['risk_score']}")


def test_04_clean_road_surface_behavior():
    """Verifies that clean road surface is correctly identified with 0 defects."""
    # Create clean grey road image
    import cv2
    import numpy as np

    img = np.full((480, 640, 3), 110, dtype=np.uint8)
    # Add lane markings
    cv2.line(img, (320, 0), (320, 480), (255, 255, 255), 4)
    _, buf = cv2.imencode(".jpg", img)
    clean_bytes = buf.tobytes()

    res = client.post(
        "/api/potholes/analyze",
        files={"file": ("clean_road_surface.jpg", clean_bytes, "image/jpeg")},
        data={"road_id": 1},
    )
    assert res.status_code == 200
    data = res.json()
    assert data["pothole_count"] == 0
    assert data["risk_score"] == 0.0
    print("\n[TEST 4 PASSED] Clean road surface correctly reported 0 defects.")


def test_05_demo_sample_backward_compatibility():
    """Verifies that demo sample requests maintain deterministic benchmark behavior."""
    import numpy as np
    import cv2

    img = np.full((480, 640, 3), 80, dtype=np.uint8)
    _, buf = cv2.imencode(".jpg", img)

    res = client.post(
        "/api/potholes/analyze",
        files={"file": ("sample_pothole_demo.jpg", buf.tobytes(), "image/jpeg")},
        data={"road_id": 1, "is_demo_sample": True},
    )
    assert res.status_code == 200
    data = res.json()
    assert data["is_precomputed_demo"] is True
    assert data["pothole_count"] == 2
    assert data["detections"][0]["is_demo"] is True
    print("\n[TEST 5 PASSED] Demo sample backward compatibility verified.")


def test_06_openvino_runtime_inference():
    """Verifies that OpenVINO model can be directly loaded and executed via Ultralytics."""
    ov_model_dir = PROJECT_ROOT / "models" / "pothole_yolo26s_openvino_model"
    from ultralytics import YOLO

    ov_model = YOLO(str(ov_model_dir), task="detect")
    test_img = PROJECT_ROOT / "ml" / "pothole" / "dataset" / "processed" / "images" / "test"
    sample = list(test_img.glob("*.jpg"))[0]

    results = ov_model.predict(source=str(sample), imgsz=512, conf=0.25, verbose=False)
    assert results is not None
    assert len(results) > 0
    print(f"\n[TEST 6 PASSED] Direct OpenVINO IR model inference validated on {sample.name}.")


if __name__ == "__main__":
    test_01_model_files_and_metadata_exist()
    test_02_ai_status_endpoint()
    test_03_live_inference_on_real_dataset_image()
    test_04_clean_road_surface_behavior()
    test_05_demo_sample_backward_compatibility()
    test_06_openvino_runtime_inference()
    print("\n" + "=" * 60)
    print("ALL POTHOLE DETECTOR INTEGRATION TESTS PASSED (6/6)!")
    print("=" * 60)
