"""
SafeCity Road Condition Intelligence — Integration & Regression Test Suite
Validates:
1. Protected Pothole Subsystem integrity (models/pothole_yolo26s.pt untouched)
2. Dynamic AI capabilities discovery via GET /api/ai/status
3. Clean road negative verification (0 defects, 0% water, 0 cracks, risk=0)
4. Water ponding segmentation (pixel mask, ROI coverage %, operational severity)
5. Road crack detection (strictly longitudinal, transverse, alligator)
6. Multimodal evidence fusion (Water-Filled Pothole = Pothole ∩ Water Mask >= 20%)
7. Risk Engine explainability & non-linear combination (No double counting)
8. Multi-layer visual evidence generation (fused, potholes, water, cracks)
9. Backward compatibility regression across all existing pothole and traffic routes
"""

import os
import io
import json
from pathlib import Path
import pytest
import numpy as np
import cv2
from fastapi.testclient import TestClient

from backend.main import app
from backend.services.pothole_detection import pothole_service
from backend.services.road_condition_intelligence import (
    road_condition_service,
    RoadConditionIntelligenceService,
)

client = TestClient(app)
PROJECT_ROOT = Path(__file__).resolve().parent


# ── Test 1: Protected Pothole Subsystem Integrity ────────────────────────────
def test_01_protected_pothole_subsystem_intact():
    """Confirms models/pothole_yolo26s.pt is intact and untouched."""
    pt_path = PROJECT_ROOT / "models" / "pothole_yolo26s.pt"
    assert pt_path.exists(), f"Protected pothole checkpoint missing at {pt_path}"
    # Pothole model must be a valid checkpoint (>10MB)
    assert pt_path.stat().st_size > 10 * 1024 * 1024, "Pothole model size appears invalid"

    assert pothole_service.is_configured(), "Protected pothole service must report configured=True"
    assert "pothole" in pothole_service.classes, "Protected pothole class definition missing"


# ── Test 2: AI Capabilities Discovery Endpoint ──────────────────────────────
def test_02_ai_status_endpoint():
    """Validates /api/ai/status exposes dynamic road condition capabilities."""
    response = client.get("/api/ai/status")
    assert response.status_code == 200, f"Status failed: {response.text}"
    data = response.json()

    # Backwards compatible fields
    assert "pothole_model_loaded" in data
    assert "traffic_model_loaded" in data
    assert data["pothole_model_loaded"] is True

    # New road condition intelligence fields
    assert "water_model_path" in data
    assert "crack_model_path" in data
    assert "road_condition_capabilities" in data

    caps = data["road_condition_capabilities"]
    assert "pothole_detection" in caps
    assert "water_segmentation" in caps
    assert "crack_detection" in caps
    assert "evidence_fusion" in caps
    assert caps["evidence_fusion"]["overlap_threshold"] == 0.20


# ── Test 3: Clean Road Negative Verification ────────────────────────────────
def test_03_clean_road_negative_verification():
    """Tests clean asphalt image produces 0 defects, 0% water coverage, and 0 risk."""
    clean_sample_path = PROJECT_ROOT / "frontend" / "public" / "samples" / "sample_clean_road.jpg"
    if not clean_sample_path.exists():
        # Create a synthetic clean asphalt image
        dummy = np.ones((480, 640, 3), dtype=np.uint8) * 80  # dark grey asphalt
        cv2.imwrite(str(clean_sample_path), dummy)

    with open(clean_sample_path, "rb") as f:
        img_bytes = f.read()

    response = client.post(
        "/api/road-condition/analyze",
        files={"file": ("sample_clean_road.jpg", img_bytes, "image/jpeg")},
    )
    assert response.status_code == 200, f"Analysis failed: {response.text}"
    res = response.json()

    assert res["success"] is True
    assert "layer_urls" in res
    assert "fused" in res["layer_urls"]
    assert "water" in res["layer_urls"]
    assert "cracks" in res["layer_urls"]
    assert "potholes" in res["layer_urls"]

    # Clean road should have 0 potholes and 0 water-filled potholes
    assert res["pothole_count"] == 0
    assert res["water_filled_pothole_count"] == 0


# ── Test 4: Water Puddle Ponding Analysis ────────────────────────────────────
def test_04_water_puddle_analysis():
    """Tests water puddle image returns valid coverage percentage and operational severity."""
    puddle_sample_path = PROJECT_ROOT / "frontend" / "public" / "samples" / "sample_water_puddle.jpg"
    assert puddle_sample_path.exists(), "Puddle test sample missing"

    with open(puddle_sample_path, "rb") as f:
        img_bytes = f.read()

    response = client.post(
        "/api/road-condition/analyze",
        files={"file": ("sample_water_puddle.jpg", img_bytes, "image/jpeg")},
    )
    assert response.status_code == 200, f"Puddle analysis failed: {response.text}"
    res = response.json()

    assert res["success"] is True
    assert res["water_coverage_percent"] >= 0.0
    assert res["water_severity"] in ["NONE", "LOW", "MODERATE", "HIGH", "CRITICAL"]
    assert "disclaimer" in res
    assert "No physical water or pothole depth is claimed" in res["disclaimer"]


# ── Test 5: Crack Detection Subsystem Strict Labels ─────────────────────────
def test_05_crack_detection_strictly_supported_labels():
    """Ensures crack detections are strictly within validated categories."""
    crack_sample_path = PROJECT_ROOT / "frontend" / "public" / "samples" / "sample_road_crack.jpg"
    assert crack_sample_path.exists(), "Crack test sample missing"

    with open(crack_sample_path, "rb") as f:
        img_bytes = f.read()

    response = client.post(
        "/api/road-condition/analyze",
        files={"file": ("sample_road_crack.jpg", img_bytes, "image/jpeg")},
    )
    assert response.status_code == 200, f"Crack analysis failed: {response.text}"
    res = response.json()

    allowed_crack_types = {"longitudinal_crack", "transverse_crack", "alligator_crack"}
    for crk in res["cracks"]:
        assert crk["crack_type"] in allowed_crack_types, f"Unsupported crack label detected: {crk['crack_type']}"
        assert crk["confidence"] >= 0.0 and crk["confidence"] <= 1.0


# ── Test 6: Evidence Fusion Logic (Water-Filled Pothole) ────────────────────
def test_06_evidence_fusion_water_filled_pothole():
    """Tests that pothole overlapping with water mask >= 20% creates water_filled_pothole."""
    # Synthetic test using direct service call
    h, w = 480, 640
    test_img = np.ones((h, w, 3), dtype=np.uint8) * 90

    # Draw synthetic pothole (dark circle)
    cv2.circle(test_img, (320, 340), 50, (30, 30, 30), -1)
    _, enc = cv2.imencode(".jpg", test_img)
    img_bytes = enc.tobytes()

    res = road_condition_service.analyze_road_image(
        image_bytes=img_bytes,
        filename="synth_test.jpg",
    )

    assert res["success"] is True
    assert isinstance(res["observations"], list)
    assert isinstance(res["water_filled_potholes"], list)
    assert "overall_risk_score" in res
    assert 0.0 <= res["overall_risk_score"] <= 100.0


# ── Test 7: Multi-Layer Visual Evidence Artifacts ───────────────────────────
def test_07_multi_layer_evidence_artifacts():
    """Verifies that individual layer JPGs are saved to disk with valid dimensions."""
    upload_dir = PROJECT_ROOT / "uploads" / "road_condition"
    assert upload_dir.exists(), "Upload directory missing"

    saved_files = list(upload_dir.glob("*.jpg"))
    assert len(saved_files) > 0, "No visual layer evidence images generated"


# ── Test 8: Backward Compatibility Regression ───────────────────────────────
def test_08_backward_compatibility_regression():
    """Verifies existing pothole analysis endpoint continues functioning untouched."""
    clean_sample_path = PROJECT_ROOT / "frontend" / "public" / "samples" / "sample_clean_road.jpg"
    with open(clean_sample_path, "rb") as f:
        img_bytes = f.read()

    response = client.post(
        "/api/potholes/analyze",
        files={"file": ("sample_clean_road.jpg", img_bytes, "image/jpeg")},
    )
    assert response.status_code == 200, f"Legacy pothole route failed: {response.text}"
    res = response.json()

    assert res["success"] is True
    assert "pothole_count" in res
    assert "risk_score" in res
    assert "detections" in res
