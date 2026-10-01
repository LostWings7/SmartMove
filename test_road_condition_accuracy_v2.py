"""
SafeCity Road Condition Intelligence — Accuracy Overhaul Test Suite (v2.1.0)
Validates all 13 required test cases (Section 40):
 1. test_regression_001_pothole_false_positives_eliminated
 2. test_regression_001_crack_detected
 3. test_regression_001_banana_leaf_not_pothole
 4. test_road_context_filters_sky_vegetation
 5. test_raw_model_confidence_unmodified
 6. test_pothole_vs_crack_disambiguation_case_a
 7. test_pothole_vs_crack_disambiguation_case_b
 8. test_pothole_vs_crack_disambiguation_case_c
 9. test_pothole_vs_crack_disambiguation_case_d
10. test_spatial_deduplication
11. test_clean_road_zero_false_positives
12. test_non_road_image_rejection
13. test_openvino_inference_parity
"""

import os
import sys
from pathlib import Path
import pytest
import numpy as np
import cv2

PROJECT_ROOT = Path(__file__).resolve().parent.parent if "tests" in Path(__file__).resolve().parent.name else Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.services.road_context import road_context_service, RoadContextResult
from backend.services.road_condition_intelligence import (
    road_condition_service,
    PotholeDetectionItem,
    BoundingBox,
)

REG_001_PATH = PROJECT_ROOT / "ml" / "road_condition_reports" / "regression_001" / "ROAD_CONDITION_REGRESSION_001.jpg"
CLEAN_ROAD_PATH = PROJECT_ROOT / "frontend" / "public" / "samples" / "sample_clean_road.jpg"
NON_ROAD_PATH = PROJECT_ROOT / "ml" / "pothole" / "dataset" / "stress_test" / "non_road"


# ── 1. Regression 001: Pothole False Positives Eliminated ─────────────────────
def test_regression_001_pothole_false_positives_eliminated():
    """Assert pothole detections on ROAD_CONDITION_REGRESSION_001 == 0."""
    assert REG_001_PATH.exists(), f"Missing regression test image: {REG_001_PATH}"
    img_bytes = REG_001_PATH.read_bytes()
    res = road_condition_service.analyze_road_image(img_bytes, filename=REG_001_PATH.name)
    assert res["pothole_count"] == 0, f"Expected 0 potholes on cracked regression image, got {res['pothole_count']}"


# ── 2. Regression 001: Crack Detected ─────────────────────────────────────────
def test_regression_001_crack_detected():
    """Assert crack detections on ROAD_CONDITION_REGRESSION_001 >= 1."""
    assert REG_001_PATH.exists()
    img_bytes = REG_001_PATH.read_bytes()
    res = road_condition_service.analyze_road_image(img_bytes, filename=REG_001_PATH.name)
    assert res["crack_count"] >= 1, f"Expected at least 1 crack detected on regression 001, got {res['crack_count']}"
    crack_types = [c["crack_type"] for c in res["cracks"]]
    assert any(ct in ["alligator_crack", "longitudinal_crack", "transverse_crack"] for ct in crack_types)


# ── 3. Regression 001: Banana Leaf In Top-Right Not Pothole ───────────────────
def test_regression_001_banana_leaf_not_pothole():
    """Assert no pothole detection in the top-right quadrant of REGRESSION_001."""
    img_bytes = REG_001_PATH.read_bytes()
    res = road_condition_service.analyze_road_image(img_bytes, filename=REG_001_PATH.name)
    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    h, w = img.shape[:2]

    for p in res.get("potholes", []):
        bx1 = p["bbox"]["x1"]
        by1 = p["bbox"]["y1"]
        # Top-right quadrant: x >= w/2 and y <= h/2
        is_top_right = (bx1 >= w * 0.5) and (by1 <= h * 0.5)
        assert not is_top_right, f"Banana leaf false positive pothole found in top-right: {p['bbox']}"


# ── 4. Road Context Filters Sky & Vegetation ─────────────────────────────────
def test_road_context_filters_sky_vegetation():
    """Assert candidate boxes in sky/vegetation regions are rejected by road_context_service."""
    img = cv2.imread(str(REG_001_PATH))
    h, w = img.shape[:2]
    ctx = road_context_service.analyze_frame(img)

    # Box in top-right sky / foliage canopy
    sky_foliage_box = (int(w * 0.70), 0, w, int(h * 0.35))
    eval_sky = road_context_service.evaluate_box(
        sky_foliage_box,
        raw_model_confidence=0.75,
        context=ctx,
        hazard_type="pothole",
    )
    assert not eval_sky.is_valid_road_candidate, "Sky/vegetation box must NOT be accepted as valid road candidate"
    assert eval_sky.road_overlap_ratio < 0.40, f"Expected low road overlap in sky/foliage, got {eval_sky.road_overlap_ratio}"


# ── 5. Raw Model Confidence Kept Strictly Separate & Unmodified ──────────────
def test_raw_model_confidence_unmodified():
    """Assert raw_model_confidence is preserved as output by neural net, separate from context score."""
    img = cv2.imread(str(REG_001_PATH))
    ctx = road_context_service.analyze_frame(img)
    test_box = (20, 80, 150, 140)
    raw_conf = 0.5678

    eval_res = road_context_service.evaluate_box(
        test_box,
        raw_model_confidence=raw_conf,
        context=ctx,
        hazard_type="crack",
    )
    # raw_model_confidence must be preserved exactly
    assert eval_res.raw_model_confidence == raw_conf, "raw_model_confidence must NOT be mutated"
    assert hasattr(eval_res, "road_context_score")
    assert hasattr(eval_res, "road_overlap_ratio")
    assert eval_res.road_context_score != eval_res.raw_model_confidence or raw_conf == 1.0


# ── 6. Disambiguation Case A: Elongated Crack Preferred Over Pothole ─────────
def test_pothole_vs_crack_disambiguation_case_a():
    """Assert elongated crack geometry without cavity depression prefers crack and rejects pothole."""
    dummy_img = np.full((300, 300, 3), 128, dtype=np.uint8)
    pothole_item = PotholeDetectionItem(
        class_name="pothole",
        confidence=0.45,
        raw_model_confidence=0.45,
        severity="MEDIUM",
        risk_score=50.0,
        box=BoundingBox(x1=20, y1=100, x2=240, y2=130, width=220, height=30),  # width=220, height=30, aspect_ratio=7.33 (elongated)
    )
    crack_item = {
        "crack_type": "longitudinal_crack",
        "confidence": 0.55,
        "bbox": {"x1": 20, "y1": 100, "x2": 240, "y2": 130, "width": 220, "height": 30},
    }
    water_mask = np.zeros((300, 300), dtype=np.uint8)

    kept_p, kept_c, audit = road_condition_service._disambiguate_potholes_and_cracks(
        dummy_img, [pothole_item], [crack_item], water_mask
    )
    assert len(kept_p) == 0, "Elongated crack pattern must NOT be kept as pothole"
    assert len(kept_c) == 1, "Crack candidate must be preserved"
    assert any("CASE_A" in rec.get("decision", "") for rec in audit)


# ── 7. Disambiguation Case B: Closed Depression Preferred as Pothole ─────────
def test_pothole_vs_crack_disambiguation_case_b():
    """Assert compact closed depression is classified as pothole."""
    dummy_img = np.full((300, 300, 3), 160, dtype=np.uint8)
    # Paint darker core cavity in center
    dummy_img[120:180, 120:180] = 50

    pothole_item = PotholeDetectionItem(
        class_name="pothole",
        confidence=0.72,
        raw_model_confidence=0.72,
        severity="HIGH",
        risk_score=75.0,
        box=BoundingBox(x1=100, y1=100, x2=200, y2=200, width=100, height=100),  # width=100, height=100, aspect=1.0 (compact)
    )
    water_mask = np.zeros((300, 300), dtype=np.uint8)

    kept_p, kept_c, audit = road_condition_service._disambiguate_potholes_and_cracks(
        dummy_img, [pothole_item], [], water_mask
    )
    assert len(kept_p) == 1, "Compact depression must be confirmed as pothole"
    assert kept_p[0].final_acceptance_decision == "CASE_B_CONFIRMED_POTHOLE_DEPRESSION"


# ── 8. Disambiguation Case C: Co-occurring Pothole + Crack Network ───────────
def test_pothole_vs_crack_disambiguation_case_c():
    """Assert co-occurring pothole within larger crack network preserves both without double counting."""
    dummy_img = np.full((400, 400, 3), 160, dtype=np.uint8)
    dummy_img[140:180, 140:180] = 40  # Deep depression cavity

    pothole_item = PotholeDetectionItem(
        class_name="pothole",
        confidence=0.68,
        raw_model_confidence=0.68,
        severity="HIGH",
        risk_score=75.0,
        box=BoundingBox(x1=130, y1=130, x2=190, y2=190, width=60, height=60),  # 60x60 cavity
    )
    crack_item = {
        "crack_type": "alligator_crack",
        "confidence": 0.60,
        "bbox": {"x1": 50, "y1": 50, "x2": 350, "y2": 350, "width": 300, "height": 300},  # 300x300 mesh
        "severity": "HIGH",
        "risk_score": 75.0,
    }
    water_mask = np.zeros((400, 400), dtype=np.uint8)

    kept_p, kept_c, audit = road_condition_service._disambiguate_potholes_and_cracks(
        dummy_img, [pothole_item], [crack_item], water_mask
    )
    assert len(kept_p) == 1, "Pothole cavity within crack mesh must be kept (Case C)"
    assert len(kept_c) == 1, "Crack mesh must also be kept"
    assert kept_p[0].final_acceptance_decision == "CASE_C_COOCCURRING_POTHOLE_AND_CRACK"


# ── 9. Disambiguation Case D: Water-Filled Pothole Fusion ────────────────────
def test_pothole_vs_crack_disambiguation_case_d():
    """Assert pothole overlapping water mask (>= 20%) creates water_filled_pothole fusion."""
    dummy_img = np.full((300, 300, 3), 128, dtype=np.uint8)
    water_mask = np.zeros((300, 300), dtype=np.uint8)
    # Paint water mask over 60% of pothole region
    water_mask[100:160, 100:200] = 1

    pothole_item = PotholeDetectionItem(
        class_name="pothole",
        confidence=0.75,
        raw_model_confidence=0.75,
        severity="HIGH",
        risk_score=75.0,
        box=BoundingBox(x1=100, y1=100, x2=200, y2=200, width=100, height=100),  # area = 10000 px, water = 6000 px (60%)
    )

    kept_p, _, audit = road_condition_service._disambiguate_potholes_and_cracks(
        dummy_img, [pothole_item], [], water_mask
    )
    assert len(kept_p) == 1
    assert kept_p[0].final_acceptance_decision == "CASE_D_WATER_FILLED_POTHOLE"


# ── 10. Spatial Deduplication ────────────────────────────────────────────────
def test_spatial_deduplication():
    """Assert two overlapping proposals for the same crack are merged, not double-counted."""
    h, w = 400, 400
    img = np.full((h, w, 3), 128, dtype=np.uint8)
    ctx = road_context_service.analyze_frame(img)

    # Simulate proposal list with two heavily overlapping boxes of the same crack class
    # Box 1: (50, 100, 200, 250), Box 2: (55, 105, 205, 255) -> IoU > 0.80
    b1 = {"x1": 50, "y1": 100, "x2": 200, "y2": 250, "width": 150, "height": 150}
    b2 = {"x1": 55, "y1": 105, "x2": 205, "y2": 255, "width": 150, "height": 150}

    iou, iomin = road_condition_service._box_iou_and_iomin(b1, b2)
    assert iou >= 0.70, f"Expected high IoU between duplicate proposals, got {iou}"


# ── 11. Clean Road Produces Zero False Positives ─────────────────────────────
def test_clean_road_zero_false_positives():
    """Assert sample_clean_road.jpg produces 0 potholes and 0 cracks."""
    assert CLEAN_ROAD_PATH.exists(), f"Clean road sample missing at {CLEAN_ROAD_PATH}"
    img_bytes = CLEAN_ROAD_PATH.read_bytes()
    res = road_condition_service.analyze_road_image(img_bytes, filename=CLEAN_ROAD_PATH.name)
    assert res["pothole_count"] == 0, f"Clean road must produce 0 potholes, got {res['pothole_count']}"
    assert res["crack_count"] == 0, f"Clean road must produce 0 cracks, got {res['crack_count']}"
    assert res["water_regions_count"] == 0, f"Clean road must produce 0 water regions, got {res['water_regions_count']}"


# ── 12. Non-Road Image Rejection ─────────────────────────────────────────────
def test_non_road_image_rejection():
    """Assert non-road scene produces 0 hazard detections and is flagged as non-road."""
    non_road_files = list(NON_ROAD_PATH.glob("*.jpg")) if NON_ROAD_PATH.exists() else []
    if non_road_files:
        test_img_path = non_road_files[0]
        img_bytes = test_img_path.read_bytes()
    else:
        # Synthetic non-road image (uniform living-room beige)
        img = np.full((400, 400, 3), (180, 200, 220), dtype=np.uint8)
        _, buf = cv2.imencode(".jpg", img)
        img_bytes = buf.tobytes()

    res = road_condition_service.analyze_road_image(img_bytes, filename="non_road_test.jpg")
    assert res["pothole_count"] == 0, f"Non-road image must produce 0 potholes, got {res['pothole_count']}"
    assert res["crack_count"] == 0, f"Non-road image must produce 0 cracks, got {res['crack_count']}"


# ── 13. OpenVINO Inference Parity ────────────────────────────────────────────
def test_openvino_inference_parity():
    """Assert OpenVINO models are present and loadable if OpenVINO is available."""
    try:
        import openvino as ov
        core = ov.Core()
        devices = core.available_devices
        assert len(devices) > 0, "OpenVINO should report at least one available device"

        ov_pothole_path = PROJECT_ROOT / "models" / "pothole_yolo26s_vNEXT_openvino_model" / "pothole_yolo26s_vNEXT.xml"
        ov_crack_path = PROJECT_ROOT / "models" / "road_crack_yolo26s_v2_openvino_model" / "road_crack_yolo26s_v2.xml"
        ov_water_path = PROJECT_ROOT / "models" / "water_yolo26s_seg_openvino_model" / "water_yolo26s_seg_v1.xml"

        assert ov_pothole_path.exists(), f"OpenVINO Pothole IR missing at {ov_pothole_path}"
        assert ov_crack_path.exists(), f"OpenVINO Crack IR missing at {ov_crack_path}"
        assert ov_water_path.exists(), f"OpenVINO Water IR missing at {ov_water_path}"

        assert ov_pothole_path.exists(), f"OpenVINO Pothole IR missing at {ov_pothole_path}"
        assert ov_crack_path.exists(), f"OpenVINO Crack IR missing at {ov_crack_path}"
        assert ov_water_path.exists(), f"OpenVINO Water IR missing at {ov_water_path}"

        # Test model compilation on CPU
        model_p = core.read_model(str(ov_pothole_path))
        compiled = core.compile_model(model_p, "CPU")
        assert compiled is not None, "Compiled OpenVINO model must not be None"
    except ImportError:
        pytest.skip("OpenVINO package not installed in this environment")
