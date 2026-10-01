"""
SafeCity Road Condition AI — Post-Overhaul Regression & 11-Scenario Real-World Stress Suite
Implements Sections 3, 14, 27, 34, 35, 36:
1. Re-runs ROAD_CONDITION_REGRESSION_001 .. 005 with the upgraded pipeline:
   - Saves post-overhaul visual layers to ml/road_condition_reports/regression_001/post_overhaul_outputs/
   - Saves post_overhaul_regression_001_report.json
2. Evaluates all 11 required real-world stress test scenarios (Section 35):
   1. road with large cracks only
   2. road with small cracks only
   3. road with potholes only
   4. road with cracks + potholes
   5. road with puddle only
   6. road with water-filled pothole
   7. road with trees/plants/buildings/sky
   8. road with shadows
   9. road with manholes/drains
   10. normal clean road
   11. non-road image
"""

import os
import sys
import json
import shutil
from pathlib import Path
import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.services.road_condition_intelligence import road_condition_service

REPORTS_DIR = PROJECT_ROOT / "ml" / "road_condition_reports"
REG_DIR = REPORTS_DIR / "regression_001"
POST_OUT_DIR = REG_DIR / "post_overhaul_outputs"
POST_OUT_DIR.mkdir(parents=True, exist_ok=True)
STRESS_OUT_DIR = REPORTS_DIR / "stress_11_outputs"
STRESS_OUT_DIR.mkdir(parents=True, exist_ok=True)


def copy_layer_outputs(res: dict, prefix: str, target_dir: Path):
    for layer_name, rel_url in res.get("layer_urls", {}).items():
        src_p = PROJECT_ROOT / rel_url.lstrip("/")
        if src_p.exists():
            shutil.copy2(src_p, target_dir / f"{prefix}_{layer_name}.jpg")
    orig_p = PROJECT_ROOT / res.get("original_image_url", "").lstrip("/")
    if orig_p.exists():
        shutil.copy2(orig_p, target_dir / f"{prefix}_orig.jpg")


def main():
    print("=" * 72)
    print("RUNNING POST-OVERHAUL REGRESSION SUITE (001 .. 005)")
    print("=" * 72)

    reg_files = [
        ("ROAD_CONDITION_REGRESSION_001", REG_DIR / "ROAD_CONDITION_REGRESSION_001.jpg"),
        ("ROAD_CONDITION_REGRESSION_002_forest_cracks", REG_DIR / "ROAD_CONDITION_REGRESSION_002_forest_cracks.jpg"),
        ("ROAD_CONDITION_REGRESSION_003_alligator_closeup", REG_DIR / "ROAD_CONDITION_REGRESSION_003_alligator_closeup.jpg"),
        ("ROAD_CONDITION_REGRESSION_004_overhead_vehicle", REG_DIR / "ROAD_CONDITION_REGRESSION_004_overhead_vehicle.jpg"),
        ("ROAD_CONDITION_REGRESSION_005_water_pothole", REG_DIR / "ROAD_CONDITION_REGRESSION_005_water_pothole.jpg"),
    ]

    post_regression_results = {}
    for tag, img_p in reg_files:
        img_bytes = img_p.read_bytes()
        res = road_condition_service.analyze_road_image(img_bytes, filename=img_p.name)
        copy_layer_outputs(res, tag, POST_OUT_DIR)
        post_regression_results[tag] = {
            "pothole_count": res["pothole_count"],
            "crack_count": res["crack_count"],
            "longitudinal_crack_count": res["longitudinal_crack_count"],
            "transverse_crack_count": res["transverse_crack_count"],
            "alligator_crack_count": res["alligator_crack_count"],
            "water_regions_count": res["water_regions_count"],
            "water_coverage_percent": res["water_coverage_percent"],
            "water_filled_pothole_count": res["water_filled_pothole_count"],
            "overall_risk_score": res["overall_risk_score"],
            "highest_severity": res["highest_severity"],
            "potholes": res["potholes"],
            "cracks": res["cracks"],
            "water_regions": res["water_regions"],
            "water_filled_potholes": res["water_filled_potholes"],
            "disambiguation_audit": res.get("disambiguation_audit", []),
            "road_context_summary": res.get("road_context_summary", {}),
            "processing_time_ms": res["processing_time_ms"],
        }
        print(
            f"  {tag:48s} -> Potholes={res['pothole_count']} | Cracks={res['crack_count']} "
            f"(L={res['longitudinal_crack_count']}, T={res['transverse_crack_count']}, A={res['alligator_crack_count']}) | "
            f"Water={res['water_regions_count']} ({res['water_coverage_percent']}%) | WF_Pothole={res['water_filled_pothole_count']}"
        )

    post_reg_json = REG_DIR / "post_overhaul_regression_001_report.json"
    with open(post_reg_json, "w", encoding="utf-8") as f:
        json.dump(post_regression_results, f, indent=2)
    print(f"Saved post-overhaul regression report to: {post_reg_json}")

    print("\n" + "=" * 72)
    print("RUNNING 11 REAL-WORLD STRESS TEST SCENARIOS (SECTION 35)")
    print("=" * 72)

    stress_dir = PROJECT_ROOT / "ml" / "pothole" / "dataset" / "stress_test"
    crack_test_dir = PROJECT_ROOT / "ml" / "cracks" / "dataset" / "processed_v2" / "images" / "test"
    pothole_test_dir = PROJECT_ROOT / "ml" / "pothole" / "dataset" / "processed_v3" / "images" / "test"

    def first_img(folder: Path, pattern: str = "*.jpg") -> Path:
        files = sorted(list(folder.glob(pattern)))
        return files[0] if files else (REG_DIR / "ROAD_CONDITION_REGRESSION_001.jpg")

    scenarios_11 = [
        ("01_large_cracks_only", REG_DIR / "ROAD_CONDITION_REGRESSION_001.jpg"),
        ("02_small_cracks_only", first_img(crack_test_dir, "*Japan*.jpg")),
        ("03_potholes_only", PROJECT_ROOT / "frontend" / "public" / "samples" / "sample_pothole_road.jpg"),
        ("04_cracks_plus_potholes", first_img(pothole_test_dir, "*India*.jpg")),
        ("05_puddle_only", PROJECT_ROOT / "frontend" / "public" / "samples" / "sample_water_puddle.jpg"),
        ("06_water_filled_pothole", REG_DIR / "ROAD_CONDITION_REGRESSION_005_water_pothole.jpg"),
        ("07_road_with_trees_plants_sky", REG_DIR / "ROAD_CONDITION_REGRESSION_002_forest_cracks.jpg"),
        ("08_road_with_shadows", first_img(stress_dir / "shadows")),
        ("09_road_with_manholes_drains", first_img(stress_dir / "manholes")),
        ("10_normal_clean_road", PROJECT_ROOT / "frontend" / "public" / "samples" / "sample_clean_road.jpg"),
        ("11_non_road_image", first_img(stress_dir / "non_road")),
    ]

    stress_11_results = {}
    for s_name, s_path in scenarios_11:
        if not s_path.exists():
            continue
        res = road_condition_service.analyze_road_image(s_path.read_bytes(), filename=s_path.name)
        copy_layer_outputs(res, s_name, STRESS_OUT_DIR)
        stress_11_results[s_name] = {
            "source_image": str(s_path.relative_to(PROJECT_ROOT)),
            "pothole_count": res["pothole_count"],
            "crack_count": res["crack_count"],
            "water_regions_count": res["water_regions_count"],
            "water_coverage_percent": res["water_coverage_percent"],
            "water_filled_pothole_count": res["water_filled_pothole_count"],
            "overall_risk_score": res["overall_risk_score"],
            "highest_severity": res["highest_severity"],
            "road_context_summary": res.get("road_context_summary", {}),
            "disambiguation_audit": res.get("disambiguation_audit", []),
        }
        print(
            f"  {s_name:32s} -> Potholes={res['pothole_count']} | Cracks={res['crack_count']} | "
            f"Water={res['water_regions_count']} ({res['water_coverage_percent']}%) | WF_Pothole={res['water_filled_pothole_count']}"
        )

    stress_json = REPORTS_DIR / "stress_test_11_scenarios.json"
    with open(stress_json, "w", encoding="utf-8") as f:
        json.dump(stress_11_results, f, indent=2)
    print(f"Saved 11-scenario stress test report to: {stress_json}")


if __name__ == "__main__":
    main()
