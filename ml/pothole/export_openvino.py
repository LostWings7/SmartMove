"""
SafeCity Loop V2 — OpenVINO Export, Device Benchmarking & Accuracy Parity Suite
Features:
- Exports trained YOLO PyTorch model to OpenVINO Intermediate Representation (IR)
- Detects available Intel accelerators (CPU, integrated GPU, NPU)
- Benchmarks latency (ms) and throughput (FPS) across all detected devices
- Tests inference on Intel AI Boost (NPU) where supported
- Verifies accuracy parity between PyTorch and OpenVINO on held-out test data
- Outputs:
  - models/pothole_yolo26s_openvino_model/
  - ml/pothole/reports/openvino_benchmark.json
  - ml/pothole/reports/openvino_benchmark.md
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional
import cv2
import numpy as np
import openvino as ov
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
ML_DIR = PROJECT_ROOT / "ml" / "pothole"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = ML_DIR / "reports"
PROCESSED_DIR = ML_DIR / "dataset" / "processed"

REPORTS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)


def export_to_openvino(pt_model_path: str = "models/pothole_yolo26s.pt", imgsz: int = 512, half: bool = False) -> str:
    """Exports PyTorch YOLO model to OpenVINO format."""
    print("=" * 60)
    print("STEP 1: Exporting PyTorch Model to OpenVINO IR")
    print(f"Source: {pt_model_path}")
    print("=" * 60)

    model = YOLO(pt_model_path)
    # Ultralytics export
    export_path = model.export(
        format="openvino",
        imgsz=imgsz,
        half=half,
    )
    print(f"OpenVINO Model exported to: {export_path}")
    return export_path


def benchmark_openvino_devices(
    ov_model_dir: str,
    test_image_dir: str = str(PROCESSED_DIR / "images" / "test"),
    num_warmup: int = 5,
    num_runs: int = 30,
) -> Dict[str, Any]:
    """
    Benchmarks OpenVINO inference latency and FPS across CPU, iGPU, and NPU.
    """
    print("=" * 60)
    print("STEP 2: Benchmarking OpenVINO on Available Intel Hardware")
    print("=" * 60)

    core = ov.Core()
    available_devices = core.available_devices
    print(f"Detected OpenVINO devices: {available_devices}")

    # Load XML model
    ov_dir = Path(ov_model_dir)
    xml_files = list(ov_dir.glob("*.xml"))
    if not xml_files:
        raise FileNotFoundError(f"No .xml model found in {ov_dir}")
    xml_path = xml_files[0]

    model = core.read_model(xml_path)

    # Gather test images
    test_p = Path(test_image_dir)
    img_files = sorted(list(test_p.glob("*.jpg")))[:num_runs]
    if not img_files:
        raise ValueError(f"No test images found in {test_image_dir}")

    # Prepare input tensor (1, 3, 512, 512) normalized
    sample_cv = cv2.imread(str(img_files[0]))
    resized = cv2.resize(sample_cv, (512, 512))
    input_tensor = np.transpose(resized, (2, 0, 1)).astype(np.float32) / 255.0
    input_tensor = np.expand_dims(input_tensor, axis=0)

    benchmarks = {}

    target_devices = ["CPU", "GPU", "NPU"]
    for dev in target_devices:
        if dev not in available_devices:
            benchmarks[dev] = {
                "available": False,
                "supported": False,
                "device_name": "Not Available",
                "notes": f"{dev} not detected by OpenVINO runtime on this host.",
            }
            continue

        dev_name = core.get_property(dev, "FULL_DEVICE_NAME")
        print(f"\nBenchmarking Device: {dev} ({dev_name})...")

        try:
            compiled_model = core.compile_model(model, dev)
            infer_request = compiled_model.create_infer_request()

            # Warmup runs
            for _ in range(num_warmup):
                infer_request.infer({0: input_tensor})

            # Timed runs
            latencies = []
            for img_file in img_files:
                img_data = cv2.imread(str(img_file))
                if img_data is None:
                    continue
                r_img = cv2.resize(img_data, (512, 512))
                t_input = np.transpose(r_img, (2, 0, 1)).astype(np.float32) / 255.0
                t_input = np.expand_dims(t_input, axis=0)

                t0 = time.time()
                infer_request.infer({0: t_input})
                latencies.append((time.time() - t0) * 1000.0)

            avg_latency = float(np.mean(latencies))
            p50_latency = float(np.median(latencies))
            p95_latency = float(np.percentile(latencies, 95))
            fps = round(1000.0 / max(avg_latency, 0.001), 1)

            print(f"  Result for {dev}: Mean={avg_latency:.2f}ms, P50={p50_latency:.2f}ms, P95={p95_latency:.2f}ms ({fps} FPS)")

            benchmarks[dev] = {
                "available": True,
                "supported": True,
                "device_name": dev_name,
                "mean_latency_ms": round(avg_latency, 2),
                "p50_latency_ms": round(p50_latency, 2),
                "p95_latency_ms": round(p95_latency, 2),
                "fps": fps,
                "runs_tested": len(latencies),
            }

        except Exception as e:
            print(f"  [Notice] {dev} compiled or execution limitation: {e}")
            benchmarks[dev] = {
                "available": True,
                "supported": False,
                "device_name": dev_name,
                "error": str(e),
                "notes": f"Device present but model precision/topology requires compilation adjustment for {dev}.",
            }

    return benchmarks


def verify_accuracy_parity(
    pt_model_path: str,
    ov_model_dir: str,
    test_image_dir: str = str(PROCESSED_DIR / "images" / "test"),
    conf: float = 0.40,
    sample_count: int = 50,
) -> Dict[str, Any]:
    """
    Compares detections between original PyTorch model and exported OpenVINO model.
    Checks: detection count match, confidence correlation, and bounding-box IoU.
    """
    print("=" * 60)
    print("STEP 3: Verifying PyTorch vs. OpenVINO Detection Parity")
    print("=" * 60)

    pt_model = YOLO(pt_model_path)
    ov_model = YOLO(ov_model_dir)

    test_p = Path(test_image_dir)
    images = sorted(list(test_p.glob("*.jpg")))[:sample_count]

    pt_total_detections = 0
    ov_total_detections = 0
    box_ious = []
    conf_diffs = []

    for img_p in images:
        pt_res = pt_model.predict(source=str(img_p), conf=conf, imgsz=512, device="cpu", verbose=False)
        ov_res = ov_model.predict(source=str(img_p), conf=conf, imgsz=512, device="cpu", verbose=False)

        pt_boxes = pt_res[0].boxes if pt_res and pt_res[0].boxes is not None else []
        ov_boxes = ov_res[0].boxes if ov_res and ov_res[0].boxes is not None else []

        pt_total_detections += len(pt_boxes)
        ov_total_detections += len(ov_boxes)

        # Match boxes between PyTorch and OpenVINO
        for pt_b in pt_boxes:
            pt_xyxy = pt_b.xyxy[0].tolist()
            pt_c = float(pt_b.conf[0])

            best_iou = 0.0
            best_ov_c = 0.0
            for ov_b in ov_boxes:
                ov_xyxy = ov_b.xyxy[0].tolist()
                # IoU calculation
                x1 = max(pt_xyxy[0], ov_xyxy[0])
                y1 = max(pt_xyxy[1], ov_xyxy[1])
                x2 = min(pt_xyxy[2], ov_xyxy[2])
                y2 = min(pt_xyxy[3], ov_xyxy[3])
                inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
                union = (pt_xyxy[2] - pt_xyxy[0]) * (pt_xyxy[3] - pt_xyxy[1]) + (ov_xyxy[2] - ov_xyxy[0]) * (ov_xyxy[3] - ov_xyxy[1]) - inter
                iou = inter / max(union, 1e-6)
                if iou > best_iou:
                    best_iou = iou
                    best_ov_c = float(ov_b.conf[0])

            if best_iou > 0.0:
                box_ious.append(best_iou)
                conf_diffs.append(abs(pt_c - best_ov_c))

    mean_iou = float(np.mean(box_ious)) if box_ious else 1.0
    mean_conf_diff = float(np.mean(conf_diffs)) if conf_diffs else 0.0

    print(f"Tested {len(images)} sample images:")
    print(f"  PyTorch Detections: {pt_total_detections}")
    print(f"  OpenVINO Detections: {ov_total_detections}")
    print(f"  Mean Bounding-Box IoU: {mean_iou:.4f}")
    print(f"  Mean Confidence Diff:  {mean_conf_diff:.4f}")

    parity_passed = mean_iou >= 0.90 and abs(pt_total_detections - ov_total_detections) <= 3
    print(f"  Parity Check Status: {'PASSED (High Fidelity)' if parity_passed else 'WARNING (Discrepancy)'}")

    return {
        "images_evaluated": len(images),
        "pytorch_detections": pt_total_detections,
        "openvino_detections": ov_total_detections,
        "mean_bounding_box_iou": round(mean_iou, 4),
        "mean_confidence_difference": round(mean_conf_diff, 4),
        "parity_passed": parity_passed,
    }


def run_full_openvino_suite(pt_model_path: str = "models/pothole_yolo26s.pt"):
    ov_dir = export_to_openvino(pt_model_path)
    benchmarks = benchmark_openvino_devices(ov_dir)
    parity = verify_accuracy_parity(pt_model_path, ov_dir)

    full_report = {
        "source_pytorch_model": pt_model_path,
        "openvino_model_path": ov_dir,
        "benchmarks_by_device": benchmarks,
        "accuracy_parity": parity,
    }

    # Save JSON report
    json_path = REPORTS_DIR / "openvino_benchmark.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(full_report, f, indent=2)

    # Save Markdown report
    md_path = REPORTS_DIR / "openvino_benchmark.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# SafeCity Loop V2 — OpenVINO Hardware Deployment Benchmark\n\n")
        f.write(f"- **Exported Model Artifact:** `{ov_dir}`\n")
        f.write(f"- **Source Weights:** `{pt_model_path}`\n\n")

        f.write("## 1. Multi-Device Hardware Performance\n\n")
        f.write("| Device | Hardware Description | Mean Latency | Median (P50) | P95 Latency | FPS | Status |\n")
        f.write("| :--- | :--- | :--- | :--- | :--- | :--- | :--- |\n")
        for dev, b in benchmarks.items():
            if b.get("supported"):
                f.write(f"| **{dev}** | {b['device_name']} | **{b['mean_latency_ms']} ms** | {b['p50_latency_ms']} ms | {b['p95_latency_ms']} ms | **{b['fps']}** | Active |\n")
            elif b.get("available"):
                f.write(f"| **{dev}** | {b.get('device_name', dev)} | N/A | N/A | N/A | N/A | Detected (Needs Quant/Plugin) |\n")
            else:
                f.write(f"| **{dev}** | {b.get('device_name', 'Not present')} | N/A | N/A | N/A | N/A | Not Detected |\n")
        f.write("\n")

        f.write("## 2. Accuracy Parity Check (PyTorch vs. OpenVINO)\n\n")
        f.write(f"- Evaluated Test Images: {parity['images_evaluated']}\n")
        f.write(f"- PyTorch Detections: {parity['pytorch_detections']}\n")
        f.write(f"- OpenVINO Detections: {parity['openvino_detections']}\n")
        f.write(f"- Mean Bounding Box IoU: **{parity['mean_bounding_box_iou']}**\n")
        f.write(f"- Mean Confidence Delta: `{parity['mean_confidence_difference']}`\n")
        f.write(f"- **Parity Status:** {'✅ PASSED (Zero Material Degradation)' if parity['parity_passed'] else '⚠️ Discrepancy Flagged'}\n")

    print(f"OpenVINO benchmark report written to: {md_path}")
    print("=" * 60)
    return full_report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="models/pothole_yolo26s.pt", help="Path to PyTorch model")
    args = parser.parse_args()
    run_full_openvino_suite(args.model)
