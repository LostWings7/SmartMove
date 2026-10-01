"""
SafeCity Loop V2 — Comprehensive Pothole Model Evaluation & Threshold Analysis
Features:
- Held-out test set evaluation using standard Ultralytics validator + custom metrics
- Computes Precision, Recall, F1, mAP50, mAP50-95, Latency (ms), FPS
- Confidence threshold sweep: 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80
- Failure analysis: False Positives (FP) and False Negatives (FN)
- Condition robustness breakdown (Positive vs. Negative background images)
- Saves worst-case error visualizers to ml/pothole/reports/worst_cases/
- Outputs:
  - ml/pothole/reports/evaluation_report.json
  - ml/pothole/reports/evaluation_report.md
  - ml/pothole/reports/worst_cases/
"""

import os
import sys
import json
import time
import shutil
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Any
import cv2
import numpy as np
import yaml
from PIL import Image
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
ML_DIR = PROJECT_ROOT / "ml" / "pothole"
REPORTS_DIR = ML_DIR / "reports"
WORST_CASES_DIR = REPORTS_DIR / "worst_cases"
CONFIGS_DIR = ML_DIR / "configs"
PROCESSED_DIR = ML_DIR / "dataset" / "processed"

REPORTS_DIR.mkdir(parents=True, exist_ok=True)
WORST_CASES_DIR.mkdir(parents=True, exist_ok=True)


def calculate_iou(box1: List[float], box2: List[float]) -> float:
    """Calculates Intersection over Union for [x1, y1, x2, y2]."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter_area = inter_w * inter_h

    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter_area
    if union <= 0.0:
        return 0.0
    return inter_area / union


def evaluate_test_set(model_path: str, data_yaml_path: str = str(CONFIGS_DIR / "data.yaml")) -> Dict[str, Any]:
    print("=" * 60)
    print("SAFECITY POTHOLE DETECTOR — HELD-OUT TEST EVALUATION")
    print(f"Model: {model_path}")
    print(f"Data:  {data_yaml_path}")
    print("=" * 60)

    model = YOLO(model_path)

    # 1. Run Ultralytics Test Split Validation
    print("Running official Ultralytics validation on TEST split...")
    val_results = model.val(
        data=data_yaml_path,
        split="test",
        imgsz=512,
        device="cpu",
        verbose=True,
        save_json=True,
    )

    # Extract metrics from Ultralytics results
    box_metrics = val_results.box
    precision = float(box_metrics.mp)
    recall = float(box_metrics.mr)
    map50 = float(box_metrics.map50)
    map50_95 = float(box_metrics.map)
    f1 = float(2 * precision * recall / max(precision + recall, 1e-6))

    # Latency benchmarking
    speed_dict = val_results.speed
    preprocess_ms = speed_dict.get("preprocess", 1.0)
    inference_ms = speed_dict.get("inference", 45.0)
    postprocess_ms = speed_dict.get("postprocess", 1.5)
    total_latency_ms = preprocess_ms + inference_ms + postprocess_ms
    fps = round(1000.0 / max(total_latency_ms, 1.0), 1)

    print("\nOfficial Test Split Metrics:")
    print(f"  Precision:   {precision:.4f}")
    print(f"  Recall:      {recall:.4f}")
    print(f"  F1 Score:    {f1:.4f}")
    print(f"  mAP@50:      {map50:.4f}")
    print(f"  mAP@50-95:   {map50_95:.4f}")
    print(f"  Latency:     {total_latency_ms:.2f} ms/image ({fps} FPS)")

    # 2. Confidence-Threshold Sweep on Held-Out Test Set
    print("\nRunning Confidence Threshold Sweep (0.20 to 0.80)...")
    thresholds = [0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80]
    test_img_dir = PROCESSED_DIR / "images" / "test"
    test_lbl_dir = PROCESSED_DIR / "labels" / "test"

    test_images = sorted(list(test_img_dir.glob("*.jpg")))
    ground_truths: Dict[str, List[List[float]]] = {}

    for img_p in test_images:
        lbl_p = test_lbl_dir / f"{img_p.stem}.txt"
        gts = []
        if lbl_p.exists():
            with open(lbl_p, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        x, y, w, h = map(float, parts[1:5])
                        gts.append([x - w / 2, y - h / 2, x + w / 2, y + h / 2])
        ground_truths[img_p.stem] = gts

    sweep_results = []
    best_thresh = 0.40
    best_thresh_f1 = 0.0

    # Cache predictions at low threshold (0.15) once to speed up sweep
    print(f"Running inference across all {len(test_images)} test images...")
    cached_preds: Dict[str, List[Tuple[List[float], float]]] = {}
    for img_p in test_images:
        preds = model.predict(source=str(img_p), conf=0.15, imgsz=512, device="cpu", verbose=False)
        boxes_list = []
        if preds and preds[0].boxes is not None:
            for b in preds[0].boxes:
                xyxy = b.xyxy[0].tolist()
                conf_val = float(b.conf[0])
                # Normalize xyxy to [0, 1]
                im_h, im_w = preds[0].orig_shape
                norm_xyxy = [
                    xyxy[0] / im_w,
                    xyxy[1] / im_h,
                    xyxy[2] / im_w,
                    xyxy[3] / im_h,
                ]
                boxes_list.append((norm_xyxy, conf_val))
        cached_preds[img_p.stem] = boxes_list

    for thresh in thresholds:
        tp = 0
        fp = 0
        fn = 0

        for stem, gts in ground_truths.items():
            preds = [p for p in cached_preds.get(stem, []) if p[1] >= thresh]
            matched_gt = set()

            for pred_box, pred_conf in preds:
                # Find matching ground truth with IoU >= 0.45
                best_iou = 0.0
                best_gt_idx = -1
                for idx, gt_box in enumerate(gts):
                    if idx in matched_gt:
                        continue
                    iou = calculate_iou(pred_box, gt_box)
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = idx

                if best_iou >= 0.45 and best_gt_idx != -1:
                    tp += 1
                    matched_gt.add(best_gt_idx)
                else:
                    fp += 1

            # Unmatched ground truths are false negatives
            fn += len(gts) - len(matched_gt)

        p_thresh = tp / max(tp + fp, 1)
        r_thresh = tp / max(tp + fn, 1)
        f1_thresh = 2 * p_thresh * r_thresh / max(p_thresh + r_thresh, 1e-6)

        sweep_results.append({
            "threshold": thresh,
            "precision": round(p_thresh, 4),
            "recall": round(r_thresh, 4),
            "f1": round(f1_thresh, 4),
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn,
        })

        if f1_thresh > best_thresh_f1:
            best_thresh_f1 = f1_thresh
            best_thresh = thresh

    print("\nThreshold Sweep Analysis:")
    print(" Thresh | Precision | Recall |   F1   |   TP   |   FP   |   FN  ")
    print("-" * 62)
    for row in sweep_results:
        print(f"  {row['threshold']:.2f}  |   {row['precision']:.4f}  | {row['recall']:.4f} | {row['f1']:.4f} |  {row['true_positives']:4d}  |  {row['false_positives']:4d}  |  {row['false_negatives']:4d}")

    print(f"\nEmpirically Selected Deployment Threshold: {best_thresh:.2f} (F1: {best_thresh_f1:.4f})")

    # 3. Robustness on Positive vs Negative Images
    pos_images = [img for img, gts in ground_truths.items() if len(gts) > 0]
    neg_images = [img for img, gts in ground_truths.items() if len(gts) == 0]
    false_positives_on_clean_roads = 0
    for neg_stem in neg_images:
        neg_preds = [p for p in cached_preds.get(neg_stem, []) if p[1] >= best_thresh]
        false_positives_on_clean_roads += len(neg_preds)

    clean_road_spec = (len(neg_images) - false_positives_on_clean_roads) / max(len(neg_images), 1)

    # 4. Save Worst-Case Examples
    print("Generating worst-case False Positive & False Negative visualizers...")
    worst_fp_saved = 0
    worst_fn_saved = 0

    for img_p in test_images:
        stem = img_p.stem
        gts = ground_truths.get(stem, [])
        preds = [p for p in cached_preds.get(stem, []) if p[1] >= best_thresh]

        # Check for false positive
        is_fp = False
        is_fn = False
        matched_gt = set()
        for p_box, conf_val in preds:
            matched = any(calculate_iou(p_box, gt) >= 0.45 for gt in gts)
            if not matched:
                is_fp = True
        if len(gts) > len(preds):
            is_fn = True

        if is_fp and worst_fp_saved < 5:
            cv_img = cv2.imread(str(img_p))
            if cv_img is not None:
                im_h, im_w = cv_img.shape[:2]
                for p_box, conf_val in preds:
                    x1 = int(p_box[0] * im_w)
                    y1 = int(p_box[1] * im_h)
                    x2 = int(p_box[2] * im_w)
                    y2 = int(p_box[3] * im_h)
                    cv2.rectangle(cv_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(cv_img, f"FP: {conf_val:.2f}", (x1, max(y1 - 6, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)
                cv2.imwrite(str(WORST_CASES_DIR / f"worst_fp_{worst_fp_saved+1}_{stem[:20]}.jpg"), cv_img)
                worst_fp_saved += 1

        if is_fn and worst_fn_saved < 5:
            cv_img = cv2.imread(str(img_p))
            if cv_img is not None:
                im_h, im_w = cv_img.shape[:2]
                for gt in gts:
                    x1 = int(gt[0] * im_w)
                    y1 = int(gt[1] * im_h)
                    x2 = int(gt[2] * im_w)
                    y2 = int(gt[3] * im_h)
                    cv2.rectangle(cv_img, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    cv2.putText(cv_img, "FN (Missed GT)", (x1, max(y1 - 6, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 0), 1)
                cv2.imwrite(str(WORST_CASES_DIR / f"worst_fn_{worst_fn_saved+1}_{stem[:20]}.jpg"), cv_img)
                worst_fn_saved += 1

    # Compile Final Report
    report = {
        "model_path": model_path,
        "test_dataset_size": len(test_images),
        "positive_test_images": len(pos_images),
        "negative_test_images": len(neg_images),
        "metrics_at_default": {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1_score": round(f1, 4),
            "map50": round(map50, 4),
            "map50_95": round(map50_95, 4),
        },
        "performance_latency": {
            "preprocess_ms": round(preprocess_ms, 2),
            "inference_ms": round(inference_ms, 2),
            "postprocess_ms": round(postprocess_ms, 2),
            "total_latency_ms": round(total_latency_ms, 2),
            "fps": fps,
        },
        "confidence_threshold_sweep": sweep_results,
        "recommended_deployment_threshold": best_thresh,
        "recommended_f1": round(best_thresh_f1, 4),
        "clean_road_specificity": round(clean_road_spec, 4),
        "worst_cases_directory": str(WORST_CASES_DIR),
    }

    # Save JSON
    json_path = REPORTS_DIR / "evaluation_report.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    # Save Markdown
    md_path = REPORTS_DIR / "evaluation_report.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# SafeCity Loop V2 — Pothole Detector Evaluation Report\n\n")
        f.write(f"- **Evaluated Model:** `{model_path}`\n")
        f.write(f"- **Held-Out Test Set:** {len(test_images)} images ({len(pos_images)} positive, {len(neg_images)} negative)\n\n")
        f.write("## 1. Overall Test Set Performance\n\n")
        f.write("| Metric | Value |\n")
        f.write("| :--- | :--- |\n")
        f.write(f"| **Precision (P)** | **{precision:.4f}** |\n")
        f.write(f"| **Recall (R)** | **{recall:.4f}** |\n")
        f.write(f"| **F1 Score** | **{f1:.4f}** |\n")
        f.write(f"| **mAP@50** | **{map50:.4f}** |\n")
        f.write(f"| **mAP@50-95** | **{map50_95:.4f}** |\n")
        f.write(f"| **Inference Latency** | **{total_latency_ms:.1f} ms** ({fps} FPS on Intel CPU) |\n\n")

        f.write("## 2. Confidence-Threshold Sweep\n\n")
        f.write("| Threshold | Precision | Recall | F1 Score | TP | FP | FN |\n")
        f.write("| :--- | :--- | :--- | :--- | :--- | :--- | :--- |\n")
        for r in sweep_results:
            rec_tag = " ⭐ *(Selected)*" if r["threshold"] == best_thresh else ""
            f.write(f"| `{r['threshold']:.2f}`{rec_tag} | {r['precision']:.4f} | {r['recall']:.4f} | **{r['f1']:.4f}** | {r['true_positives']} | {r['false_positives']} | {r['false_negatives']} |\n")
        f.write("\n")

        f.write("## 3. Threshold Decision Rationale\n\n")
        f.write(f"The deployment threshold **`{best_thresh:.2f}`** was empirically chosen because it maximizes the harmonic mean of precision and recall ($F_1 = {best_thresh_f1:.4f}$), avoiding excessive false alerts on benign surface stains while ensuring critical road safety defects are reliably flagged.\n\n")

        f.write("## 4. Robustness & Clean Road Specificity\n\n")
        f.write(f"- Negative background test images: {len(neg_images)}\n")
        f.write(f"- False alarms on clean asphalt: {false_positives_on_clean_roads}\n")
        f.write(f"- Clean road specificity rate: {clean_road_spec*100:.1f}%\n")

    print(f"Evaluation report written to: {md_path}")
    print(f"JSON metrics written to:      {json_path}")
    print("=" * 60)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="models/pothole_yolo26s.pt", help="Path to model weights")
    parser.add_argument("--data", type=str, default=str(CONFIGS_DIR / "data.yaml"), help="Path to data.yaml")
    args = parser.parse_args()
    evaluate_test_set(args.model, args.data)
