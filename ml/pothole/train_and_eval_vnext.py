"""
SafeCity Road Condition AI — Pothole Model VNEXT Training, Sweep, NMS & Stress-Test (Sections 9-13)
1. Fine-tunes models/pothole_yolo26s_vNEXT.pt on sanitized, hard-negative-enriched data_v3.yaml
   without overwriting baseline checkpoints.
2. Evaluates NMS & Containment-Aware Deduplication (IoU + IoMin) to eliminate nested duplicate boxes.
3. Runs the 11-point confidence sweep (0.20 to 0.80) on the held-out test split.
4. Evaluates false positives per 100 images and false positive rate across all 7 held-out stress-test
   categories: non_road, cracks, shadows, manholes, road_patches, wet_roads, normal_roads.
"""

import os
import sys
import json
import time
import shutil
from pathlib import Path
from typing import List, Tuple, Dict, Any
import cv2
import numpy as np
import torch
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.services.road_context import road_context_service

POTHOLE_DIR = PROJECT_ROOT / "ml" / "pothole"
CONFIGS_DIR = POTHOLE_DIR / "configs"
RUNS_DIR = POTHOLE_DIR / "runs"
REPORTS_DIR = POTHOLE_DIR / "reports"
STRESS_DIR = POTHOLE_DIR / "dataset" / "stress_test"
PROC_V3 = POTHOLE_DIR / "dataset" / "processed_v3"
MODELS_DIR = PROJECT_ROOT / "models"


def compute_iou_and_iomin(box_a: List[float], box_b: List[float]) -> Tuple[float, float]:
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(1e-6, (box_a[2] - box_a[0]) * (box_a[3] - box_a[1]))
    area_b = max(1e-6, (box_b[2] - box_b[0]) * (box_b[3] - box_b[1]))
    union = area_a + area_b - inter
    iou = inter / max(union, 1e-6)
    iomin = inter / max(min(area_a, area_b), 1e-6)
    return iou, iomin


def containment_nms(
    boxes_xyxy: List[List[float]],
    confs: List[float],
    iou_thresh: float = 0.45,
    iomin_thresh: float = 0.75,
) -> List[int]:
    """
    Suppresses both overlapping boxes (IoU >= iou_thresh) and nested/concentric duplicate boxes
    for a single physical defect (IoMin >= iomin_thresh).
    """
    if not boxes_xyxy:
        return []
    order = sorted(range(len(confs)), key=lambda i: confs[i], reverse=True)
    keep = []
    for i in order:
        suppress = False
        for k in keep:
            iou, iomin = compute_iou_and_iomin(boxes_xyxy[i], boxes_xyxy[k])
            if iou >= iou_thresh or iomin >= iomin_thresh:
                suppress = True
                break
        if not suppress:
            keep.append(i)
    return keep


def evaluate_predictions_against_gt(
    model: YOLO,
    test_img_dir: Path,
    test_lbl_dir: Path,
    thresholds: List[float],
    use_road_context: bool = True,
    use_containment_nms: bool = True,
) -> Dict[str, Dict[str, Any]]:
    img_files = sorted(list(test_img_dir.glob("*.jpg")))
    all_preds = []
    all_gts = []

    for img_p in img_files:
        im = cv2.imread(str(img_p))
        if im is None:
            continue
        h, w = im.shape[:2]
        ctx = road_context_service.analyze_frame(im) if use_road_context else None

        lbl_p = test_lbl_dir / f"{img_p.stem}.txt"
        gt_boxes = []
        if lbl_p.exists():
            for line in open(lbl_p, encoding="utf-8"):
                parts = line.strip().split()
                if len(parts) >= 5:
                    cx, cy, bw, bh = map(float, parts[1:5])
                    gt_boxes.append([
                        (cx - bw / 2.0) * w,
                        (cy - bh / 2.0) * h,
                        (cx + bw / 2.0) * w,
                        (cy + bh / 2.0) * h,
                    ])
        all_gts.append(gt_boxes)

        res = model.predict(str(img_p), conf=0.15, iou=0.45, imgsz=512, verbose=False)[0]
        raw_boxes = []
        raw_confs = []
        for b in res.boxes:
            xyxy = b.xyxy[0].tolist()
            cf = float(b.conf[0])
            if use_road_context and ctx is not None:
                sc = road_context_service.evaluate_box(tuple(xyxy), cf, ctx, "pothole")
                if not sc.is_valid_road_candidate:
                    continue
            raw_boxes.append(xyxy)
            raw_confs.append(cf)

        if use_containment_nms:
            keep_idx = containment_nms(raw_boxes, raw_confs, iou_thresh=0.45, iomin_thresh=0.75)
            raw_boxes = [raw_boxes[i] for i in keep_idx]
            raw_confs = [raw_confs[i] for i in keep_idx]

        all_preds.append((raw_boxes, raw_confs))

    sweep_results = {}
    for th in thresholds:
        tp = 0
        fp = 0
        fn = 0
        for gt_boxes, (p_boxes, p_confs) in zip(all_gts, all_preds):
            filt_boxes = [b for b, c in zip(p_boxes, p_confs) if c >= th]
            filt_confs = [c for c in p_confs if c >= th]
            matched_gt = set()

            for pb in filt_boxes:
                best_iou = 0.0
                best_gi = -1
                for gi, gb in enumerate(gt_boxes):
                    if gi in matched_gt:
                        continue
                    iou, _ = compute_iou_and_iomin(pb, gb)
                    if iou > best_iou:
                        best_iou = iou
                        best_gi = gi
                if best_iou >= 0.45 and best_gi >= 0:
                    tp += 1
                    matched_gt.add(best_gi)
                else:
                    fp += 1
            fn += len(gt_boxes) - len(matched_gt)

        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        f1 = (2 * prec * rec) / max(prec + rec, 1e-6)
        sweep_results[f"{th:.2f}"] = {
            "threshold": th,
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
            "tp": tp,
            "fp": fp,
            "fn": fn,
        }
    return sweep_results


def evaluate_stress_categories(
    model: YOLO,
    conf_thresh: float = 0.30,
    use_road_context: bool = True,
) -> Dict[str, Any]:
    categories = [
        "non_road",
        "cracks",
        "shadows",
        "manholes",
        "road_patches",
        "wet_roads",
        "normal_roads",
    ]
    results = {}
    total_imgs = 0
    total_fp_boxes = 0
    total_fp_imgs = 0

    for cat in categories:
        cat_dir = STRESS_DIR / cat
        imgs = sorted(list(cat_dir.glob("*.jpg"))) if cat_dir.exists() else []
        n_imgs = len(imgs)
        fp_boxes = 0
        fp_imgs = 0

        for img_p in imgs:
            im = cv2.imread(str(img_p))
            if im is None:
                continue
            ctx = road_context_service.analyze_frame(im) if use_road_context else None
            res = model.predict(str(img_p), conf=conf_thresh, iou=0.45, imgsz=512, verbose=False)[0]
            valid_boxes = []
            valid_confs = []
            for b in res.boxes:
                xyxy = b.xyxy[0].tolist()
                cf = float(b.conf[0])
                if use_road_context and ctx is not None:
                    sc = road_context_service.evaluate_box(tuple(xyxy), cf, ctx, "pothole")
                    if not sc.is_valid_road_candidate:
                        continue
                valid_boxes.append(xyxy)
                valid_confs.append(cf)

            keep = containment_nms(valid_boxes, valid_confs, 0.45, 0.75)
            cnt = len(keep)
            if cnt > 0:
                fp_imgs += 1
                fp_boxes += cnt

        fp_per_100 = round((fp_boxes / max(n_imgs, 1)) * 100.0, 2)
        fpr_pct = round((fp_imgs / max(n_imgs, 1)) * 100.0, 2)
        results[cat] = {
            "images": n_imgs,
            "fp_images": fp_imgs,
            "fp_boxes": fp_boxes,
            "fp_per_100_images": fp_per_100,
            "false_positive_rate_pct": fpr_pct,
        }
        total_imgs += n_imgs
        total_fp_boxes += fp_boxes
        total_fp_imgs += fp_imgs

    results["overall"] = {
        "images": total_imgs,
        "fp_images": total_fp_imgs,
        "fp_boxes": total_fp_boxes,
        "fp_per_100_images": round((total_fp_boxes / max(total_imgs, 1)) * 100.0, 2),
        "false_positive_rate_pct": round((total_fp_imgs / max(total_imgs, 1)) * 100.0, 2),
    }
    return results


def main():
    num_threads = min(os.cpu_count() or 14, 14)
    torch.set_num_threads(num_threads)

    base_v2_pt = MODELS_DIR / "pothole_yolo26s_v2.pt"
    if not base_v2_pt.exists():
        base_v2_pt = MODELS_DIR / "pothole_yolo26s.pt"

    print("=" * 70)
    print("STEP 1: TRAINING POTHOLE MODEL VNEXT (models/pothole_yolo26s_vNEXT.pt)")
    print("=" * 70)

    model_train = YOLO(str(base_v2_pt))
    t0 = time.time()
    model_train.train(
        data=str(CONFIGS_DIR / "data_v3.yaml").replace("\\", "/"),
        epochs=2,
        fraction=0.50,
        imgsz=512,
        batch=16,
        device="cpu",
        workers=0,
        freeze=10,
        lr0=0.0006,
        lrf=0.1,
        mosaic=0.15,
        project=str(RUNS_DIR).replace("\\", "/"),
        name="pothole_yolo26s_vnext",
        exist_ok=True,
        save=True,
        val=True,
        seed=42,
    )
    train_sec = round(time.time() - t0, 2)

    run_dir = RUNS_DIR / "pothole_yolo26s_vnext"
    best_pt = run_dir / "weights" / "best.pt"
    if not best_pt.exists():
        best_pt = run_dir / "weights" / "last.pt"

    vnext_path = MODELS_DIR / "pothole_yolo26s_vNEXT.pt"
    shutil.copy2(best_pt, vnext_path)
    print(f"Saved VNEXT candidate weights to: {vnext_path} (trained in {train_sec}s)")

    # Evaluate Baseline V2 vs VNEXT
    print("\n" + "=" * 70)
    print("STEP 2: EVALUATING BASELINE V2 VS VNEXT (SWEEP + 7-CATEGORY STRESS TEST)")
    print("=" * 70)

    thresholds = [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.70, 0.80]
    test_imgs = PROC_V3 / "images" / "test"
    test_lbls = PROC_V3 / "labels" / "test"

    base_model = YOLO(str(base_v2_pt))
    vnext_model = YOLO(str(vnext_path))

    sweep_base_no_ctx = evaluate_predictions_against_gt(
        base_model, test_imgs, test_lbls, thresholds, use_road_context=False, use_containment_nms=False
    )
    sweep_vnext_with_ctx = evaluate_predictions_against_gt(
        vnext_model, test_imgs, test_lbls, thresholds, use_road_context=True, use_containment_nms=True
    )

    stress_base = evaluate_stress_categories(base_model, conf_thresh=0.20, use_road_context=False)
    stress_vnext = evaluate_stress_categories(vnext_model, conf_thresh=0.30, use_road_context=True)

    # Also test NMS duplicate suppression specifically on REGRESSION_005_water_pothole.jpg
    reg5_path = PROJECT_ROOT / "ml" / "road_condition_reports" / "regression_001" / "ROAD_CONDITION_REGRESSION_005_water_pothole.jpg"
    r5_pred = vnext_model.predict(str(reg5_path), conf=0.25, iou=0.45, imgsz=512, verbose=False)[0]
    r5_boxes = [b.xyxy[0].tolist() for b in r5_pred.boxes]
    r5_confs = [float(b.conf[0]) for b in r5_pred.boxes]
    r5_after_containment = containment_nms(r5_boxes, r5_confs, 0.45, 0.75)

    summary = {
        "training_time_sec": train_sec,
        "vnext_weights": str(vnext_path),
        "nms_analysis": {
            "regression_005_boxes_standard_nms_0_45": len(r5_boxes),
            "regression_005_boxes_containment_nms_0_75": len(r5_after_containment),
            "selected_iou_threshold": 0.45,
            "selected_iomin_containment_threshold": 0.75,
        },
        "threshold_sweep_baseline_v2": sweep_base_no_ctx,
        "threshold_sweep_vnext_with_road_context": sweep_vnext_with_ctx,
        "stress_test_baseline_v2_conf_0_20": stress_base,
        "stress_test_vnext_conf_0_30_with_context": stress_vnext,
    }

    out_file = REPORTS_DIR / "pothole_vnext_evaluation.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("Stress Test Comparison (Baseline V2 @ 0.20 vs VNEXT + RoadContext @ 0.30):")
    for cat in stress_vnext:
        b_fp = stress_base[cat]["fp_per_100_images"]
        n_fp = stress_vnext[cat]["fp_per_100_images"]
        print(f"  {cat:15s}: Baseline={b_fp:6.1f} FP/100 ({stress_base[cat]['false_positive_rate_pct']:5.1f}%) -> VNEXT={n_fp:6.1f} FP/100 ({stress_vnext[cat]['false_positive_rate_pct']:5.1f}%)")

    print(f"\nSaved full evaluation JSON to: {out_file}")


if __name__ == "__main__":
    main()
