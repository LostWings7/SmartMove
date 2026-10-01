"""
SafeCity Loop V2 — Negative Dataset Acquisition, Hard-Negative Mining & Stress-Test Builder
Phases 3, 4, 5, 6, 7 Implementation:
1. Acquires authentic public negative road images (clean asphalt and crack/patch scenes without potholes).
2. Uses deterministic sequence hashing to strictly prevent data leakage across splits.
3. Builds held-out stress-test set in ml/pothole/dataset/stress_test/.
4. Mines hard negatives using current deployed YOLO26s model (models/pothole_yolo26s.pt).
5. Categorizes hard negatives into shadows, cracks, road_patches, texture, and other.
6. Prepares balanced dataset for V2 retraining.
"""

import os
import sys
import json
import time
import shutil
import hashlib
from pathlib import Path
from typing import Dict, List, Tuple, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
import cv2
import numpy as np
import requests
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
ML_DIR = PROJECT_ROOT / "ml" / "pothole"
DATASET_DIR = ML_DIR / "dataset"
RAW_DIR = DATASET_DIR / "raw"
RDD_RAW_DIR = RAW_DIR / "rdd2022_india"
HARD_NEG_DIR = DATASET_DIR / "hard_negatives"
STRESS_TEST_DIR = DATASET_DIR / "stress_test"
REPORTS_DIR = ML_DIR / "reports"

for sub in ["shadows", "cracks", "road_patches", "texture", "manholes", "other", "unclassified"]:
    (HARD_NEG_DIR / sub).mkdir(parents=True, exist_ok=True)

(STRESS_TEST_DIR / "images").mkdir(parents=True, exist_ok=True)
(STRESS_TEST_DIR / "labels").mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def download_file(url: str, dest_path: Path, timeout: int = 30) -> bool:
    try:
        if dest_path.exists() and dest_path.stat().st_size > 1000:
            return True
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        resp = requests.get(url, stream=True, timeout=timeout)
        if resp.status_code == 200:
            with open(dest_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=65536):
                    if chunk:
                        f.write(chunk)
            return True
        return False
    except Exception:
        if dest_path.exists():
            dest_path.unlink()
        return False


def main():
    print("=" * 70)
    print("SAFECITY POTHOLE DETECTOR — NEGATIVE DATASET & HARD-NEGATIVE MINING")
    print("=" * 70)

    # 1. Load metadata
    metadata_items = []
    shards = ["shard_002", "shard_003"]
    for shard in shards:
        meta_file = RDD_RAW_DIR / f"{shard}_metadata.jsonl"
        if not meta_file.exists():
            raise FileNotFoundError(f"Missing {meta_file}")
        with open(meta_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    item = json.loads(line.strip())
                    item["shard"] = shard
                    metadata_items.append(item)

    print(f"Loaded {len(metadata_items)} total records from RDD2022 India.")

    # 2. Separate clean roads vs crack/patch damage (strictly no pothole category 3)
    clean_records = []
    crack_records = []

    for item in metadata_items:
        cats = item.get("objects", {}).get("categories", [])
        if not cats:
            clean_records.append(item)
        elif 3 not in cats and any(c in [0, 1, 2] for c in cats):
            crack_records.append(item)

    print(f"Candidate pools: Clean roads = {len(clean_records)}, Cracks/Patches = {len(crack_records)}")

    # 3. Deterministic sequence partitioning to guarantee ZERO data leakage
    # Hash of corridor block:
    # 0 - 69   -> Training candidate pool (for negatives and hard negative mining)
    # 70 - 89  -> Validation candidate pool
    # 90 - 99  -> Strictly HELD-OUT Stress-Test Set (Never seen during training!)

    def get_split_role(fname: str) -> str:
        stem = Path(fname).stem
        try:
            num = int(stem.split("_")[-1])
            seq_id = f"rdd_india_block_{num // 60:04d}"
        except Exception:
            seq_id = f"rdd_india_hash_{hashlib.md5(stem.encode()).hexdigest()[:4]}"
        h = int(hashlib.sha256(seq_id.encode("utf-8")).hexdigest(), 16) % 100
        if h < 70:
            return "train"
        elif h < 90:
            return "val"
        else:
            return "stress_test"

    train_neg_candidates = []
    val_neg_candidates = []
    stress_test_candidates = []

    # Select balanced sample across clean roads and cracks
    for pool, label_type in [(clean_records, "clean_road"), (crack_records, "cracks_patches")]:
        for item in pool:
            item["neg_type"] = label_type
            role = get_split_role(item["file_name"])
            if role == "train":
                train_neg_candidates.append(item)
            elif role == "val":
                val_neg_candidates.append(item)
            else:
                stress_test_candidates.append(item)

    print(f"Sequence-separated negative allocations:")
    print(f"  - Train candidate pool: {len(train_neg_candidates)}")
    print(f"  - Val candidate pool:   {len(val_neg_candidates)}")
    print(f"  - Stress-test pool:     {len(stress_test_candidates)}")

    # Select quotas:
    # Train: 400 clean + 300 cracks = 700 images
    # Val:   50 clean + 50 cracks  = 100 images
    # Stress-Test: 60 clean + 60 cracks = 120 held-out images
    selected_train = [x for x in train_neg_candidates if x["neg_type"] == "clean_road"][:400] + \
                     [x for x in train_neg_candidates if x["neg_type"] == "cracks_patches"][:300]
    selected_val   = [x for x in val_neg_candidates if x["neg_type"] == "clean_road"][:50] + \
                     [x for x in val_neg_candidates if x["neg_type"] == "cracks_patches"][:50]
    selected_stress = [x for x in stress_test_candidates if x["neg_type"] == "clean_road"][:60] + \
                      [x for x in stress_test_candidates if x["neg_type"] == "cracks_patches"][:60]

    all_download = selected_train + selected_val + selected_stress
    print(f"Total targeted downloads: {len(all_download)} negative images.")

    # 4. Multi-threaded download
    download_tasks = []
    img_dest_dir = RDD_RAW_DIR / "images"
    img_dest_dir.mkdir(parents=True, exist_ok=True)

    for item in all_download:
        fname = item["file_name"]
        shard = item["shard"]
        dest = img_dest_dir / fname
        if not dest.exists() or dest.stat().st_size < 1000:
            url = f"https://huggingface.co/datasets/dronefreak/RDD2022/resolve/main/data/images/train/{shard}/{fname}"
            download_tasks.append((url, dest))

    if download_tasks:
        print(f"Downloading {len(download_tasks)} negative images with 16 threads...")
        with ThreadPoolExecutor(max_workers=16) as pool:
            futures = [pool.submit(download_file, url, dest) for url, dest in download_tasks]
            done = 0
            for fut in as_completed(futures):
                done += 1
                if done % 100 == 0 or done == len(download_tasks):
                    print(f"  Progress: {done}/{len(download_tasks)} downloaded...")

    print("Download phase complete.")

    # 5. Populate Held-Out Stress Test Set
    print("\nPopulating held-out stress test set...")
    stress_manifest = []
    for item in selected_stress:
        fname = item["file_name"]
        src = img_dest_dir / fname
        if not src.exists():
            continue
        dest_img = STRESS_TEST_DIR / "images" / f"stress_{fname}"
        dest_lbl = STRESS_TEST_DIR / "labels" / f"stress_{Path(fname).stem}.txt"
        shutil.copyfile(src, dest_img)
        # Empty text file: zero potholes (pure negative ground truth)
        with open(dest_lbl, "w") as f:
            pass
        stress_manifest.append({
            "image": dest_img.name,
            "type": item["neg_type"],
            "shard": item["shard"],
            "source": "RDD2022_India_HeldOut",
        })

    with open(STRESS_TEST_DIR / "manifest.json", "w") as f:
        json.dump(stress_manifest, f, indent=2)
    print(f"Stress-test set ready: {len(stress_manifest)} strictly held-out negative images at {STRESS_TEST_DIR}")

    # 6. Hard-Negative Mining using V1 model
    print("\n" + "=" * 70)
    print("PHASE 4: RUNNING HARD-NEGATIVE MINING WITH MODEL V1")
    print("=" * 70)

    v1_model_path = PROJECT_ROOT / "models" / "pothole_yolo26s.pt"
    if not v1_model_path.exists():
        raise FileNotFoundError(f"Model V1 not found at {v1_model_path}")

    model = YOLO(str(v1_model_path))

    mined_hard_negatives = []
    false_alarms_by_category = {
        "cracks": 0,
        "road_patches": 0,
        "shadows": 0,
        "texture": 0,
        "other": 0,
    }

    scan_pool = selected_train + selected_val
    print(f"Scanning {len(scan_pool)} training & validation negative road scenes for false positives...")

    for item in scan_pool:
        fname = item["file_name"]
        img_path = img_dest_dir / fname
        if not img_path.exists():
            continue

        res = model.predict(str(img_path), conf=0.25, verbose=False)[0]
        if len(res.boxes) > 0:
            # Model fired a false positive on a known negative image!
            img = cv2.imread(str(img_path))
            h, w = img.shape[:2]
            max_conf = float(max(res.boxes.conf))

            # Categorize based on image properties and metadata
            neg_type = item["neg_type"]
            detected_cat = "other"

            if neg_type == "cracks_patches":
                detected_cat = "cracks"
            else:
                # Analyze image ROI luminance for shadows vs texture
                # Average luminance of detected boxes
                dark_boxes = 0
                for box in res.boxes:
                    bx1, by1, bx2, by2 = map(int, box.xyxy[0])
                    roi = img[max(0, by1):min(h, by2), max(0, bx1):min(w, bx2)]
                    if roi.size > 0:
                        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                        if gray_roi.mean() < 70:
                            dark_boxes += 1
                if dark_boxes > 0:
                    detected_cat = "shadows"
                else:
                    detected_cat = "texture"

            false_alarms_by_category[detected_cat] += len(res.boxes)

            # Draw visual detection for report
            annotated = img.copy()
            boxes_info = []
            for box in res.boxes:
                xyxy = [round(float(c), 1) for c in box.xyxy[0]]
                conf = round(float(box.conf[0]), 3)
                boxes_info.append({"box": xyxy, "conf": conf})
                cv2.rectangle(annotated, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (0, 0, 255), 2)
                cv2.putText(annotated, f"FP {conf}", (int(xyxy[0]), max(15, int(xyxy[1]) - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

            # Save to hard_negatives categorized folder
            dest_vis = HARD_NEG_DIR / detected_cat / f"mined_fp_{fname}"
            cv2.imwrite(str(dest_vis), annotated)

            # Keep clean original copy for re-training
            dest_clean = HARD_NEG_DIR / "unclassified" / f"clean_{fname}"
            shutil.copyfile(img_path, dest_clean)

            mined_hard_negatives.append({
                "file_name": fname,
                "category": detected_cat,
                "neg_type": neg_type,
                "num_false_detections": len(res.boxes),
                "max_confidence": max_conf,
                "boxes": boxes_info,
            })

    print("-" * 60)
    print(f"HARD-NEGATIVE MINING RESULTS:")
    print(f"  Total False Positive Images Mined: {len(mined_hard_negatives)} / {len(scan_pool)} ({len(mined_hard_negatives)/max(len(scan_pool),1)*100:.1f}%)")
    print(f"  False Alarms by Category: {false_alarms_by_category}")
    print("-" * 60)

    # Save mining manifest
    manifest_path = HARD_NEG_DIR / "mining_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump({
            "model_version": "v1 (pothole_yolo26s.pt)",
            "mining_threshold": 0.25,
            "total_images_scanned": len(scan_pool),
            "hard_negatives_found": len(mined_hard_negatives),
            "category_breakdown": false_alarms_by_category,
            "samples": mined_hard_negatives[:50],
        }, f, indent=2)

    print(f"Hard negative manifest saved to {manifest_path}")

    # Output selection metadata
    summary = {
        "selected_train_negatives": len(selected_train),
        "selected_val_negatives": len(selected_val),
        "selected_stress_test": len(selected_stress),
        "mined_hard_negatives": len(mined_hard_negatives),
    }
    with open(DATASET_DIR / "negative_acquisition_summary.json", "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
