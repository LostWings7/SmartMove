"""
SafeCity Loop V2 — Dataset Quality Analysis & Reporting Tool
Analyzes:
- Split statistics (train, val, test)
- Positive vs. negative images (images with and without potholes)
- Total and split pothole instances
- Image resolution distributions
- Bounding-box dimensions (width, height, area, aspect ratio)
- Duplicate detection via MD5 content hashing
- Source dataset contributions
- Visual ground-truth sample generation
Outputs:
- ml/pothole/reports/dataset_report.json
- ml/pothole/reports/dataset_report.md
- ml/pothole/reports/visual_samples/
"""

import os
import json
import hashlib
from pathlib import Path
from collections import Counter
from typing import Dict, List, Any
import cv2
import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
ML_DIR = PROJECT_ROOT / "ml" / "pothole"
PROCESSED_DIR = ML_DIR / "dataset" / "processed"
REPORTS_DIR = ML_DIR / "reports"
SAMPLES_DIR = REPORTS_DIR / "visual_samples"

REPORTS_DIR.mkdir(parents=True, exist_ok=True)
SAMPLES_DIR.mkdir(parents=True, exist_ok=True)


def compute_stats(values: List[float]) -> Dict[str, float]:
    """Computes basic statistical metrics."""
    if not values:
        return {"min": 0.0, "max": 0.0, "mean": 0.0, "median": 0.0, "std": 0.0}
    arr = np.array(values, dtype=np.float64)
    return {
        "min": round(float(np.min(arr)), 4),
        "max": round(float(np.max(arr)), 4),
        "mean": round(float(np.mean(arr)), 4),
        "median": round(float(np.median(arr)), 4),
        "std": round(float(np.std(arr)), 4),
    }


def generate_dataset_report() -> Dict[str, Any]:
    print("=" * 60)
    print("Generating Comprehensive SafeCity Pothole Dataset Report")
    print("=" * 60)

    splits = ["train", "val", "test"]
    stats = {
        "total_images": 0,
        "splits": {},
        "positive_images": 0,
        "negative_images": 0,
        "total_instances": 0,
        "sources": Counter(),
        "image_resolutions": Counter(),
        "duplicates_detected": 0,
    }

    all_bb_widths = []
    all_bb_heights = []
    all_bb_areas = []
    all_bb_aspect_ratios = []

    hashes = set()
    sample_images_to_draw = []

    for split in splits:
        img_dir = PROCESSED_DIR / "images" / split
        lbl_dir = PROCESSED_DIR / "labels" / split

        split_images = 0
        split_positives = 0
        split_negatives = 0
        split_instances = 0
        split_bb_areas = []

        for img_path in sorted(list(img_dir.glob("*.jpg"))):
            split_images += 1
            stats["total_images"] += 1

            # Source attribution
            if "RDD2022_India" in img_path.name:
                stats["sources"]["RDD2022 (India Subset)"] += 1
            elif "Secondary_Pothole_Dataset" in img_path.name:
                stats["sources"]["Secondary Pothole Dataset (Roboflow/rupesh002)"] += 1
            else:
                stats["sources"]["Other"] += 1

            # Hash for duplicate detection
            with open(img_path, "rb") as f:
                img_hash = hashlib.md5(f.read()).hexdigest()
                if img_hash in hashes:
                    stats["duplicates_detected"] += 1
                else:
                    hashes.add(img_hash)

            # Check image dimensions
            try:
                with Image.open(img_path) as im:
                    w, h = im.size
                    stats["image_resolutions"][f"{w}x{h}"] += 1
            except Exception:
                continue

            # Read labels
            lbl_path = lbl_dir / f"{img_path.stem}.txt"
            boxes = []
            if lbl_path.exists():
                with open(lbl_path, "r", encoding="utf-8") as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            cls_id = int(parts[0])
                            bx, by, bw, bh = map(float, parts[1:5])
                            boxes.append((cls_id, bx, by, bw, bh))
                            split_instances += 1
                            stats["total_instances"] += 1

                            all_bb_widths.append(bw)
                            all_bb_heights.append(bh)
                            area = bw * bh
                            all_bb_areas.append(area)
                            split_bb_areas.append(area)
                            if bh > 0:
                                all_bb_aspect_ratios.append(bw / bh)

            if len(boxes) > 0:
                split_positives += 1
                stats["positive_images"] += 1
                if len(sample_images_to_draw) < 8:
                    sample_images_to_draw.append((img_path, boxes, split))
            else:
                split_negatives += 1
                stats["negative_images"] += 1

        stats["splits"][split] = {
            "images": split_images,
            "positive_images": split_positives,
            "negative_images": split_negatives,
            "instances": split_instances,
            "area_stats": compute_stats(split_bb_areas),
        }

    stats["bbox_statistics"] = {
        "width": compute_stats(all_bb_widths),
        "height": compute_stats(all_bb_heights),
        "area": compute_stats(all_bb_areas),
        "aspect_ratio": compute_stats(all_bb_aspect_ratios),
    }

    # Render Visual Samples with Ground Truth
    print("Generating annotated ground-truth visual samples...")
    for idx, (img_path, boxes, split) in enumerate(sample_images_to_draw):
        cv_img = cv2.imread(str(img_path))
        if cv_img is None:
            continue
        h, w = cv_img.shape[:2]
        for cls_id, bx, by, bw, bh in boxes:
            x1 = int((bx - bw / 2.0) * w)
            y1 = int((by - bh / 2.0) * h)
            x2 = int((bx + bw / 2.0) * w)
            y2 = int((by + bh / 2.0) * h)
            cv2.rectangle(cv_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(
                cv_img,
                f"GT: pothole ({bw*w:.0f}x{bh*h:.0f}px)",
                (x1, max(y1 - 6, 12)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 0, 255),
                1,
            )

        out_name = f"sample_gt_{split}_{idx+1}_{img_path.stem[:20]}.jpg"
        cv2.imwrite(str(SAMPLES_DIR / out_name), cv_img)

    # Convert Counters to dict for JSON serialization
    stats["sources"] = dict(stats["sources"])
    stats["image_resolutions"] = dict(stats["image_resolutions"].most_common(5))

    # Save JSON report
    json_path = REPORTS_DIR / "dataset_report.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    # Save Markdown report
    md_path = REPORTS_DIR / "dataset_report.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# SafeCity Loop V2 — Pothole Dataset Verification Report\n\n")
        f.write("Generated from verified dataset in `ml/pothole/dataset/processed/`.\n\n")
        f.write("## 1. Split & Instance Breakdown\n\n")
        f.write("| Split | Images | Positive Images | Negative (Background) Images | Pothole Instances |\n")
        f.write("| :--- | :--- | :--- | :--- | :--- |\n")
        for s in splits:
            info = stats["splits"][s]
            f.write(f"| **{s.capitalize()}** | {info['images']} | {info['positive_images']} | {info['negative_images']} | {info['instances']} |\n")
        f.write(f"| **Total** | **{stats['total_images']}** | **{stats['positive_images']}** | **{stats['negative_images']}** | **{stats['total_instances']}** |\n\n")

        f.write("## 2. Dataset Sources & Licensing\n\n")
        for src, count in stats["sources"].items():
            f.write(f"- **{src}:** {count} images ({count/stats['total_images']*100:.1f}%)\n")
        f.write(f"- **Duplicate Images Detected:** {stats['duplicates_detected']}\n\n")

        f.write("## 3. Bounding Box Geometry Statistics\n\n")
        f.write("| Metric | Min | Max | Mean | Median | Std Dev |\n")
        f.write("| :--- | :--- | :--- | :--- | :--- | :--- |\n")
        for metric, vals in stats["bbox_statistics"].items():
            f.write(f"| Normalized {metric.capitalize()} | {vals['min']} | {vals['max']} | {vals['mean']} | {vals['median']} | {vals['std']} |\n")
        f.write("\n")

        f.write("## 4. Top Image Resolutions\n\n")
        for res, count in stats["image_resolutions"].items():
            f.write(f"- `{res}`: {count} images\n")
        f.write("\n")

        f.write("## 5. Visual Samples\n\n")
        f.write(f"Generated ground truth annotation visualizers in `{SAMPLES_DIR}`.\n")

    print(f"Dataset report written to: {md_path}")
    print(f"JSON metrics written to:   {json_path}")
    print(f"Visual samples saved to:   {SAMPLES_DIR}")
    print("=" * 60)
    return stats


if __name__ == "__main__":
    generate_dataset_report()
