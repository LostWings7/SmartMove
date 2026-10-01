import os
import glob
import json
import hashlib
from pathlib import Path
from collections import defaultdict
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
processed_dir = PROJECT_ROOT / "ml" / "pothole" / "dataset" / "processed"
reports_dir = PROJECT_ROOT / "ml" / "pothole" / "reports"
reports_dir.mkdir(parents=True, exist_ok=True)

splits = ["train", "val", "test"]
audit_data = {}

for split in splits:
    img_files = sorted(list((processed_dir / "images" / split).glob("*.jpg")))
    lbl_files = sorted(list((processed_dir / "labels" / split).glob("*.txt")))
    
    pos_count = 0
    neg_count = 0
    all_boxes = []
    tiny_boxes = 0
    huge_boxes = 0
    multi_box_imgs = 0
    img_hashes = {}
    sources = defaultdict(int)
    
    for img_p in img_files:
        src = "RDD2022_India" if "RDD2022" in img_p.name else "Secondary_Roboflow"
        sources[src] += 1
        with open(img_p, "rb") as f:
            h = hashlib.md5(f.read()).hexdigest()
        img_hashes[img_p.name] = h
        
        lbl_p = processed_dir / "labels" / split / f"{img_p.stem}.txt"
        if not lbl_p.exists() or lbl_p.stat().st_size == 0:
            neg_count += 1
        else:
            boxes = []
            with open(lbl_p, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        cls_id, x, y, w, h = map(float, parts[:5])
                        area = w * h
                        boxes.append({"cls": cls_id, "x": x, "y": y, "w": w, "h": h, "area": area, "aspect": w / max(h, 1e-4)})
                        if area < 0.002:
                            tiny_boxes += 1
                        elif area > 0.40:
                            huge_boxes += 1
            if len(boxes) > 0:
                pos_count += 1
                if len(boxes) > 1:
                    multi_box_imgs += 1
                all_boxes.extend(boxes)
            else:
                neg_count += 1
                
    audit_data[split] = {
        "total_images": len(img_files),
        "positive_images": pos_count,
        "negative_images": neg_count,
        "negative_ratio_pct": round((neg_count / max(len(img_files), 1)) * 100, 2),
        "total_instances": len(all_boxes),
        "instances_per_pos_image": round(len(all_boxes) / max(pos_count, 1), 2),
        "tiny_boxes_lt_002": tiny_boxes,
        "huge_boxes_gt_040": huge_boxes,
        "multi_box_images": multi_box_imgs,
        "sources": dict(sources),
        "hashes": img_hashes,
    }

train_hashes = set(audit_data["train"]["hashes"].values())
val_hashes = set(audit_data["val"]["hashes"].values())
test_hashes = set(audit_data["test"]["hashes"].values())

cross_train_val = len(train_hashes.intersection(val_hashes))
cross_train_test = len(train_hashes.intersection(test_hashes))
cross_val_test = len(val_hashes.intersection(test_hashes))

print("=" * 60)
print("PHASE 2: DATASET AUDIT RESULTS")
print("=" * 60)
for split, d in audit_data.items():
    print(f"[{split.upper()}] Images: {d['total_images']}, Pos: {d['positive_images']}, Neg: {d['negative_images']} ({d['negative_ratio_pct']}%), Instances: {d['total_instances']}")
    print(f"   Sources: {d['sources']}")
    print(f"   Tiny boxes (<0.002): {d['tiny_boxes_lt_002']}, Huge boxes (>0.40): {d['huge_boxes_gt_040']}, Multi-box imgs: {d['multi_box_images']}")

print("-" * 60)
print(f"Cross-split hash overlap: Train-Val: {cross_train_val}, Train-Test: {cross_train_test}, Val-Test: {cross_val_test}")
print("=" * 60)

# Save audit report
with open(reports_dir / "phase2_dataset_audit.json", "w") as f:
    # Remove large hash dict for json readability
    dump_data = {k: {sk: sv for sk, sv in v.items() if sk != "hashes"} for k, v in audit_data.items()}
    dump_data["cross_split_duplicates"] = {
        "train_val": cross_train_val,
        "train_test": cross_train_test,
        "val_test": cross_val_test,
    }
    json.dump(dump_data, f, indent=2)

print(f"Saved audit report to {reports_dir / 'phase2_dataset_audit.json'}")
