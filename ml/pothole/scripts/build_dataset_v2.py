"""
SafeCity Loop V2 — Build V2 Dataset with Balanced Negative Representation
Phases 5 & 7:
- Preserves V1 processed dataset intact.
- Creates ml/pothole/dataset/processed_v2/
- Sanitizes positive annotations (filters out loose bounding boxes >0.55 image area and tiny noise specks).
- Integrates diverse clean negative road scenes (shadows, lighting, normal pavement).
- Integrates hard negatives (cracks, patches, shadows, rough aggregate).
- Generates ml/pothole/configs/data_v2.yaml
- Logs full split statistics.
"""

import os
import shutil
import json
import hashlib
from pathlib import Path
from collections import defaultdict
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
ML_DIR = PROJECT_ROOT / "ml" / "pothole"
DATASET_DIR = ML_DIR / "dataset"
PROCESSED_V1 = DATASET_DIR / "processed"
PROCESSED_V2 = DATASET_DIR / "processed_v2"
RAW_RDD = DATASET_DIR / "raw" / "rdd2022_india"
CONFIGS_DIR = ML_DIR / "configs"
REPORTS_DIR = ML_DIR / "reports"

PROCESSED_V2.mkdir(parents=True, exist_ok=True)
CONFIGS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

for split in ["train", "val", "test"]:
    (PROCESSED_V2 / "images" / split).mkdir(parents=True, exist_ok=True)
    (PROCESSED_V2 / "labels" / split).mkdir(parents=True, exist_ok=True)


def sanitize_box(line: str) -> str:
    parts = line.strip().split()
    if len(parts) < 5:
        return ""
    try:
        cls_id = int(float(parts[0]))
        x, y, w, h = map(float, parts[1:5])
        area = w * h
        # Reject pathological boxes that are absurdly large (e.g. >55% of image) or specks
        if area > 0.55:
            return ""
        if w < 0.003 or h < 0.003:
            return ""
        x = max(0.0, min(1.0, x))
        y = max(0.0, min(1.0, y))
        w = max(0.003, min(1.0, w))
        h = max(0.003, min(1.0, h))
        return f"{cls_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n"
    except Exception:
        return ""


def main():
    print("=" * 70)
    print("BUILDING BALANCED DATASET V2 (WITH HARD NEGATIVES & CLEAN ROADS)")
    print("=" * 70)

    stats = {
        "train": {"total": 0, "pos": 0, "neg": 0, "instances": 0},
        "val": {"total": 0, "pos": 0, "neg": 0, "instances": 0},
        "test": {"total": 0, "pos": 0, "neg": 0, "instances": 0},
    }

    # 1. Copy and sanitize existing V1 positive and existing negative images
    for split in ["train", "val", "test"]:
        v1_img_dir = PROCESSED_V1 / "images" / split
        v1_lbl_dir = PROCESSED_V1 / "labels" / split

        for img_p in v1_img_dir.glob("*.jpg"):
            dest_img = PROCESSED_V2 / "images" / split / img_p.name
            dest_lbl = PROCESSED_V2 / "labels" / split / f"{img_p.stem}.txt"

            shutil.copyfile(img_p, dest_img)

            lbl_p = v1_lbl_dir / f"{img_p.stem}.txt"
            valid_lines = []
            if lbl_p.exists() and lbl_p.stat().st_size > 0:
                with open(lbl_p, "r", encoding="utf-8") as f:
                    for line in f:
                        clean_line = sanitize_box(line)
                        if clean_line:
                            valid_lines.append(clean_line)

            with open(dest_lbl, "w", encoding="utf-8") as f:
                f.writelines(valid_lines)

            stats[split]["total"] += 1
            if len(valid_lines) > 0:
                stats[split]["pos"] += 1
                stats[split]["instances"] += len(valid_lines)
            else:
                stats[split]["neg"] += 1

    print("Initial V1 dataset imported:")
    for s, st in stats.items():
        print(f"  {s}: Total={st['total']}, Pos={st['pos']}, Neg={st['neg']}, Instances={st['instances']}")

    # 2. Add New Verified Negative Images and Mined Hard Negatives
    # Load shard metadata to track sequence blocks and types
    metadata_map = {}
    for shard in ["shard_002", "shard_003"]:
        meta_file = RAW_RDD / f"{shard}_metadata.jsonl"
        with open(meta_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    item = json.loads(line.strip())
                    metadata_map[item["file_name"]] = item

    def get_seq_hash(fname: str) -> int:
        stem = Path(fname).stem
        try:
            num = int(stem.split("_")[-1])
            seq_id = f"rdd_india_block_{num // 60:04d}"
        except Exception:
            seq_id = f"rdd_india_hash_{hashlib.md5(stem.encode()).hexdigest()[:4]}"
        return int(hashlib.sha256(seq_id.encode("utf-8")).hexdigest(), 16) % 100

    raw_img_dir = RAW_RDD / "images"
    existing_v2_names = set(p.name for p in (PROCESSED_V2 / "images").rglob("*.jpg"))

    added_negatives = {"train": 0, "val": 0, "test": 0}

    for img_p in sorted(raw_img_dir.glob("*.jpg")):
        orig_name = img_p.name
        # Skip if already in dataset
        if any(orig_name in existing_name for existing_name in existing_v2_names):
            continue

        meta = metadata_map.get(orig_name)
        if not meta:
            continue

        cats = meta.get("objects", {}).get("categories", [])
        # Strictly ensure NO pothole category (3)
        if 3 in cats:
            continue

        h = get_seq_hash(orig_name)
        # Sequence-leakage prevention:
        # h < 70 -> train
        # 70 <= h < 85 -> val
        # 85 <= h < 90 -> test
        # h >= 90 -> HELD-OUT STRESS TEST (DO NOT TOUCH!)
        if h < 70:
            target_split = "train"
        elif h < 85:
            target_split = "val"
        elif h < 90:
            target_split = "test"
        else:
            continue  # Reserved for stress test!

        target_name = f"RDD2022_Negative_{orig_name}"
        dest_img = PROCESSED_V2 / "images" / target_split / target_name
        dest_lbl = PROCESSED_V2 / "labels" / target_split / f"{Path(target_name).stem}.txt"

        shutil.copyfile(img_p, dest_img)
        # Empty text file = negative background sample
        with open(dest_lbl, "w") as f:
            pass

        existing_v2_names.add(target_name)
        stats[target_split]["total"] += 1
        stats[target_split]["neg"] += 1
        added_negatives[target_split] += 1

    print("-" * 60)
    print(f"Added New Verified Negatives:")
    print(f"  Train: +{added_negatives['train']} negative images")
    print(f"  Val:   +{added_negatives['val']} negative images")
    print(f"  Test:  +{added_negatives['test']} negative images")
    print("-" * 60)

    print("FINAL V2 DATASET BREAKDOWN:")
    for s, st in stats.items():
        neg_pct = (st["neg"] / max(st["total"], 1)) * 100
        print(f"  [{s.upper()}] Total: {st['total']} | Pos: {st['pos']} | Neg: {st['neg']} ({neg_pct:.1f}%) | Instances: {st['instances']}")

    # 3. Create data_v2.yaml
    yaml_content = {
        "path": str(PROCESSED_V2).replace("\\", "/"),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": 1,
        "names": {0: "pothole"},
    }

    yaml_path = CONFIGS_DIR / "data_v2.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(yaml_content, f, sort_keys=False)
    print(f"Saved {yaml_path}")

    # Save summary report
    summary = {
        "dataset_name": "SafeCity Pothole Dataset V2 (Balanced & Hard-Negative Augmented)",
        "date_created": "2026-09-30",
        "splits": stats,
        "added_negatives": added_negatives,
        "data_yaml": str(yaml_path).replace("\\", "/"),
    }
    with open(REPORTS_DIR / "dataset_v2_report.json", "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
