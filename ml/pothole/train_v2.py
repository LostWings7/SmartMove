"""
SafeCity Loop V2 — Pothole Detector V2 Retraining Pipeline
Phases 8, 9, 10 Implementation:
- Uses improved balanced dataset (data_v2.yaml) with 726 negative images & 224 mined hard negatives.
- Starts from base pretrained YOLO26s (yolo26s.pt).
- Configures realistic road augmentations (exposure variation, shadows, scale, mosaic).
- Optimizes multi-core CPU execution for Intel Core Ultra 5 225H.
- Saves best model to models/pothole_yolo26s_v2.pt while preserving models/pothole_yolo26s_v1.pt.
- Records all hyperparameters and environment parameters.
"""

import os
import sys
import json
import time
import shutil
import argparse
import platform
import subprocess
from datetime import datetime
from pathlib import Path
import torch
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
ML_DIR = PROJECT_ROOT / "ml" / "pothole"
RUNS_DIR = ML_DIR / "runs"
CONFIGS_DIR = ML_DIR / "configs"
MODELS_DIR = PROJECT_ROOT / "models"

RUNS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)


def main():
    parser = argparse.ArgumentParser(description="Train SafeCity Pothole Detector V2")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs")
    parser.add_argument("--patience", type=int, default=6, help="Early stopping patience")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--imgsz", type=int, default=512, help="Image size")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--name", type=str, default="pothole_yolo26s_v2", help="Run name")
    args = parser.parse_args()

    print("=" * 70)
    print("SAFECITY POTHOLE DETECTOR V2 RETRAINING")
    print(f"Base model:  yolo26s.pt")
    print(f"Data config: {CONFIGS_DIR / 'data_v2.yaml'}")
    print(f"Epochs:      {args.epochs} (Patience: {args.patience})")
    print(f"Batch size:  {args.batch}")
    print(f"Image size:  {args.imgsz}")
    print(f"Seed:        {args.seed}")
    print("=" * 70)

    # Hardware setup
    num_threads = min(os.cpu_count() or 14, 14)
    torch.set_num_threads(num_threads)
    print(f"[Hardware Setup] PyTorch configured with {num_threads} CPU threads on Intel Core Ultra 5.")

    base_model_path = PROJECT_ROOT / "yolo26s.pt"
    if not base_model_path.exists():
        raise FileNotFoundError(f"Missing base weights at {base_model_path}")

    model = YOLO(str(base_model_path))

    start_time = time.time()

    # Train model with realistic road domain augmentations
    train_kwargs = {
        "data": str(CONFIGS_DIR / "data_v2.yaml").replace("\\", "/"),
        "epochs": args.epochs,
        "patience": args.patience,
        "batch": args.batch,
        "imgsz": args.imgsz,
        "device": "cpu",
        "workers": 0,  # 0 workers on Windows prevents multiprocessing spawn deadlock
        "project": str(RUNS_DIR).replace("\\", "/"),
        "name": args.name,
        "exist_ok": True,
        "save": True,
        "save_period": 5,
        "val": True,
        "plots": True,
        "verbose": True,
        "seed": args.seed,
        # Realistic road domain augmentations
        "hsv_h": 0.015,
        "hsv_s": 0.5,
        "hsv_v": 0.4,       # Realistic exposure / shadow variation
        "degrees": 0.0,     # Roads are flat from vehicle perspective
        "translate": 0.1,   # Moderate translation
        "scale": 0.25,      # Scale variation for distance
        "fliplr": 0.5,      # Horizontal flip
        "mosaic": 0.5,      # Mosaic augmentation for small defect context
        "erasing": 0.1,     # Occlusion robustness
    }

    results = model.train(**train_kwargs)

    duration_sec = time.time() - start_time
    print("=" * 70)
    print(f"TRAINING COMPLETE IN {duration_sec:.1f}s ({duration_sec/60:.2f} mins)")
    print("=" * 70)

    # Locate best.pt
    run_path = RUNS_DIR / args.name
    best_weights = run_path / "weights" / "best.pt"
    if not best_weights.exists():
        best_weights = run_path / "weights" / "last.pt"

    target_v2 = MODELS_DIR / "pothole_yolo26s_v2.pt"
    target_active = MODELS_DIR / "pothole_yolo26s.pt"

    if best_weights.exists():
        shutil.copyfile(best_weights, target_v2)
        shutil.copyfile(best_weights, target_active)
        print(f"Saved trained V2 checkpoint to:")
        print(f"  - {target_v2}")
        print(f"  - {target_active}")

    # Metadata
    meta = {
        "timestamp": datetime.utcnow().strftime("%Y%m%d_%H%M%S"),
        "run_name": args.name,
        "model_architecture": "YOLO26s",
        "model_version": "2.0.0",
        "base_model": "yolo26s.pt",
        "dataset_yaml": str(CONFIGS_DIR / "data_v2.yaml").replace("\\", "/"),
        "device_used": "cpu",
        "training_duration_seconds": round(duration_sec, 2),
        "epochs_requested": args.epochs,
        "batch_size": args.batch,
        "image_size": args.imgsz,
        "patience": args.patience,
        "seed": args.seed,
    }
    with open(run_path / "v2_training_metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Metadata saved to {run_path / 'v2_training_metadata.json'}")


if __name__ == "__main__":
    main()
