"""
SafeCity Road Condition Intelligence — Water Segmentation Model Training
Architecture: YOLO26s-seg (Ultralytics Lightweight Segmentation)
Hardware target: Intel Core Ultra 5 225H / OpenVINO / CPU
Priority: Intel XPU -> CPU fallback
Outputs: models/water_yolo26s_seg_v1.pt, models/water_model.json
"""

import os
import sys
import json
import time
import shutil
import argparse
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
import torch
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
WATER_DIR = PROJECT_ROOT / "ml" / "water"
RUNS_DIR = WATER_DIR / "runs"
CONFIGS_DIR = WATER_DIR / "configs"
MODELS_DIR = PROJECT_ROOT / "models"

RUNS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)


def get_git_commit() -> str:
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(PROJECT_ROOT), stderr=subprocess.DEVNULL
        ).decode("utf-8").strip()
        return commit
    except Exception:
        return "UNKNOWN_GIT_COMMIT"


def detect_and_verify_device(requested_device: str = "auto") -> str:
    print("=" * 60)
    print("HARDWARE DEVICE DETECTION & VERIFICATION (WATER MODEL)")
    print("=" * 60)

    if requested_device.lower() in ["auto", "xpu"]:
        has_xpu = hasattr(torch, "xpu") and torch.xpu.is_available()
        if has_xpu:
            try:
                test_tensor = torch.zeros((2, 2), device="xpu")
                _ = test_tensor + 1.0
                print("[Device Check] Intel XPU tensor execution verified: SUCCESS.")
                return "xpu"
            except Exception as e:
                print(f"[Device Check] Intel XPU available but test failed: {e}")
                print("[Device Check] Falling back to CPU.")
                return "cpu"
        else:
            print("[Device Check] Native PyTorch Intel XPU not available in current environment.")
            print("[Device Check] Using CPU.")
            return "cpu"
    elif requested_device.lower() == "cuda":
        if torch.cuda.is_available():
            return "0"
        print("[Device Check] CUDA not available. Falling back to CPU.")
        return "cpu"
    else:
        return requested_device


def train_water_model(args) -> Path:
    device = detect_and_verify_device(args.device)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_name = args.name or f"water_yolo26s_seg_{timestamp}"
    save_dir = RUNS_DIR / run_name

    print(f"Training Run:  {run_name}")
    print(f"Target Device: {device}")
    print(f"Base Model:    {args.model}")
    print(f"Data Config:   {args.data}")
    print(f"Epochs:        {args.epochs} (Patience: {args.patience})")
    print(f"Batch Size:    {args.batch}")
    print(f"Image Size:    {args.imgsz}")
    print(f"Workers:       {args.workers}")
    print("=" * 60)

    # Initialize YOLO segmentation model
    model = YOLO(args.model)
    start_time = time.time()

    # Train model
    train_results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=device,
        workers=args.workers,
        patience=args.patience,
        freeze=args.freeze,
        fraction=args.fraction,
        project=str(RUNS_DIR),
        name=run_name,
        seed=args.seed,
        resume=args.resume,
        exist_ok=True,
        save=True,
        val=True,
        plots=True,
        task="segment",
    )

    elapsed_sec = round(time.time() - start_time, 2)
    print(f"Training completed in {elapsed_sec:.2f} seconds.")

    run_path = RUNS_DIR / run_name
    best_weights = run_path / "weights" / "best.pt"
    last_weights = run_path / "weights" / "last.pt"

    target_deploy = MODELS_DIR / "water_yolo26s_seg_v1.pt"
    if best_weights.exists():
        shutil.copy2(best_weights, target_deploy)
        print(f"[Model Deployment] Copied best weights to: {target_deploy}")
    elif last_weights.exists():
        shutil.copy2(last_weights, target_deploy)
        print(f"[Model Deployment] Copied last weights to: {target_deploy}")
    else:
        print(f"[WARNING] No checkpoint weights found in {run_path / 'weights'}")

    # Gather training metrics
    metrics_summary = {}
    if hasattr(train_results, "results_dict") and train_results.results_dict:
        metrics_summary = {k: round(float(v), 5) for k, v in train_results.results_dict.items() if isinstance(v, (int, float))}
    
    # Save training metadata
    meta = {
        "model_name": "SafeCity YOLO26s Water Segmentation",
        "model_version": "1.0.0",
        "architecture": "Ultralytics YOLO26s-seg",
        "task": "segment",
        "date_trained": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "training_time_sec": elapsed_sec,
        "base_model": args.model,
        "dataset_config": args.data,
        "epochs_requested": args.epochs,
        "batch_size": args.batch,
        "image_size": args.imgsz,
        "patience": args.patience,
        "seed": args.seed,
        "device": device,
        "software_versions": {
            "os": platform.platform(),
            "python_version": sys.version.split()[0],
            "torch_version": torch.__version__,
            "ultralytics_version": getattr(sys.modules.get("ultralytics"), "__version__", "unknown"),
            "git_commit": get_git_commit(),
        },
        "metrics_summary": metrics_summary,
        "weights_path": str(target_deploy.resolve()),
    }

    meta_file = MODELS_DIR / "water_model.json"
    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"Deployment metadata saved to: {meta_file}")

    return target_deploy


def parse_arguments():
    parser = argparse.ArgumentParser(description="SafeCity Water Segmentation Training")
    parser.add_argument("--model", type=str, default="yolo26s-seg.pt", help="Base pretrained segmentation model")
    parser.add_argument("--data", type=str, default=str(CONFIGS_DIR / "water_data.yaml"), help="Path to water data yaml")
    parser.add_argument("--epochs", type=int, default=20, help="Max training epochs")
    parser.add_argument("--imgsz", type=int, default=512, help="Image resolution")
    parser.add_argument("--batch", type=int, default=8, help="Batch size")
    parser.add_argument("--device", type=str, default="auto", help="auto / xpu / cpu / cuda")
    parser.add_argument("--workers", type=int, default=0, help="DataLoader workers (0 for Windows stability)")
    parser.add_argument("--patience", type=int, default=8, help="Early stopping patience")
    parser.add_argument("--freeze", type=int, default=10, help="Number of backbone layers to freeze (default: 10 for CPU efficiency)")
    parser.add_argument("--fraction", type=float, default=1.0, help="Fraction of dataset to use for training (default: 1.0)")
    parser.add_argument("--name", type=str, default=None, help="Run name")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--resume", action="store_true", help="Resume training")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    train_water_model(args)
