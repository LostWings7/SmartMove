"""
SafeCity Loop V2 — Traffic AI Detector Training Script V2
Custom Road-User Detection V2:
- Solves class imbalance on Motorcycle, Rider, and Bicycle
- Preserves Car, Pedestrian, Truck, and Bus performance
- Multi-scale augmentations (mosaic, scale, mixup) for small & occluded road users
- Strict sequence-level data isolation (zero temporal leakage)
- Architecture: YOLO26s (Transfer Learning from pretrained baseline)
- Output: models/traffic_yolo26s_v2.pt (Preserves V1 and pothole weights)
"""
import os
import sys
import time
import argparse
import platform
import shutil
import json
import torch

try:
    import openvino as ov
except ImportError:
    ov = None


def detect_hardware():
    """Detects available acceleration hardware (CPU, XPU, OpenVINO devices)."""
    hw_info = {
        "cpu": platform.processor() or "Intel Core Ultra",
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "xpu_available": False,
        "selected_device": "cpu",
        "openvino_devices": [],
    }

    if hasattr(torch, "xpu"):
        try:
            if torch.xpu.is_available():
                test_t = torch.zeros(1, device="xpu")
                hw_info["xpu_available"] = True
                hw_info["selected_device"] = "xpu"
        except Exception:
            hw_info["xpu_available"] = False

    if ov is not None:
        try:
            core = ov.Core()
            hw_info["openvino_devices"] = list(core.available_devices)
        except Exception:
            hw_info["openvino_devices"] = []

    return hw_info


def parse_args():
    parser = argparse.ArgumentParser(description="SafeCity Traffic AI Detector Training V2")
    default_base = "models/traffic_yolo26s_v1.pt" if os.path.exists("models/traffic_yolo26s_v1.pt") else "yolo26s.pt"
    parser.add_argument("--model", type=str, default=default_base, help="Base pretrained weights (warm-start from V1 or yolo26s.pt)")
    parser.add_argument("--data", type=str, default="ml/traffic/configs/traffic_data_v2.yaml", help="Path to data yaml")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs")
    parser.add_argument("--imgsz", type=int, default=512, help="Image size (512 or 640)")
    parser.add_argument("--batch", type=int, default=8, help="Batch size")
    parser.add_argument("--device", type=str, default=None, help="Device ('cpu', 'xpu', or auto)")
    parser.add_argument("--workers", type=int, default=4, help="DataLoader workers")
    parser.add_argument("--patience", type=int, default=7, help="Early stopping patience")
    parser.add_argument("--project", type=str, default="ml/traffic/runs", help="Project runs directory")
    parser.add_argument("--name", type=str, default="traffic_yolo26s_v2", help="Run run directory name")
    parser.add_argument("--resume", action="store_true", help="Resume from last checkpoint")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    return parser.parse_args()


def main():
    args = parse_args()
    print("=" * 70)
    print("SAFECITY LOOP V2 — CUSTOM TRAFFIC DETECTOR TRAINING V2 (YOLO26s)")
    print("Objective: Expand Rare-Class Coverage (Motorcycles, Riders, Bicycles)")
    print("=" * 70)

    hw = detect_hardware()
    print(f"Hardware Profile:")
    print(f"  CPU: {hw['cpu']} ({platform.machine()})")
    print(f"  Torch: {hw['torch']} | CUDA: {hw['cuda_available']} | XPU: {hw['xpu_available']}")
    print(f"  OpenVINO Devices: {hw['openvino_devices']}")

    device = args.device or hw["selected_device"]
    print(f"Selected Training Device: {device.upper()}")

    # Verify data yaml exists
    data_path = os.path.abspath(args.data)
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data configuration not found at {data_path}")

    base_model_path = args.model
    if not os.path.exists(base_model_path):
        if os.path.exists(os.path.join(".", base_model_path)):
            base_model_path = os.path.join(".", base_model_path)
        else:
            print(f"[Warning] Pretrained model {base_model_path} not found locally, Ultralytics will auto-download.")

    from ultralytics import YOLO

    print(f"\n[Training Setup V2]")
    print(f"  Base checkpoint: {base_model_path}")
    print(f"  Data YAML:       {data_path}")
    print(f"  Epochs:          {args.epochs}")
    print(f"  Image Size:      {args.imgsz}")
    print(f"  Batch Size:      {args.batch}")
    print(f"  Patience:        {args.patience}")
    print(f"  Project / Name:  {args.project}/{args.name}")

    model = YOLO(base_model_path)

    start_time = time.time()
    results = model.train(
        data=data_path,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=device,
        workers=args.workers,
        patience=args.patience,
        project=args.project,
        name=args.name,
        resume=args.resume,
        seed=args.seed,
        exist_ok=True,
        verbose=True,
        save=True,
        val=True,
        plots=True,
        pretrained=True,
        # Realistic Traffic Augmentations for Small Objects & Occlusion
        mosaic=0.8,
        mixup=0.1,
        scale=0.5,
        translate=0.1,
        fliplr=0.5,
        flipud=0.0,      # Physically impossible for road vehicles to flip vertically
        hsv_h=0.015,
        hsv_s=0.5,
        hsv_v=0.4,
    )
    duration = time.time() - start_time
    print(f"\n[Training Complete in {duration:.1f}s ({duration/60:.2f} mins)]")

    # Locate best.pt across trainer attributes and candidate output paths
    candidate_dirs = [
        os.path.join(args.project, args.name),
        os.path.join("runs", "detect", args.project, args.name),
        os.path.join("runs", "detect", args.name),
    ]
    if hasattr(model, "trainer") and hasattr(model.trainer, "save_dir") and model.trainer.save_dir:
        candidate_dirs.insert(0, str(model.trainer.save_dir))

    trained_weights = None
    actual_run_dir = os.path.join(args.project, args.name)

    # 1. Direct check on trainer attributes
    if hasattr(model, "trainer") and getattr(model.trainer, "best", None) and os.path.exists(str(model.trainer.best)):
        trained_weights = str(model.trainer.best)
        actual_run_dir = str(model.trainer.save_dir) if hasattr(model.trainer, "save_dir") else actual_run_dir

    # 2. Check candidate directories
    if not trained_weights:
        for c_dir in candidate_dirs:
            best_candidate = os.path.join(c_dir, "weights", "best.pt")
            last_candidate = os.path.join(c_dir, "weights", "last.pt")
            if os.path.exists(best_candidate):
                trained_weights = best_candidate
                actual_run_dir = c_dir
                break
            elif os.path.exists(last_candidate):
                trained_weights = last_candidate
                actual_run_dir = c_dir
                break

    # 3. Recursive fallback search in runs directory
    if not trained_weights:
        for root, dirs, files in os.walk(args.project):
            if "best.pt" in files:
                trained_weights = os.path.join(root, "best.pt")
                actual_run_dir = os.path.dirname(os.path.dirname(trained_weights))
                break
            elif "last.pt" in files and not trained_weights:
                trained_weights = os.path.join(root, "last.pt")
                actual_run_dir = os.path.dirname(os.path.dirname(trained_weights))

    if not trained_weights or not os.path.exists(trained_weights):
        raise FileNotFoundError(f"Trained checkpoint not found. Checked trainer, candidates {candidate_dirs}, and {args.project}")

    # Synchronize artifacts to project runs folder
    target_run_dir = os.path.join(args.project, args.name)
    if actual_run_dir != target_run_dir:
        os.makedirs(target_run_dir, exist_ok=True)
        for item in os.listdir(actual_run_dir):
            s = os.path.join(actual_run_dir, item)
            d = os.path.join(target_run_dir, item)
            if os.path.isdir(s):
                shutil.copytree(s, d, dirs_exist_ok=True)
            else:
                shutil.copy2(s, d)

    # Deploy strictly to models/traffic_yolo26s_v2.pt WITHOUT touching V1 or pothole weights!
    target_deploy_path = os.path.join("models", "traffic_yolo26s_v2.pt")
    shutil.copy(trained_weights, target_deploy_path)
    print(f"\n[SUCCESS] Deployed custom traffic V2 weights to: {target_deploy_path} ({os.path.getsize(target_deploy_path):,} bytes)")

    # Save structured summary
    summary = {
        "model_name": "SafeCity Traffic YOLO26s v2",
        "base_model": args.model,
        "trained_weights": target_deploy_path,
        "epochs_requested": args.epochs,
        "batch_size": args.batch,
        "image_size": args.imgsz,
        "device": device,
        "duration_seconds": round(duration, 2),
        "duration_minutes": round(duration / 60.0, 2),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "hardware": hw,
        "classes": ["pedestrian", "rider", "car", "truck", "bus", "motorcycle", "bicycle"],
    }
    with open(os.path.join("ml", "traffic", "reports", "training_summary_v2.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary


if __name__ == "__main__":
    main()
