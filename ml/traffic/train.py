"""
SafeCity Loop V2 — Traffic AI Detector Training Script
Custom Road-User Detection (pedestrian, rider, car, truck, bus, motorcycle, bicycle)
Architecture: YOLO26s (Transfer Learning from pretrained baseline)
Hardware target: Intel Core Ultra 5 225H / OpenVINO / CPU
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

    # Check torch.xpu
    if hasattr(torch, "xpu"):
        try:
            if torch.xpu.is_available():
                # Perform a dry-run tensor test
                test_t = torch.zeros(1, device="xpu")
                hw_info["xpu_available"] = True
                hw_info["selected_device"] = "xpu"
        except Exception:
            hw_info["xpu_available"] = False

    # Check OpenVINO
    if ov is not None:
        try:
            core = ov.Core()
            hw_info["openvino_devices"] = list(core.available_devices)
        except Exception as e:
            hw_info["openvino_devices"] = []

    return hw_info


def parse_args():
    parser = argparse.ArgumentParser(description="SafeCity Traffic AI Detector Training")
    parser.add_argument("--model", type=str, default="yolo26s.pt", help="Base pretrained weights")
    parser.add_argument("--data", type=str, default="ml/traffic/configs/traffic_data.yaml", help="Path to data yaml")
    parser.add_argument("--epochs", type=int, default=8, help="Number of training epochs")
    parser.add_argument("--imgsz", type=int, default=512, help="Image size (e.g. 512 or 640)")
    parser.add_argument("--batch", type=int, default=8, help="Batch size")
    parser.add_argument("--device", type=str, default=None, help="Device to use ('cpu', 'xpu', or auto)")
    parser.add_argument("--workers", type=int, default=4, help="DataLoader workers")
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience")
    parser.add_argument("--project", type=str, default="ml/traffic/runs", help="Project runs directory")
    parser.add_argument("--name", type=str, default="traffic_yolo26s_v1", help="Run run directory name")
    parser.add_argument("--resume", action="store_true", help="Resume from last checkpoint")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    return parser.parse_args()


def main():
    args = parse_args()
    print("=" * 70)
    print("SAFECITY LOOP V2 — CUSTOM TRAFFIC DETECTOR TRAINING (YOLO26s)")
    print("=" * 70)

    hw = detect_hardware()
    print(f"Hardware Profile:")
    print(f"  CPU: {hw['cpu']} ({platform.machine()})")
    print(f"  Torch: {hw['torch']} | CUDA: {hw['cuda_available']} | XPU: {hw['xpu_available']}")
    print(f"  OpenVINO Devices: {hw['openvino_devices']}")

    device = args.device or hw["selected_device"]
    print(f"Selected Training Device: {device.upper()}")

    # Verify data yaml exists and resolve absolute paths
    data_path = os.path.abspath(args.data)
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data configuration not found at {data_path}")

    # Check base model exists
    base_model_path = args.model
    if not os.path.exists(base_model_path):
        # Look in workspace root
        if os.path.exists(os.path.join(".", base_model_path)):
            base_model_path = os.path.join(".", base_model_path)
        else:
            print(f"[Warning] Pretrained model {base_model_path} not found locally, Ultralytics will auto-download.")

    from ultralytics import YOLO

    print(f"\n[Training Setup]")
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
    )
    duration = time.time() - start_time
    print(f"\n[Training Complete in {duration:.1f}s ({duration/60:.2f} mins)]")

    # Locate best.pt across candidate output paths
    candidate_dirs = [
        os.path.join(args.project, args.name),
        os.path.join("runs", "detect", args.project, args.name),
        os.path.join("runs", "detect", args.name),
    ]
    trained_weights = None
    actual_run_dir = os.path.join(args.project, args.name)
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

    if not trained_weights or not os.path.exists(trained_weights):
        raise FileNotFoundError(f"Trained checkpoint not found in candidate paths: {candidate_dirs}")

    # Deploy to models/ directory as models/traffic_yolo26s_v1.pt without overwriting existing pothole weights!
    target_deploy_path = os.path.join("models", "traffic_yolo26s_v1.pt")
    shutil.copy(trained_weights, target_deploy_path)
    print(f"\n[SUCCESS] Deployed custom traffic weights to: {target_deploy_path} ({os.path.getsize(target_deploy_path):,} bytes)")

    # Save training session summary
    summary = {
        "model_name": "SafeCity Traffic YOLO26s v1",
        "base_model": args.model,
        "trained_weights": target_deploy_path,
        "epochs": args.epochs,
        "batch_size": args.batch,
        "image_size": args.imgsz,
        "device": device,
        "duration_seconds": round(duration, 2),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "hardware": hw,
        "classes": ["pedestrian", "rider", "car", "truck", "bus", "motorcycle", "bicycle"],
    }
    with open(os.path.join(run_dir, "training_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary


if __name__ == "__main__":
    main()
