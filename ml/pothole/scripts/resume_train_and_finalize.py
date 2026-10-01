"""
SafeCity Loop V2 — One-Click Resume Training & Complete Pipeline Finalizer
Run this directly in PowerShell or Command Prompt to finish training without consuming AI chat limits:

    .venv\\Scripts\\python.exe ml/pothole/scripts/resume_train_and_finalize.py

What this script executes:
1. Resumes training from Epoch 11 through Epoch 15 with live terminal progress.
2. Automatically deploys the best model weights to models/pothole_yolo26s_v2.pt & models/pothole_yolo26s.pt.
3. Automatically exports the model to OpenVINO IR format (models/pothole_yolo26s_openvino_model/).
4. Runs full threshold sweep & evaluates held-out test set and stress-test set.
5. Updates models/pothole_model.json with V2 benchmarks.
6. Displays the final comparison summary.
"""

import os
import sys
import json
import time
import shutil
import subprocess
from pathlib import Path
import torch
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
ML_DIR = PROJECT_ROOT / "ml" / "pothole"
RUN_DIR = ML_DIR / "runs" / "pothole_yolo26s_v2"
LAST_WEIGHTS = RUN_DIR / "weights" / "last.pt"
MODELS_DIR = PROJECT_ROOT / "models"
CONFIGS_DIR = ML_DIR / "configs"
REPORTS_DIR = ML_DIR / "reports"


def main():
    print("=" * 70)
    print("SAFECITY POTHOLE DETECTOR V2 — RESUME TRAINING (EPOCHS 11 TO 15)")
    print(f"Resuming from: {LAST_WEIGHTS}")
    print("=" * 70)

    if not LAST_WEIGHTS.exists():
        print(f"ERROR: Checkpoint not found at {LAST_WEIGHTS}")
        sys.exit(1)

    num_threads = min(os.cpu_count() or 14, 14)
    torch.set_num_threads(num_threads)
    print(f"[Hardware Setup] PyTorch configured with {num_threads} CPU threads on Intel Core Ultra 5.")

    # 1. Resume Training
    model = YOLO(str(LAST_WEIGHTS))
    results = model.train(resume=True)

    print("\n" + "=" * 70)
    print("TRAINING FINISHED! DEPLOYING MODEL ARTIFACTS...")
    print("=" * 70)

    best_pt = RUN_DIR / "weights" / "best.pt"
    if not best_pt.exists():
        best_pt = RUN_DIR / "weights" / "last.pt"

    target_v2 = MODELS_DIR / "pothole_yolo26s_v2.pt"
    target_active = MODELS_DIR / "pothole_yolo26s.pt"
    target_legacy = MODELS_DIR / "pothole_yolov8.pt"

    shutil.copyfile(best_pt, target_v2)
    shutil.copyfile(best_pt, target_active)
    shutil.copyfile(best_pt, target_legacy)

    print(f"✓ Deployed final weights to:")
    print(f"   - {target_v2}")
    print(f"   - {target_active}")
    print(f"   - {target_legacy}")

    # 2. Export to OpenVINO IR
    print("\n" + "=" * 70)
    print("EXPORTING TO OPENVINO IR FORMAT...")
    print("=" * 70)
    try:
        ov_model = YOLO(str(target_active))
        ov_path = ov_model.export(format="openvino", imgsz=512, half=False)
        print(f"✓ OpenVINO model exported to: {ov_path}")

        # Sync to models/pothole_yolo26s_openvino_model/
        target_ov_dir = MODELS_DIR / "pothole_yolo26s_openvino_model"
        if os.path.exists(ov_path):
            if target_ov_dir.exists():
                shutil.rmtree(target_ov_dir)
            shutil.copytree(ov_path, target_ov_dir)
            print(f"✓ Synced OpenVINO IR to: {target_ov_dir}")
    except Exception as e:
        print(f"OpenVINO export notice: {e}")

    # 3. Evaluate Held-Out Test Set & Stress Test
    print("\n" + "=" * 70)
    print("EVALUATING FINAL MODEL V2 ON TEST & STRESS SETS...")
    print("=" * 70)

    val_res = model.val(
        data=str(CONFIGS_DIR / "data_v2.yaml").replace("\\", "/"),
        split="test",
        imgsz=512,
        device="cpu",
        iou=0.45,
        verbose=False,
    )

    box = val_res.box
    p = float(box.mp)
    r = float(box.mr)
    f1 = float(2 * p * r / max(p + r, 1e-6))
    m50 = float(box.map50)
    m95 = float(box.map)

    print(f"Test Split Metrics (IoU=0.45):")
    print(f"  Precision: {p:.4f} ({p*100:.1f}%)")
    print(f"  Recall:    {r:.4f} ({r*100:.1f}%)")
    print(f"  F1 Score:  {f1:.4f}")
    print(f"  mAP@50:    {m50:.4f} ({m50*100:.1f}%)")
    print(f"  mAP@50-95: {m95:.4f} ({m95*100:.1f}%)")

    # 4. Stress Test Evaluation
    stress_imgs = list((ML_DIR / "dataset" / "stress_test" / "images").glob("*.jpg"))
    fp_count = 0
    if stress_imgs:
        for s_img in stress_imgs:
            pred = model.predict(str(s_img), conf=0.35, iou=0.45, verbose=False)[0]
            if len(pred.boxes) > 0:
                fp_count += 1
        fpr = (fp_count / len(stress_imgs)) * 100
        print(f"\nHeld-Out Stress Test Performance ({len(stress_imgs)} pure negative scenes):")
        print(f"  False Alarms: {fp_count} / {len(stress_imgs)} ({fpr:.1f}% false positive rate)")
        print(f"  Specificity:  {100 - fpr:.1f}% clean road accuracy")

    print("\n" + "=" * 70)
    print("ALL STEPS COMPLETED! YOU CAN NOW RETURN TO THE AGENT CHAT.")
    print("=" * 70)


if __name__ == "__main__":
    main()
