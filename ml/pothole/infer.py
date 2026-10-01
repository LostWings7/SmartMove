"""
SafeCity Loop V2 — Pothole Inference Engine (PyTorch & OpenVINO)
Supports:
- Single image inference
- Image directory inference
- Video file inference
- Both PyTorch (.pt) and OpenVINO runtime backends
- Measures per-frame latency and FPS
- Outputs annotated media and structured JSON detections
"""

import os
import sys
import time
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional
import cv2
import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MODELS_DIR = PROJECT_ROOT / "models"


class PotholeDetector:
    """Unified detector supporting both PyTorch (.pt) and OpenVINO runtime models."""

    def __init__(self, model_path: str = "models/pothole_yolo26s.pt", device: str = "cpu", conf: float = 0.40):
        self.model_path = model_path
        self.conf = conf
        self.device = device
        self.is_openvino = "openvino" in str(model_path).lower() or Path(model_path).is_dir()
        self.model = None
        self._load_model()

    def _load_model(self):
        print(f"[Detector] Loading model from: {self.model_path}")
        from ultralytics import YOLO
        self.model = YOLO(self.model_path)
        print(f"[Detector] Model loaded successfully. Backend: {'OpenVINO' if self.is_openvino else 'PyTorch'} (Device: {self.device})")

    def predict_image(self, image: np.ndarray) -> Dict[str, Any]:
        """Runs detection on a single BGR OpenCV frame."""
        t0 = time.time()
        results = self.model.predict(
            source=image,
            conf=self.conf,
            device=self.device,
            verbose=False,
            imgsz=512,
        )
        latency_ms = (time.time() - t0) * 1000.0

        detections = []
        annotated_img = image.copy()
        h, w = image.shape[:2]

        if results and results[0].boxes is not None:
            boxes = results[0].boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf_val = float(box.conf[0])
                cls_id = int(box.cls[0])
                cls_name = "pothole"

                detections.append({
                    "class": cls_name,
                    "confidence": round(conf_val, 3),
                    "box": {
                        "x1": round(x1, 1),
                        "y1": round(y1, 1),
                        "x2": round(x2, 1),
                        "y2": round(y2, 1),
                        "width": round(x2 - x1, 1),
                        "height": round(y2 - y1, 1),
                    },
                })

                # Draw bounding box and label
                color = (0, 0, 255) if conf_val >= 0.65 else (0, 140, 255)
                cv2.rectangle(annotated_img, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                lbl = f"Pothole {conf_val:.0%}"
                cv2.putText(
                    annotated_img,
                    lbl,
                    (int(x1), max(int(y1) - 6, 14)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    2,
                )

        return {
            "pothole_count": len(detections),
            "detections": detections,
            "latency_ms": round(latency_ms, 2),
            "annotated_image": annotated_img,
        }

    def process_image_file(self, input_path: Path, output_path: Optional[Path] = None) -> Dict[str, Any]:
        """Runs detection on an image file on disk."""
        img = cv2.imread(str(input_path))
        if img is None:
            raise ValueError(f"Could not load image: {input_path}")

        res = self.predict_image(img)
        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(output_path), res["annotated_image"])
            res["output_path"] = str(output_path)

        del res["annotated_image"]  # Remove raw numpy array from return dict
        res["input_path"] = str(input_path)
        return res

    def process_video_file(self, input_path: Path, output_path: Path) -> Dict[str, Any]:
        """Runs temporal detection on video frames with deduplication."""
        cap = cv2.VideoCapture(str(input_path))
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {input_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        output_path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

        total_detections = 0
        frame_idx = 0
        t0 = time.time()

        # Temporal deduplication: cluster detections across frames
        unique_events = 0
        frames_since_last_detection = 999

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            res = self.predict_image(frame)
            count = res["pothole_count"]
            total_detections += count

            if count > 0:
                if frames_since_last_detection > 25:  # ~1 second gap indicates new pothole event
                    unique_events += 1
                frames_since_last_detection = 0
            else:
                frames_since_last_detection += 1

            # Watermark telemetry
            status_text = f"SafeCity Pothole AI | Frame {frame_idx+1}/{total_frames} | Potholes: {count} | Unique Events: {unique_events}"
            cv2.putText(res["annotated_image"], status_text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)

            out.write(res["annotated_image"])
            frame_idx += 1

        cap.release()
        out.release()
        duration = time.time() - t0

        return {
            "input_path": str(input_path),
            "output_path": str(output_path),
            "total_frames": frame_idx,
            "total_detections": total_detections,
            "deduplicated_events": unique_events,
            "processing_duration_sec": round(duration, 2),
            "average_fps": round(frame_idx / max(duration, 0.01), 1),
        }


def main():
    parser = argparse.ArgumentParser(description="SafeCity Pothole Inference CLI")
    parser.add_argument("--source", type=str, required=True, help="Path to image, directory, or video")
    parser.add_argument("--model", type=str, default="models/pothole_yolo26s.pt", help="Model weights or OpenVINO dir")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu, openvino, etc.)")
    parser.add_argument("--conf", type=float, default=0.40, help="Confidence threshold (default: 0.40)")
    parser.add_argument("--output", type=str, default="ml/pothole/reports/predictions", help="Output directory")
    args = parser.parse_args()

    detector = PotholeDetector(model_path=args.model, device=args.device, conf=args.conf)
    src_path = Path(args.source)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    if src_path.is_file():
        ext = src_path.suffix.lower()
        if ext in [".jpg", ".jpeg", ".png", ".webp"]:
            out_file = out_dir / f"pred_{src_path.name}"
            res = detector.process_image_file(src_path, out_file)
            print(json.dumps(res, indent=2))
        elif ext in [".mp4", ".mov", ".avi"]:
            out_file = out_dir / f"pred_{src_path.stem}.mp4"
            res = detector.process_video_file(src_path, out_file)
            print(json.dumps(res, indent=2))
    elif src_path.is_dir():
        results = []
        for img_p in sorted(list(src_path.glob("*.jpg")) + list(src_path.glob("*.png"))):
            out_file = out_dir / f"pred_{img_p.name}"
            res = detector.process_image_file(img_p, out_file)
            results.append(res)
        print(f"Processed {len(results)} images into {out_dir}")


if __name__ == "__main__":
    main()
