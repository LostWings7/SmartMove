"""
SafeCity Road Condition Intelligence — Water Segmentation Inference Engine
Supports:
- Single image inference and video stream inference
- Precise pixel mask extraction
- Documented Road ROI estimation (lower drivable corridor) for visible-road coverage calculation:
    water_coverage_percent = (water_pixels / relevant_visible_road_pixels) * 100
- Multi-region extraction via connected components
- Transparent, configurable operational severity engine (LOW / MODERATE / HIGH / CRITICAL)
- Video temporal persistence & deduplication (multi-frame water events)
- Visual overlay rendering with configurable alpha
"""

import os
import sys
import time
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any
from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


@dataclass
class WaterRegion:
    region_id: int
    bounding_box: Tuple[int, int, int, int]  # x1, y1, x2, y2
    pixel_count: int
    coverage_percent: float                  # % of visible road ROI
    confidence: float
    polygon: Optional[List[List[float]]] = None  # Normalized coordinates [[x, y], ...]


@dataclass
class WaterSegmentationResult:
    has_water: bool
    total_water_regions: int
    regions: List[WaterRegion]
    total_water_pixels: int
    visible_road_pixels: int
    water_coverage_percent: float            # Operational visible-road coverage %
    severity: str                            # LOW / MODERATE / HIGH / CRITICAL
    mean_confidence: float
    max_confidence: float
    processing_time_ms: float
    mask: Optional[np.ndarray] = None        # Binary mask (H, W) uint8
    annotated_image: Optional[np.ndarray] = None


class WaterSeverityConfig:
    """Configurable operational thresholds for road water severity."""
    LOW_COVERAGE_MAX: float = 3.0           # Under 3% of road ROI: LOW
    MODERATE_COVERAGE_MAX: float = 10.0     # 3% to 10%: MODERATE
    HIGH_COVERAGE_MAX: float = 25.0         # 10% to 25%: HIGH
    CRITICAL_COVERAGE_MIN: float = 25.0     # > 25%: CRITICAL

    # Single large connected region trigger
    LARGE_REGION_PIXELS_CRITICAL: int = 40000
    LARGE_REGION_PIXELS_HIGH: int = 15000


class WaterSegmenter:
    """
    Production-grade Road Water & Ponding Segmentation Adapter.
    Follows SafeCity RoadHazardDetector architecture.
    """

    def __init__(
        self,
        model_path: str = "models/water_yolo26s_seg_v1.pt",
        conf_threshold: float = 0.25,
        device: str = "cpu",
        severity_config: Optional[WaterSeverityConfig] = None,
    ):
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.device = device
        self.config = severity_config or WaterSeverityConfig()
        self.model = None
        self.model_name = "SafeCity YOLO26s Water Segmenter"
        self.model_version = "1.0.0"
        self.is_loaded = False
        self._load_model()

    def _load_model(self):
        if Path(self.model_path).exists():
            try:
                self.model = YOLO(self.model_path)
                self.is_loaded = True
                print(f"[Water Segmenter] Successfully loaded weights from: {self.model_path}")
            except Exception as e:
                print(f"[Water Segmenter] Error loading model: {e}")
                self.is_loaded = False
        else:
            print(f"[Water Segmenter] Model file not found at {self.model_path}. Running in standby.")
            self.is_loaded = False

    def estimate_road_roi_mask(self, height: int, width: int) -> np.ndarray:
        """
        Creates a documented geometric Road Region-of-Interest (ROI) approximation
        representing the drivable roadway (lower perspective trapezoid / road corridor).
        This ensures water coverage is calculated against relevant visible road pixels,
        NOT against the sky, trees, or vehicle hood.
        """
        roi = np.zeros((height, width), dtype=np.uint8)
        # Perspective trapezoid: top at 40% height (vanishing horizon), base across full width
        pts = np.array([
            [int(width * 0.15), int(height * 0.40)],  # Top-left road horizon
            [int(width * 0.85), int(height * 0.40)],  # Top-right road horizon
            [width, height],                          # Bottom-right
            [0, height],                              # Bottom-left
        ], dtype=np.int32)
        cv2.fillPoly(roi, [pts], 1)
        return roi

    def calculate_severity(self, coverage_pct: float, max_region_pixels: int) -> str:
        """
        Transparent explainable operational severity calculation.
        Documented operational thresholds until calibrated with municipal sensor depth gauges.
        """
        if coverage_pct <= 0.0:
            return "NONE"
        if coverage_pct >= self.config.CRITICAL_COVERAGE_MIN or max_region_pixels >= self.config.LARGE_REGION_PIXELS_CRITICAL:
            return "CRITICAL"
        if coverage_pct >= self.config.HIGH_COVERAGE_MAX or max_region_pixels >= self.config.LARGE_REGION_PIXELS_HIGH:
            return "HIGH"
        if coverage_pct >= self.config.LOW_COVERAGE_MAX:
            return "MODERATE"
        return "LOW"

    def segment_image(
        self,
        image: np.ndarray,
        road_roi: Optional[np.ndarray] = None,
        overlay_alpha: float = 0.40,
    ) -> WaterSegmentationResult:
        if not self.is_loaded or self.model is None:
            return WaterSegmentationResult(
                has_water=False,
                total_water_regions=0,
                regions=[],
                total_water_pixels=0,
                visible_road_pixels=0,
                water_coverage_percent=0.0,
                severity="NONE",
                mean_confidence=0.0,
                max_confidence=0.0,
                processing_time_ms=0.0,
            )

        t_start = time.time()
        h, w = image.shape[:2]

        # Calculate or acquire road ROI
        if road_roi is None:
            road_roi = self.estimate_road_roi_mask(h, w)
        visible_road_pixels = int(road_roi.sum())
        if visible_road_pixels == 0:
            visible_road_pixels = h * w

        # Run inference
        results = self.model.predict(
            image,
            conf=self.conf_threshold,
            imgsz=512,
            device=self.device,
            verbose=False,
            task="segment",
        )

        water_mask = np.zeros((h, w), dtype=np.uint8)
        confidences = []
        raw_polygons = []

        if results and len(results) > 0 and results[0].masks is not None:
            masks_data = results[0].masks.data.cpu().numpy()
            boxes_data = results[0].boxes
            conf_list = boxes_data.conf.cpu().numpy() if boxes_data is not None else [0.8] * len(masks_data)

            for i, m in enumerate(masks_data):
                m_resized = cv2.resize((m > 0.5).astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
                # Intersect with road ROI so off-road puddles don't distort road coverage
                m_road = np.logical_and(m_resized, road_roi).astype(np.uint8)
                water_mask = np.logical_or(water_mask, m_road).astype(np.uint8)
                conf = float(conf_list[i]) if i < len(conf_list) else 0.80
                confidences.append(conf)

        total_water_pixels = int(water_mask.sum())
        coverage_pct = round((total_water_pixels / float(visible_road_pixels)) * 100.0, 2)

        # Multi-region extraction via connected components
        regions: List[WaterRegion] = []
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(water_mask, connectivity=8)

        max_reg_px = 0
        reg_id = 1
        for i in range(1, num_labels):  # Skip background (0)
            area = stats[i, cv2.CC_STAT_AREA]
            if area < 50:  # Filter minor noise
                continue
            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            rw = stats[i, cv2.CC_STAT_WIDTH]
            rh = stats[i, cv2.CC_STAT_HEIGHT]
            bbox = (x, y, x + rw, y + rh)
            reg_cov = round((area / float(visible_road_pixels)) * 100.0, 2)

            if area > max_reg_px:
                max_reg_px = area

            # Extract normalized polygon contour for region
            reg_mask = (labels == i).astype(np.uint8)
            contours, _ = cv2.findContours(reg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            poly_norm = []
            if contours:
                c_largest = max(contours, key=cv2.contourArea)
                c_approx = cv2.approxPolyDP(c_largest, 2.0, True)
                for pt in c_approx:
                    px_x, px_y = pt[0]
                    poly_norm.append([round(px_x / w, 4), round(px_y / h, 4)])

            regions.append(WaterRegion(
                region_id=reg_id,
                bounding_box=bbox,
                pixel_count=int(area),
                coverage_percent=reg_cov,
                confidence=round(float(np.mean(confidences)) if confidences else 0.85, 3),
                polygon=poly_norm if poly_norm else None,
            ))
            reg_id += 1

        has_water = len(regions) > 0 and total_water_pixels > 100
        severity = self.calculate_severity(coverage_pct, max_reg_px) if has_water else "NONE"
        mean_conf = round(float(np.mean(confidences)), 3) if confidences else 0.0
        max_conf = round(float(np.max(confidences)), 3) if confidences else 0.0
        elapsed_ms = round((time.time() - t_start) * 1000.0, 2)

        # Render visual overlay
        annotated = image.copy()
        if has_water:
            color_overlay = np.zeros_like(annotated)
            # SafeCity cyan-blue water mask (BGR: 255, 180, 0)
            color_overlay[water_mask == 1] = [255, 180, 0]
            cv2.addWeighted(color_overlay, overlay_alpha, annotated, 1.0 - overlay_alpha, 0, annotated)

            # Draw thin contour boundaries around each water region
            for r in regions:
                bx1, by1, bx2, by2 = r.bounding_box
                cv2.rectangle(annotated, (bx1, by1), (bx2, by2), (255, 140, 0), 1)
                label = f"Water #{r.region_id} ({r.coverage_percent:.1f}%)"
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
                cv2.rectangle(annotated, (bx1, max(by1 - th - 4, 0)), (bx1 + tw + 4, max(by1, th + 4)), (255, 140, 0), -1)
                cv2.putText(annotated, label, (bx1 + 2, max(by1 - 2, th + 2)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        return WaterSegmentationResult(
            has_water=has_water,
            total_water_regions=len(regions),
            regions=regions,
            total_water_pixels=total_water_pixels,
            visible_road_pixels=visible_road_pixels,
            water_coverage_percent=coverage_pct,
            severity=severity,
            mean_confidence=mean_conf,
            max_confidence=max_conf,
            processing_time_ms=elapsed_ms,
            mask=water_mask,
            annotated_image=annotated,
        )


class VideoWaterTemporalTracker:
    """
    Temporal confirmation & deduplication engine for video streams.
    Prevents alert flickering by grouping persistent water observations
    across consecutive frames into a single consolidated Road Hazard Event.
    """

    def __init__(self, persistence_frames_threshold: int = 3, max_frame_gap: int = 5):
        self.persistence_threshold = persistence_frames_threshold
        self.max_gap = max_frame_gap
        self.active_events = []
        self.finalized_events = []

    def update(self, frame_idx: int, timestamp_sec: float, result: WaterSegmentationResult, frame_bgr: np.ndarray):
        if not result.has_water:
            self._check_gap_closures(frame_idx)
            return

        # Check association with existing active event
        matched = False
        for ev in self.active_events:
            # Overlap in frame or proximity
            if frame_idx - ev["last_frame"] <= self.max_gap:
                ev["frames"].append(frame_idx)
                ev["timestamps"].append(timestamp_sec)
                ev["last_frame"] = frame_idx
                ev["coverages"].append(result.water_coverage_percent)
                ev["confidences"].append(result.mean_confidence)
                if result.water_coverage_percent > ev["max_coverage"]:
                    ev["max_coverage"] = result.water_coverage_percent
                    ev["representative_frame_idx"] = frame_idx
                    ev["representative_image"] = frame_bgr.copy()
                matched = True
                break

        if not matched:
            self.active_events.append({
                "start_frame": frame_idx,
                "last_frame": frame_idx,
                "frames": [frame_idx],
                "timestamps": [timestamp_sec],
                "coverages": [result.water_coverage_percent],
                "confidences": [result.mean_confidence],
                "max_coverage": result.water_coverage_percent,
                "representative_frame_idx": frame_idx,
                "representative_image": frame_bgr.copy(),
                "severity": result.severity,
            })

    def _check_gap_closures(self, current_frame: int):
        still_active = []
        for ev in self.active_events:
            if current_frame - ev["last_frame"] > self.max_gap:
                if len(ev["frames"]) >= self.persistence_threshold:
                    self.finalized_events.append(self._format_event(ev))
            else:
                still_active.append(ev)
        self.active_events = still_active

    def finalize(self) -> List[Dict[str, Any]]:
        for ev in self.active_events:
            if len(ev["frames"]) >= self.persistence_threshold:
                self.finalized_events.append(self._format_event(ev))
        self.active_events = []
        return self.finalized_events

    def _format_event(self, ev: dict) -> dict:
        mean_conf = round(float(np.mean(ev["confidences"])), 3)
        mean_cov = round(float(np.mean(ev["coverages"])), 2)
        return {
            "hazard_type": "water",
            "start_frame": ev["start_frame"],
            "end_frame": ev["last_frame"],
            "duration_frames": len(ev["frames"]),
            "start_time_sec": round(ev["timestamps"][0], 2),
            "end_time_sec": round(ev["timestamps"][-1], 2),
            "representative_frame": ev["representative_frame_idx"],
            "maximum_coverage_percent": ev["max_coverage"],
            "mean_coverage_percent": mean_cov,
            "mean_confidence": mean_conf,
            "severity": ev["severity"],
            "persisted_frames_count": len(ev["frames"]),
            "deduplication_status": "CONSOLIDATED_SINGLE_EVENT",
        }
