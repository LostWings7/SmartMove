"""
SafeCity Loop V2 — Unified Road-Context Service (Sections 7, 8, 25, 26)
Determines visible drivable road region, distinguishes road pavement from sky, clouds,
vegetation/plants, buildings, and vehicle structures, and computes transparent
road-overlap and road-context scores kept strictly separate from raw model confidence.

Key Design Principles:
1. Camera-Mode Adaptive: Automatically distinguishes forward-facing dashcam/street views
   (with horizon, sky, trees, buildings) from close-up/downward road surface views
   (where road pavement spans the full frame) so it NEVER blindly crops the upper half.
2. Chromatic & Structural Non-Road Exclusion: Explicitly masks out green vegetation/foliage,
   sky/clouds, above-horizon trees/buildings, and high-saturation/bright vehicle bodies
   while preserving dark pothole depressions, cracks, and water puddles inside the road.
3. Scientific Separation of Scores: Never fabricates or alters `raw_model_confidence`.
   Exposes `road_overlap_ratio` and `road_context_score` as explicit, auditable fields.
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple, List
import cv2
import numpy as np


@dataclass
class RoadContextAnalysis:
    camera_mode: str                  # "forward_facing", "close_up_surface", "top_down_overhead"
    horizon_y_ratio: float            # Estimated horizon line (0.0 for close-up, ~0.25-0.45 for forward-facing)
    road_mask: np.ndarray             # Binary uint8 mask (1 = valid road region, 0 = non-road)
    vegetation_mask: np.ndarray       # Binary uint8 mask (1 = plants/grass/trees/foliage)
    sky_bg_mask: np.ndarray           # Binary uint8 mask (1 = sky/clouds/above-horizon trees/buildings)
    structure_mask: np.ndarray        # Binary uint8 mask (1 = vehicles/buildings/high-sat non-road)
    visible_road_pixels: int
    road_coverage_of_frame: float

    @property
    def horizon_y(self) -> int:
        return int(self.horizon_y_ratio * self.road_mask.shape[0])


RoadContextResult = RoadContextAnalysis


@dataclass
class CandidateContextScore:
    raw_model_confidence: float
    road_overlap_ratio: float
    vegetation_overlap_ratio: float
    sky_bg_overlap_ratio: float
    structure_overlap_ratio: float
    center_on_road: bool
    road_context_score: float
    is_valid_road_candidate: bool
    rejection_reason: Optional[str] = None

    @property
    def final_acceptance_decision(self) -> str:
        return "ACCEPTED_ROAD_CONTEXT" if self.is_valid_road_candidate else (self.rejection_reason or "REJECTED_NON_ROAD")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "raw_model_confidence": round(float(self.raw_model_confidence), 4),
            "road_overlap_ratio": round(float(self.road_overlap_ratio), 4),
            "vegetation_overlap_ratio": round(float(self.vegetation_overlap_ratio), 4),
            "sky_bg_overlap_ratio": round(float(self.sky_bg_overlap_ratio), 4),
            "structure_overlap_ratio": round(float(self.structure_overlap_ratio), 4),
            "center_on_road": bool(self.center_on_road),
            "road_context_score": round(float(self.road_context_score), 4),
            "is_valid_road_candidate": bool(self.is_valid_road_candidate),
            "final_acceptance_decision": self.final_acceptance_decision,
            "rejection_reason": self.rejection_reason,
        }


class RoadContextService:
    """
    Lightweight, explainable road-region estimator and contextual gate consumed optionally
    by PotholeDetector, CrackDetector, and WaterSegmenter.
    """

    def __init__(
        self,
        min_road_overlap_pothole: float = 0.45,
        min_road_overlap_crack: float = 0.35,
        max_vegetation_overlap: float = 0.30,
        max_sky_bg_overlap: float = 0.25,
    ):
        self.min_road_overlap_pothole = min_road_overlap_pothole
        self.min_road_overlap_crack = min_road_overlap_crack
        self.max_vegetation_overlap = max_vegetation_overlap
        self.max_sky_bg_overlap = max_sky_bg_overlap

    def analyze_frame(self, img_bgr: np.ndarray, camera_hint: Optional[str] = None) -> RoadContextAnalysis:
        h, w = img_bgr.shape[:2]
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        b_ch, g_ch, r_ch = cv2.split(img_bgr.astype(np.int16))
        h_ch, s_ch, v_ch = cv2.split(hsv)

        # 1. Vegetation / Plants / Grass / Foliage Detection
        # Excess Green Index: ExG = 2*G - R - B, combined with HSV green/yellow-green hue
        exg = (2 * g_ch) - r_ch - b_ch
        hsv_green = (h_ch >= 24) & (h_ch <= 96) & (s_ch >= 38) & (v_ch >= 22)
        exg_green = (exg >= 20) & (g_ch >= r_ch) & (s_ch >= 30)
        veg_raw = (hsv_green | exg_green).astype(np.uint8)
        k_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        vegetation_mask = cv2.morphologyEx(veg_raw, cv2.MORPH_OPEN, k_small)
        vegetation_mask = cv2.morphologyEx(vegetation_mask, cv2.MORPH_CLOSE, k_small)

        # 2. Sky / Clouds / Upper Horizon Forest & Building Detection
        # Sky is bright low-saturation (clouds/overcast) or blue hue in the upper 50% of the frame
        y_grid = np.arange(h).reshape(h, 1) / float(max(h, 1))
        upper_half = y_grid < 0.48
        upper_third = y_grid < 0.34

        bright_sky = (v_ch >= 195) & (s_ch <= 50) & upper_half
        blue_sky = (h_ch >= 92) & (h_ch <= 135) & (s_ch >= 30) & (v_ch >= 130) & upper_half
        sky_raw = (bright_sky | blue_sky).astype(np.uint8)

        # Determine if image is forward-facing perspective vs close-up road surface
        upper_veg_ratio = float(vegetation_mask[: int(h * 0.38), :].mean()) if h > 10 else 0.0
        upper_sky_ratio = float(sky_raw[: int(h * 0.38), :].mean()) if h > 10 else 0.0

        # Check for dark forest/treeline or vertical building facade across the top 30% of the image
        # (e.g., REGRESSION_002 where top 35% is pitch-black forest treeline + bright sky)
        top_band = img_bgr[: int(h * 0.34), :]
        bot_band = img_bgr[int(h * 0.50) :, int(w * 0.20) : int(w * 0.80)]
        top_gray = cv2.cvtColor(top_band, cv2.COLOR_BGR2GRAY) if top_band.size > 0 else np.zeros((1, 1), dtype=np.uint8)
        bot_gray = cv2.cvtColor(bot_band, cv2.COLOR_BGR2GRAY) if bot_band.size > 0 else np.zeros((1, 1), dtype=np.uint8)

        # In REGRESSION_002, top band has extreme dark treeline (<40) + sky (>185) or strong dark-top/bright-road contrast
        top_extreme_dark_and_bright = (
            float((top_gray < 40).mean()) > 0.32
            and (float((top_gray > 185).mean()) > 0.04 or (float(bot_gray.mean()) - float(top_gray.mean())) > 35.0)
        )
        # Or strong colour/luminance discontinuity between top 30% and lower road center
        top_bot_diff = abs(float(top_gray.mean()) - float(bot_gray.mean()))

        if camera_hint in ("forward_facing", "close_up_surface", "top_down_overhead"):
            camera_mode = camera_hint
        elif (upper_veg_ratio + upper_sky_ratio) >= 0.14 or top_extreme_dark_and_bright:
            camera_mode = "forward_facing"
        else:
            camera_mode = "close_up_surface"

        # Estimate horizon line for forward-facing scenes
        horizon_y_ratio = 0.0
        sky_bg_mask = sky_raw.copy()

        if camera_mode == "forward_facing":
            # Scan rows from top to find where non-road upper background transitions to road corridor
            # Default conservative horizon for forward-facing street/rural camera
            horizon_y_ratio = 0.24
            if top_extreme_dark_and_bright:
                # Forest/skyline across top third (like REGRESSION_002)
                horizon_y_ratio = 0.36
                sky_bg_mask[: int(h * horizon_y_ratio), :] = 1
            else:
                # Mark above-horizon outer regions or upper sky/vegetation canopy
                sky_bg_mask[: int(h * 0.18), :] = 1

        # 3. Vehicle Body / Non-Road High-Saturation / White Roof Structure Detection
        # Detects bright white painted vehicle roofs/hoods (like SUV in REGRESSION_004) or vivid non-road objects
        vivid_non_road = (s_ch >= 110) & (v_ch >= 60) & (~hsv_green)
        # Bright metallic/white vehicle body panels with high local edge contrast
        bright_panel = (v_ch >= 222) & (s_ch <= 22) & (~bright_sky)
        structure_raw = (vivid_non_road).astype(np.uint8)

        # Check for large rectangular vehicle body in scene (like REGRESSION_004 overhead SUV)
        panel_cnts, _ = cv2.findContours(bright_panel.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in panel_cnts:
            c_area = cv2.contourArea(cnt)
            if 0.015 * (h * w) < c_area < 0.35 * (h * w):
                rx, ry, rw, rh = cv2.boundingRect(cnt)
                # Fill vehicle bounding hull (including sunroof/windows inside the white car body!)
                cv2.rectangle(structure_raw, (rx, ry), (rx + rw, ry + rh), 1, -1)

        structure_mask = cv2.morphologyEx(structure_raw, cv2.MORPH_CLOSE, k_small)

        # 4. Assemble Drivable Road Region Mask
        # Start with camera-mode geometric prior
        geo_prior = np.zeros((h, w), dtype=np.uint8)
        if camera_mode == "forward_facing":
            hy = int(h * horizon_y_ratio)
            pts = np.array([
                [int(w * 0.15), hy],
                [int(w * 0.85), hy],
                [w, int(h * 0.58)],
                [w, h],
                [0, h],
                [0, int(h * 0.58)],
            ], dtype=np.int32)
            cv2.fillPoly(geo_prior, [pts], 1)
        else:
            geo_prior[:, :] = 1

        # Exclude vegetation, sky/horizon background, and vehicle/building structures
        non_road_combined = (vegetation_mask > 0) | (sky_bg_mask > 0) | (structure_mask > 0)
        road_mask = ((geo_prior > 0) & (~non_road_combined)).astype(np.uint8)

        # Fill small internal holes inside the road surface (so dark potholes or specular puddles inside road stay 1)
        k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        road_mask = cv2.morphologyEx(road_mask, cv2.MORPH_CLOSE, k_close)
        # Re-apply strict exclusion of definite green vegetation and upper sky/horizon
        road_mask[vegetation_mask > 0] = 0
        road_mask[sky_bg_mask > 0] = 0
        road_mask[structure_mask > 0] = 0

        vis_pixels = int(road_mask.sum())
        if vis_pixels < int(0.05 * h * w):
            # Fallback safety if extreme lighting triggers over-masking
            road_mask = ((geo_prior > 0) & (vegetation_mask == 0) & (sky_bg_mask == 0)).astype(np.uint8)
            vis_pixels = max(int(road_mask.sum()), 1)

        return RoadContextAnalysis(
            camera_mode=camera_mode,
            horizon_y_ratio=round(horizon_y_ratio, 3),
            road_mask=road_mask,
            vegetation_mask=vegetation_mask,
            sky_bg_mask=sky_bg_mask,
            structure_mask=structure_mask,
            visible_road_pixels=vis_pixels,
            road_coverage_of_frame=round(vis_pixels / float(max(h * w, 1)), 4),
        )

    def evaluate_box(
        self,
        box_xyxy: Tuple[float, float, float, float],
        raw_confidence: Optional[float] = None,
        context: Optional[RoadContextAnalysis] = None,
        hazard_type: str = "pothole",
        raw_model_confidence: Optional[float] = None,
    ) -> CandidateContextScore:
        actual_conf = raw_model_confidence if raw_model_confidence is not None else (raw_confidence if raw_confidence is not None else 0.5)
        if context is None:
            raise ValueError("context (RoadContextAnalysis) must be provided")
        """
        Evaluates whether a candidate bounding box lies on valid road pavement.
        Keeps `raw_model_confidence` strictly separate from `road_context_score`.
        """
        h, w = context.road_mask.shape[:2]
        x1 = max(0, min(w - 1, int(round(box_xyxy[0]))))
        y1 = max(0, min(h - 1, int(round(box_xyxy[1]))))
        x2 = max(x1 + 1, min(w, int(round(box_xyxy[2]))))
        y2 = max(y1 + 1, min(h, int(round(box_xyxy[3]))))

        box_area = float(max((x2 - x1) * (y2 - y1), 1))
        road_sub = context.road_mask[y1:y2, x1:x2]
        veg_sub = context.vegetation_mask[y1:y2, x1:x2]
        sky_sub = context.sky_bg_mask[y1:y2, x1:x2]
        struct_sub = context.structure_mask[y1:y2, x1:x2]

        road_overlap = float(road_sub.sum()) / box_area
        veg_overlap = float(veg_sub.sum()) / box_area
        sky_overlap = float(sky_sub.sum()) / box_area
        struct_overlap = float(struct_sub.sum()) / box_area

        cx = max(0, min(w - 1, int((x1 + x2) // 2)))
        cy = max(0, min(h - 1, int((y1 + y2) // 2)))
        center_on_road = bool(context.road_mask[cy, cx] > 0)

        # Compute transparent road_context_score in [0.0, 1.0]
        penalty = (veg_overlap * 1.2) + (sky_overlap * 1.5) + (struct_overlap * 1.2)
        road_context_score = max(0.0, min(1.0, road_overlap - 0.5 * penalty + (0.10 if center_on_road else -0.10)))

        min_req = self.min_road_overlap_pothole if hazard_type == "pothole" else self.min_road_overlap_crack
        rejection_reason = None
        is_valid = True

        if sky_overlap >= self.max_sky_bg_overlap:
            is_valid = False
            rejection_reason = "sky_or_above_horizon_background"
        elif veg_overlap >= self.max_vegetation_overlap:
            is_valid = False
            rejection_reason = "vegetation_or_plants"
        elif struct_overlap >= 0.30:
            is_valid = False
            rejection_reason = "vehicle_or_non_road_structure"
        elif road_overlap < min_req:
            is_valid = False
            rejection_reason = "insufficient_road_surface_overlap"

        return CandidateContextScore(
            raw_model_confidence=actual_conf,
            road_overlap_ratio=road_overlap,
            vegetation_overlap_ratio=veg_overlap,
            sky_bg_overlap_ratio=sky_overlap,
            structure_overlap_ratio=struct_overlap,
            center_on_road=center_on_road,
            road_context_score=road_context_score,
            is_valid_road_candidate=is_valid,
            rejection_reason=rejection_reason,
        )


road_context_service = RoadContextService()
