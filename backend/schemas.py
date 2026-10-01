"""
SafeCity Loop V2 — Pydantic Schemas
Response and request models for all GovTech municipal intelligence endpoints.
"""
from datetime import datetime
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, ConfigDict


# ── Road / RoadSegment ────────────────────────────────────────────────────────

class RoadBase(BaseModel):
    name: str
    latitude: float
    longitude: float
    risk_score: float
    safe_city_score: float
    traffic_exposure: str
    vulnerability: str
    status: str


class RoadOut(RoadBase):
    model_config = ConfigDict(from_attributes=True)
    id: int
    road_type: Optional[str] = "urban_arterial"
    vulnerability_score: Optional[float] = 50.0
    importance_score: Optional[float] = 50.0
    risk_confidence: Optional[float] = 0.85
    data_coverage: Optional[float] = 75.0
    last_observed: Optional[datetime] = None
    geometry: Optional[List[List[float]]] = None


RoadSegmentOut = RoadOut


# ── Hazard / HazardObservation ────────────────────────────────────────────────

class HazardBase(BaseModel):
    road_id: int
    type: str
    severity: str
    confidence: float
    risk_score: float
    latitude: float
    longitude: float
    detected_at: datetime
    status: str


class HazardOut(HazardBase):
    model_config = ConfigDict(from_attributes=True)
    id: int
    gps_accuracy: Optional[float] = 4.5
    image_path: Optional[str] = None
    source: Optional[str] = "AI Vision"
    verified: Optional[bool] = False
    evidence_id: Optional[str] = None


HazardObservationOut = HazardOut


# ── NearMiss / ConflictEvent ──────────────────────────────────────────────────

class NearMissBase(BaseModel):
    junction_id: Optional[int] = None
    road_id: Optional[int] = None
    timestamp: datetime
    object_types: List[str]
    ttc: Optional[float] = None
    risk_level: str
    conflict_zone: Optional[str] = None


class NearMissOut(NearMissBase):
    model_config = ConfigDict(from_attributes=True)
    id: int
    video_id: Optional[str] = None
    object_type_a: Optional[str] = "vehicle"
    object_type_b: Optional[str] = "pedestrian"
    pet: Optional[float] = None
    minimum_distance: Optional[float] = 2.5
    event_severity: Optional[str] = "HIGH"
    evidence_clip: Optional[str] = None
    review_status: Optional[str] = "pending_review"


ConflictEventOut = NearMissOut


class ConflictReviewUpdate(BaseModel):
    review_status: str   # pending_review / reviewed / false_positive / actioned
    notes: Optional[str] = None


# ── Junction ──────────────────────────────────────────────────────────────────

class JunctionBase(BaseModel):
    name: str
    latitude: float
    longitude: float
    risk_score: float
    camera_id: Optional[str] = None
    status: str = "active"


class JunctionOut(JunctionBase):
    model_config = ConfigDict(from_attributes=True)
    id: int
    near_misses: List[NearMissOut] = []


# ── Repair / RepairRecord ─────────────────────────────────────────────────────

class RepairBase(BaseModel):
    road_id: int
    priority: int
    reason: Optional[str] = None
    status: str
    assigned_to: Optional[str] = None
    assigned_department: Optional[str] = "Municipal Public Works"
    recommended_action: Optional[str] = "Asphalt Resurfacing"
    approval_status: Optional[str] = "approved"
    before_score: Optional[float] = None
    after_score: Optional[float] = None


class RepairOut(RepairBase):
    model_config = ConfigDict(from_attributes=True)
    id: int
    road: Optional[RoadOut] = None
    repair_date: Optional[datetime] = None
    evidence_image: Optional[str] = None
    created_at: datetime
    updated_at: Optional[datetime] = None


RepairRecordOut = RepairOut


# ── Intervention ──────────────────────────────────────────────────────────────

class InterventionOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    road_id: Optional[int]
    name: str
    type: Optional[str]
    before_risk: Optional[float]
    after_risk: Optional[float]
    before_near_misses: int
    after_near_misses: int
    implemented_at: Optional[datetime]
    notes: Optional[str]


# ── CitizenReport ─────────────────────────────────────────────────────────────

class CitizenReportCreate(BaseModel):
    reporter_name: Optional[str] = None
    type: str
    description: Optional[str] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    severity: str = "MEDIUM"
    photo_url: Optional[str] = None


class CitizenReportOut(CitizenReportCreate):
    model_config = ConfigDict(from_attributes=True)
    id: int
    report_code: Optional[str] = None
    status: str
    submitted_at: datetime


class CitizenReportStatusUpdate(BaseModel):
    status: str  # Submitted / Under Review / Verified / Rejected / Converted to Hazard
    reviewer: Optional[str] = "Municipal Officer"
    notes: Optional[str] = None


class DecisionAuditOut(BaseModel):
    evidence_id: str
    source: str
    date: str
    location: str
    ai_result: str
    confidence: float
    reviewer: str
    decision: str
    status: str


# ── EvidenceFile & AuditLog ───────────────────────────────────────────────────

class EvidenceFileOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    evidence_code: str
    road_id: Optional[int] = None
    hazard_id: Optional[int] = None
    conflict_id: Optional[int] = None
    file_type: str
    file_url: str
    thumbnail_url: Optional[str] = None
    captured_at: datetime
    metadata_json: Dict[str, Any] = {}
    verified: bool = False
    notes: Optional[str] = None


class AuditLogOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    timestamp: datetime
    actor: str
    action: str
    target_type: str
    target_id: Optional[int] = None
    details: Optional[str] = None


# ── RiskSnapshots ─────────────────────────────────────────────────────────────

class RiskSnapshotOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    road_id: int
    timestamp: datetime
    risk_score: float
    confidence: float
    data_coverage: float
    hazard_severity: float
    traffic_exposure: float
    conflict_evidence: float
    vulnerable_exposure: float
    persistence: float
    road_importance: float
    contributing_factors: Dict[str, Any] = {}


# ── Command Center Dashboard & KPIs ───────────────────────────────────────────

class KPIStats(BaseModel):
    roads_monitored: int
    high_concern_segments: int = 0
    high_risk_roads: int = 0
    active_hazards: int = 0
    potholes_detected: int = 0
    conflict_events: int = 0
    near_misses: int = 0
    danger_zones: int = 0
    pending_repairs: int = 0
    low_confidence_segments: int = 0
    completed_repairs: int = 0


CommandCenterKPIs = KPIStats


class RiskTrendPoint(BaseModel):
    date: str
    average_risk: float
    high_risk_count: int


class DataCoverageStats(BaseModel):
    high_coverage_count: int
    moderate_coverage_count: int
    low_coverage_count: int
    average_coverage_pct: float


class LegacyAIEvent(BaseModel):
    time: str
    event: str
    type: str


class AIEvent(BaseModel):
    id: Optional[str] = None
    type: str                      # "hazard" or "near_miss"
    title: Optional[str] = None
    description: Optional[str] = None
    severity: Optional[str] = "MEDIUM"
    location: Optional[str] = None
    timestamp: Optional[datetime] = None
    risk_score: Optional[float] = 0.0
    image_url: Optional[str] = None
    time: Optional[str] = None
    event: Optional[str] = None


class DashboardOut(BaseModel):
    kpi: KPIStats
    recent_events: List[Any]
    top_repairs: List[RepairOut]
    interventions: List[InterventionOut]
    risk_trend: List[RiskTrendPoint] = []
    data_coverage_stats: Optional[DataCoverageStats] = None
    recent_conflicts: List[ConflictEventOut] = []


DashboardData = DashboardOut


# ── Road Intelligence Dossier ─────────────────────────────────────────────────

class DataQualityOut(BaseModel):
    last_observation: Optional[str]
    last_observation_hours_ago: float
    observation_count: int
    hazard_count: int
    conflict_count: int
    gps_accuracy_meters: float
    model_confidence: float
    confidence_tier: str
    data_freshness: str
    coverage_percentage: float
    coverage_tier: str
    quality_warning: Optional[str] = None
    audit_disclaimer: str


class ContributingFactorDetail(BaseModel):
    score: float
    weight: float
    weighted_contribution: float
    label: str


class RoadSegmentDossierOut(BaseModel):
    segment_id: int
    road_name: str
    road_type: str
    calculated_risk: float
    risk_score: Optional[float] = None
    safe_city_score: float
    classification: str
    is_danger_zone: bool
    confidence: float
    data_coverage: float
    traffic_exposure: str
    vulnerable_user_exposure: str
    repair_status: str
    suggested_intervention: str
    quality: DataQualityOut
    contributing_factors: Dict[str, ContributingFactorDetail]
    contributors_summary: Optional[List[Dict[str, Any]]] = []
    hazards: List[HazardOut]
    conflicts: List[ConflictEventOut]
    evidence_timeline: List[EvidenceFileOut]
    repairs: List[RepairOut]
    risk_history: List[RiskSnapshotOut]
    disclaimer: str


# ── AI Vision: Pothole & Traffic Analysis ─────────────────────────────────────

class PotholeDetectionBoundingBoxOut(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float
    width: float
    height: float


BoundingBoxOut = PotholeDetectionBoundingBoxOut


class PotholeDetectionItemOut(BaseModel):
    box: PotholeDetectionBoundingBoxOut
    confidence: float
    raw_model_confidence: Optional[float] = None
    road_overlap_ratio: Optional[float] = None
    road_context_score: Optional[float] = None
    final_acceptance_decision: Optional[str] = "ACCEPTED"
    class_name: str
    severity: str
    risk_score: float
    is_demo: bool = False
    mask_polygon: Optional[List[List[float]]] = None
    area_sq_meters: Optional[float] = None
    shape_type: Optional[str] = "bounding_box"
    depth_estimate_cm: Optional[float] = None
    severity_category: Optional[str] = None


class PotholeAnalysisOut(BaseModel):
    success: bool = True
    image_id: Optional[str] = None
    is_demo_mode: bool
    is_precomputed_demo: Optional[bool] = False
    model_status: str
    status_message: str
    model_path: str
    pothole_count: int
    detection_count: Optional[int] = 0
    confidence: float
    average_confidence: Optional[float] = 0.0
    processing_time_ms: Optional[int] = 0
    processing_time_sec: Optional[float] = 0.0
    visual_severity: Optional[str] = "HIGH"
    contextual_severity: Optional[str] = "HIGH"
    highest_severity: Optional[str] = "HIGH"
    traffic_exposure: Optional[str] = "HIGH"
    vulnerable_users: Optional[str] = "HIGH"
    persistence: Optional[str] = "MEDIUM"
    severity: str
    risk_score: float
    risk_confidence: Optional[str] = "HIGH"
    risk_formula: str
    detections: List[PotholeDetectionItemOut]
    original_image_url: str
    processed_image_url: str
    image_width: Optional[int] = None
    image_height: Optional[int] = None
    hazard_id: Optional[int] = None
    road_id: Optional[int] = None
    road_name: Optional[str] = None
    evidence_id: Optional[str] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    gps_source: Optional[str] = "Location unavailable"
    gps_accuracy: Optional[float] = 2.5
    is_simulated_gps: Optional[bool] = False
    timestamp: Optional[str] = None
    disclaimer: Optional[str] = "AI detection confidence measures visual pattern recognition accuracy only; it does not represent collision probability. NO DATA ≠ SAFE ROAD."


class TrafficConflictDetailOut(BaseModel):
    conflict_id: str
    timestamp_str: str
    timestamp_seconds: float
    object_type_a: Optional[str] = "vehicle"
    object_type_b: Optional[str] = "pedestrian"
    object_types: List[str]
    track_ids: List[int]
    ttc: float
    pet: Optional[float] = None
    minimum_distance: Optional[float] = 2.5
    risk_level: str
    event_severity: Optional[str] = "HIGH"
    conflict_zone: str
    direction: Optional[str] = None
    frame_index: int
    confidence: Optional[float] = 0.88
    snapshot_url: Optional[str] = None
    review_status: Optional[str] = "pending_review"
    disclaimer: Optional[str] = "Approximate monocular surrogate safety estimate. Potential traffic conflict."
    drac: Optional[float] = None
    relative_speed_kmh: Optional[float] = None
    interaction_geometry: Optional[str] = None
    trajectory_convergence: Optional[bool] = None
    why_conflict_explanation: Optional[Dict[str, Any]] = None


class ConflictHotspotOut(BaseModel):
    corridor_id: int
    corridor_name: str
    location_type: str            # "Corridor" or "Junction"
    conflict_count: int
    peak_period: str              # e.g. "5:00 PM – 8:00 PM"
    primary_interaction: str      # e.g. "Motorcycle ⟷ Pedestrian"
    calculated_concern: str       # "VERY HIGH", "HIGH", "MODERATE"
    data_confidence: str          # "HIGH", "MEDIUM", "LOW"
    mean_ttc: float               # e.g. 1.25s
    label: str = "Recurring Conflict Hotspot"


class TrafficAnalysisOut(BaseModel):
    success: bool = True
    video_id: Optional[str] = None
    is_demo_mode: bool
    is_demo_video: Optional[bool] = False
    is_precomputed_demo: Optional[bool] = False
    model_status: str
    status_message: str
    junction_id: Optional[int]
    junction_name: Optional[str] = None
    road_id: Optional[int] = None
    duration_seconds: float
    total_frames: int
    processed_frames: int
    processing_time_sec: Optional[float] = 0.0
    tracked_objects_count: int
    object_class_counts: Dict[str, int]
    near_miss_count: int
    near_misses: List[TrafficConflictDetailOut]
    timeline: Optional[List[Dict[str, Any]]] = []
    original_video_url: str
    processed_video_url: Optional[str]
    conflict_snapshots: List[str]
    created_near_miss_ids: List[int] = []
    privacy_applied: Optional[bool] = True
    privacy_notice: Optional[str] = "Video processing is intended to minimize unnecessary storage of personally identifiable visual information (faces and license plates anonymized)."
    tracked_objects_summary: Optional[List[Dict[str, Any]]] = []
    exposure_normalized_rates: Optional[Dict[str, Any]] = None
    traffic_model_meta: Optional[Dict[str, Any]] = None
    tracking_health_status: Optional[str] = "ANALYSIS_HEALTHY"
    metric_calibration_status: Optional[str] = "NOMINAL_PIXEL_ESTIMATE"
    tracking_health_summary: Optional[Dict[str, Any]] = None


class AIModelStatusOut(BaseModel):
    pothole_model_path: str
    pothole_model_loaded: bool
    traffic_model_path: str
    traffic_model_loaded: bool
    is_pothole_demo_mode: bool
    is_traffic_demo_mode: bool
    message: str
    model_name: Optional[str] = "SafeCity YOLO26s Pothole Detector"
    model_version: Optional[str] = "1.0.0"
    framework: Optional[str] = "Ultralytics YOLO / OpenVINO"
    device: Optional[str] = "CPU"
    confidence_threshold: Optional[float] = 0.35
    classes: Optional[List[str]] = ["pothole"]
    traffic_model_name: Optional[str] = "SafeCity Traffic YOLO26s v1"
    traffic_model_version: Optional[str] = "1.0.0"
    traffic_tracker: Optional[str] = "ByteTrack"
    traffic_classes: Optional[List[str]] = ["pedestrian", "rider", "car", "truck", "bus", "motorcycle", "bicycle"]
    traffic_runtime: Optional[str] = "PyTorch / OpenVINO"
    water_model_path: Optional[str] = None
    water_model_loaded: Optional[bool] = False
    crack_model_path: Optional[str] = None
    crack_model_loaded: Optional[bool] = False
    road_condition_capabilities: Optional[Dict[str, Any]] = None


# ── Road Condition Intelligence: Water & Crack Subsystems ────────────────────

class WaterRegionOut(BaseModel):
    region_id: int
    area_pixels: Optional[int] = None
    pixel_count: Optional[int] = None
    coverage_percent: float
    bbox: Dict[str, float]
    severity: Optional[str] = "MODERATE"
    confidence: Optional[float] = None
    polygon: Optional[List[List[float]]] = None


class WaterFilledPotholeEvidenceOut(BaseModel):
    pothole_confidence: float
    water_confidence: float
    overlap_ratio: float
    overlap_percent: float
    water_pixels_in_pothole: int
    total_pothole_pixels: int
    evidence_status: str = "CONFIRMED_FUSION"
    fusion_formula: str


class RoadCrackDetectionItemOut(BaseModel):
    crack_type: str
    confidence: float
    raw_model_confidence: Optional[float] = None
    road_overlap_ratio: Optional[float] = None
    road_context_score: Optional[float] = None
    final_acceptance_decision: Optional[str] = "ACCEPTED"
    severity: str
    risk_score: float
    bbox: Dict[str, float]
    mask_polygon: Optional[List[List[float]]] = None
    estimated_visible_area_sq_m: Optional[float] = None


class RoadConditionObservationOut(BaseModel):
    observation_id: str
    hazard_type: str
    confidence: float
    raw_model_confidence: Optional[float] = None
    road_overlap_ratio: Optional[float] = None
    road_context_score: Optional[float] = None
    final_acceptance_decision: Optional[str] = "ACCEPTED"
    severity: str
    risk_score: float
    bbox: Optional[Dict[str, float]] = None
    mask_polygon: Optional[List[List[float]]] = None
    coverage_percent: Optional[float] = None
    source_model: str
    model_version: str
    timestamp: str
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    evidence_reference: Optional[str] = None
    fusion_evidence: Optional[Dict[str, Any]] = None


class RoadConditionAnalysisOut(BaseModel):
    success: bool = True
    image_id: str
    total_hazards_count: int
    pothole_count: int
    water_regions_count: int
    crack_count: int
    longitudinal_crack_count: int = 0
    transverse_crack_count: int = 0
    alligator_crack_count: int = 0
    water_filled_pothole_count: int = 0
    water_coverage_percent: float = 0.0
    water_severity: str = "NONE"
    highest_severity: str = "NONE"
    overall_risk_score: float = 0.0
    risk_confidence: str = "HIGH"
    observations: List[RoadConditionObservationOut] = []
    water_regions: List[WaterRegionOut] = []
    water_filled_potholes: List[Dict[str, Any]] = []
    cracks: List[RoadCrackDetectionItemOut] = []
    potholes: List[Dict[str, Any]] = []
    disambiguation_audit: Optional[List[Dict[str, Any]]] = []
    road_context_summary: Optional[Dict[str, Any]] = None
    original_image_url: str
    processed_image_url: str
    layer_urls: Dict[str, str] = {}
    model_status: Dict[str, Any] = {}
    processing_time_ms: int = 0
    evidence_id: Optional[str] = None
    hazard_id: Optional[int] = None
    road_id: Optional[int] = None
    road_name: Optional[str] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    gps_source: Optional[str] = "Location unavailable"
    timestamp: Optional[str] = None
    disclaimer: str


# ── Danger Zones & Repairs ────────────────────────────────────────────────────

class SafeCityFactorsOut(BaseModel):
    pothole_risk: str
    junction_risk: str
    traffic_exposure: str
    pedestrian_exposure: str
    pothole_score: float
    junction_score: float
    traffic_score: float
    vulnerability_score: float
    report_count: int
    pothole_severity: Optional[str] = "MEDIUM"
    hazard_frequency: Optional[int] = 0
    conflict_frequency: Optional[int] = 0
    conflict_severity: Optional[str] = "MEDIUM"
    motorcycle_exposure: Optional[str] = "MEDIUM"
    persistence_score: Optional[float] = 0.0
    road_importance_score: Optional[float] = 50.0
    data_confidence: Optional[str] = "HIGH"
    main_contributing_factor: Optional[str] = "Observed Surface Defects"


class RoadSafetyEvaluationOut(BaseModel):
    road_id: int
    road_name: str
    danger_zone_score: float
    danger_zone_classification: str
    safe_city_score: float
    factors: SafeCityFactorsOut
    conflict_count: Optional[int] = 0
    hazard_count: Optional[int] = 0
    traffic_exposure: Optional[str] = "HIGH"
    main_contributing_factor: Optional[str] = "Observed Surface Defects"
    confidence: Optional[str] = "HIGH"
    coverage: Optional[str] = "HIGH COVERAGE"
    last_observed: Optional[str] = None
    is_hotspot: Optional[bool] = False
    quality_warning: Optional[str] = None
    disclaimer: str


class PrioritizedRepairItemOut(BaseModel):
    road_id: int
    road_name: str
    repair_id: Optional[int] = None
    priority_level: str
    priority_rank: int
    urgency_score: float
    hazard_summary: str
    risk_score: float
    status: str
    assigned_to: Optional[str] = None
    assigned_department: Optional[str] = "Municipal Public Works"
    recommended_action: Optional[str] = "Asphalt Resurfacing"
    approval_status: Optional[str] = "approved"
    reasons: List[str]
    reason: Optional[str] = None
    suggested_action: Optional[str] = "Road-surface inspection + pedestrian-safety review"
    evidence_ids: Optional[List[str]] = []
    human_approval_required: bool = True
    created_at: Optional[str] = None


class RepairStatusUpdate(BaseModel):
    status: Optional[str] = "NEW"  # NEW / UNDER REVIEW / VERIFIED / REPAIR ASSIGNED / REPAIRED / POST-REPAIR MONITORING / CLOSED
    action: Optional[str] = None # verify / approve / assign / reject / mark_repaired / reopen
    assigned_to: Optional[str] = None
    assigned_department: Optional[str] = None
    approval_status: Optional[str] = None
    reviewer: Optional[str] = "Municipal Officer"
    notes: Optional[str] = None


# ── Route Intelligence ────────────────────────────────────────────────────────

class RouteWaypointOut(BaseModel):
    latitude: float
    longitude: float
    name: str
    road_id: Optional[int] = None
    risk_score: float


class RouteOptionOut(BaseModel):
    route_type: str
    route_key: str
    distance_km: float
    time_minutes: float
    pothole_risk: str
    junction_risk: str
    overall_calculated_risk: float
    risk_classification: str
    description: str
    waypoints: List[RouteWaypointOut]
    risk_penalty: Optional[float] = 0.0
    route_cost: Optional[float] = 0.0
    high_risk_segments_count: Optional[int] = 0
    hazards_count: Optional[int] = 0
    conflict_hotspots_count: Optional[int] = 0
    explanation: Optional[str] = None
    maneuvers: Optional[List[Dict[str, Any]]] = None


class RouteComparisonResultOut(BaseModel):
    origin_name: str
    destination_name: str
    routes: List[RouteOptionOut]
    neutral_advisory: str


class JunctionSimulationRequest(BaseModel):
    conflict_type: str = "pedestrian-vehicle"
    ttc: float = 1.3
    risk_level: str = "HIGH"
    reset: bool = False


class SmartJunctionDisplayStateOut(BaseModel):
    junction_id: int
    junction_name: str
    display_mode: str
    signal_state: str
    countdown_seconds: int
    pedestrian_warning: bool
    road_hazard_warning: bool
    ai_risk_state: str
    risk_score: float
    active_near_misses_count: int
    recent_conflict: Optional[TrafficConflictDetailOut] = None


# ══════════════════════════════════════════════════════════════════════════════
# CITIZEN ACCOUNT, AUTH & COMPLAINT WORKFLOW SCHEMAS
# ══════════════════════════════════════════════════════════════════════════════

# ── User & Auth ───────────────────────────────────────────────────────────────

class UserRegisterRequest(BaseModel):
    full_name: str
    email: str
    password: str
    phone_number: Optional[str] = None
    preferred_language: Optional[str] = "en"


class UserLoginRequest(BaseModel):
    email: str
    password: str


class UserOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    full_name: str
    email: str
    phone_number: Optional[str] = None
    role: str
    is_active: bool
    is_verified: bool
    created_at: datetime
    last_login: Optional[datetime] = None
    profile_photo: Optional[str] = None
    preferred_language: Optional[str] = "en"
    notification_preferences: Optional[Dict[str, Any]] = None


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: UserOut


class UserProfileUpdate(BaseModel):
    full_name: Optional[str] = None
    phone_number: Optional[str] = None
    preferred_language: Optional[str] = None
    notification_preferences: Optional[Dict[str, Any]] = None


class ChangePasswordRequest(BaseModel):
    current_password: str
    new_password: str


class UserRoleUpdateRequest(BaseModel):
    role: str


class UserStatusUpdateRequest(BaseModel):
    is_active: bool


# ── Complaint Categories ──────────────────────────────────────────────────────

class ComplaintCategoryBase(BaseModel):
    name: str
    code: str
    description: Optional[str] = None
    is_ai_supported: bool = False
    is_active: bool = True


class ComplaintCategoryOut(ComplaintCategoryBase):
    model_config = ConfigDict(from_attributes=True)
    id: int


class ComplaintCategoryCreate(ComplaintCategoryBase):
    pass


# ── Evidence & Status History ─────────────────────────────────────────────────

class ComplaintEvidenceOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    complaint_id: int
    uploader_id: int
    file_type: str
    file_url: str
    filename: str
    mime_type: str
    file_size: int
    uploaded_at: datetime
    privacy_status: str
    notes: Optional[str] = None


class ComplaintStatusHistoryOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    complaint_id: int
    old_status: Optional[str] = None
    new_status: str
    changed_by_name: str
    changed_by_role: str
    comment: Optional[str] = None
    timestamp: datetime


# ── Complaint Creation & Views ────────────────────────────────────────────────

class ComplaintCreateRequest(BaseModel):
    category_id: int
    title: str
    description: str
    location_lat: float
    location_lng: float
    location_address: Optional[str] = None
    priority: Optional[str] = "MEDIUM"


class CitizenComplaintOut(BaseModel):
    """Citizen-safe complaint model: internal notes and officer identities are protected."""
    model_config = ConfigDict(from_attributes=True)
    id: int
    public_reference: str
    category: ComplaintCategoryOut
    title: str
    description: str
    status: str
    priority: str
    location_lat: float
    location_lng: float
    location_address: Optional[str] = None
    assigned_department: Optional[str] = None
    created_at: datetime
    updated_at: Optional[datetime] = None
    resolved_at: Optional[datetime] = None
    closed_at: Optional[datetime] = None
    citizen_confirmed_at: Optional[datetime] = None
    resolution_summary: Optional[str] = None
    public_response: Optional[str] = None
    evidence_items: List[ComplaintEvidenceOut] = []
    status_history: List[ComplaintStatusHistoryOut] = []
    can_confirm_resolution: bool = False
    can_reopen: bool = False


class AdminComplaintOut(BaseModel):
    """Authoritative administrative complaint view with full operational context."""
    model_config = ConfigDict(from_attributes=True)
    id: int
    public_reference: str
    citizen_id: int
    citizen_name: str
    citizen_email: str
    citizen_phone: Optional[str] = None
    category: ComplaintCategoryOut
    title: str
    description: str
    status: str
    priority: str
    location_lat: float
    location_lng: float
    location_address: Optional[str] = None
    road_segment_id: Optional[int] = None
    road_name: Optional[str] = None
    assigned_to: Optional[int] = None
    assigned_to_name: Optional[str] = None
    assigned_department: Optional[str] = None
    source: str
    created_at: datetime
    updated_at: Optional[datetime] = None
    resolved_at: Optional[datetime] = None
    closed_at: Optional[datetime] = None
    citizen_confirmed_at: Optional[datetime] = None
    resolution_summary: Optional[str] = None
    public_response: Optional[str] = None
    internal_notes: Optional[str] = None
    ai_linked: bool = False
    ai_confidence: Optional[float] = None
    hazard_id: Optional[int] = None
    repair_id: Optional[int] = None
    evidence_items: List[ComplaintEvidenceOut] = []
    status_history: List[ComplaintStatusHistoryOut] = []
    duplicate_candidates_count: int = 0


# ── Complaint Operations ──────────────────────────────────────────────────────

class ComplaintStatusUpdateRequest(BaseModel):
    new_status: str
    comment: Optional[str] = None


class ComplaintAssignRequest(BaseModel):
    assigned_to: Optional[int] = None
    assigned_department: Optional[str] = None
    notes: Optional[str] = None


class ComplaintPublicResponseRequest(BaseModel):
    public_response: str


class ComplaintInternalNoteRequest(BaseModel):
    internal_note: str


class CitizenResolutionConfirmRequest(BaseModel):
    confirmed: bool
    comment: Optional[str] = None


class ComplaintLinkHazardRequest(BaseModel):
    hazard_id: int
    notes: Optional[str] = None


class ComplaintCreateHazardRequest(BaseModel):
    road_id: Optional[int] = None
    severity: Optional[str] = "HIGH"
    notes: Optional[str] = None


class ComplaintLinkRepairRequest(BaseModel):
    repair_id: int
    notes: Optional[str] = None


class DuplicateCandidateOut(BaseModel):
    complaint_id: int
    public_reference: str
    title: str
    status: str
    distance_meters: float
    similarity_score: float
    created_at: datetime


class AIHazardCandidateOut(BaseModel):
    hazard_id: int
    road_id: int
    road_name: Optional[str] = None
    type: str
    severity: str
    risk_score: float
    distance_meters: float
    confidence: float
    image_path: Optional[str] = None
    detected_at: datetime


# ── Notifications ─────────────────────────────────────────────────────────────

class NotificationOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    type: str
    title: str
    message: str
    related_complaint_id: Optional[int] = None
    is_read: bool
    created_at: datetime


class NotificationCountOut(BaseModel):
    unread_count: int


# ── Dashboards & Pagination ───────────────────────────────────────────────────

class CitizenDashboardOut(BaseModel):
    welcome_name: str
    total_complaints: int
    active_complaints: int
    pending_review: int
    in_progress: int
    resolved: int
    closed: int
    recent_complaints: List[CitizenComplaintOut]
    recent_notifications: List[NotificationOut]


class AdminDashboardOut(BaseModel):
    total_complaints: int
    new_today: int
    under_review: int
    assigned: int
    in_progress: int
    resolved: int
    reopened: int
    closed: int
    ai_linked_count: int
    citizen_confirmed_count: int
    avg_resolution_hours: float
    status_distribution: Dict[str, int]
    category_distribution: Dict[str, int]
    priority_distribution: Dict[str, int]
    timeline_trend: List[Dict[str, Any]]
    recent_complaints: List[AdminComplaintOut]


class StaffUserOut(BaseModel):
    id: int
    full_name: str
    email: str
    role: str
    department: Optional[str] = None
