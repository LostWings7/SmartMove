// SafeCity Loop V2 — TypeScript Types & Interfaces

export type RiskLevel = 'LOW' | 'MEDIUM' | 'HIGH' | 'CRITICAL'
export type TrafficExposure = 'LOW' | 'MEDIUM' | 'HIGH' | 'VERY HIGH'
export type RoadStatus = 'monitored' | 'at-risk' | 'pending-repair' | 'repaired'
export type HazardType = 'pothole' | 'near-miss' | 'dangerous-junction' | 'road-work' | 'high-risk-zone'
export type RepairStatus = 'pending' | 'in-progress' | 'completed'

export interface Road {
  id: number
  name: string
  latitude: number
  longitude: number
  road_type?: string
  traffic_exposure: TrafficExposure
  vulnerability: string
  vulnerability_score?: number
  importance_score?: number
  risk_score: number
  risk_confidence?: number
  safe_city_score: number
  data_coverage?: number
  last_observed?: string
  status: RoadStatus
  geometry?: [number, number][]
}

export type RoadSegment = Road

export interface Hazard {
  id: number
  road_id: number
  type: HazardType
  severity: RiskLevel
  confidence: number
  risk_score: number
  latitude: number
  longitude: number
  gps_accuracy?: number
  detected_at: string
  status: string
  description?: string | null
  image_path?: string | null
  source?: string
  verified?: boolean
  evidence_id?: string | null
  evidence_code?: string | null
}

export type HazardObservation = Hazard

export interface NearMiss {
  id: number
  junction_id: number | null
  road_id?: number | null
  video_id?: string | null
  timestamp: string
  conflict_type?: string
  description?: string | null
  object_types: string[]
  object_type_a?: string
  object_type_b?: string
  ttc: number | null
  pet?: number | null
  minimum_distance?: number
  risk_level: RiskLevel
  event_severity?: string
  conflict_zone: string | null
  direction?: string | null
  evidence_clip?: string | null
  review_status?: string
  drac?: number | null
  relative_speed_kmh?: number | null
  interaction_geometry?: string | null
  trajectory_convergence?: boolean | null
  why_conflict_explanation?: any
}

export type ConflictEvent = NearMiss

export interface Junction {
  id: number
  name: string
  latitude: number
  longitude: number
  risk_score: number
  camera_id: string | null
  status: string
  near_misses: NearMiss[]
}

export interface Repair {
  id: number
  road_id: number
  priority: number
  reason: string | null
  status: RepairStatus
  assigned_to: string | null
  assigned_department?: string
  recommended_action?: string
  approval_status?: string
  repair_date?: string | null
  before_score: number | null
  after_score: number | null
  evidence_image?: string | null
  before_risk?: number | null
  after_risk?: number | null
  created_at: string
  road?: Road
}

export type RepairRecord = Repair

export interface Intervention {
  id: number
  road_id: number | null
  name: string
  type: string | null
  before_risk: number | null
  after_risk: number | null
  before_near_misses: number
  after_near_misses: number
  implemented_at: string | null
  notes: string | null
}

export interface CitizenReport {
  id: number
  report_code?: string
  reporter_name: string | null
  type: string
  description: string | null
  latitude: number | null
  longitude: number | null
  severity: RiskLevel
  photo_url?: string | null
  status: string
  submitted_at: string
}

export interface CitizenReportCreate {
  reporter_name?: string
  type: string
  description?: string
  latitude?: number
  longitude?: number
  severity?: RiskLevel
  photo_url?: string | null
}

export interface KPIStats {
  roads_monitored: number
  high_concern_segments?: number
  high_risk_roads: number
  active_hazards?: number
  potholes_detected: number
  conflict_events?: number
  near_misses: number
  danger_zones: number
  pending_repairs: number
  low_confidence_segments?: number
  completed_repairs?: number
}

export interface AIEvent {
  id?: string
  time?: string
  event?: string
  type: string
  title?: string
  description?: string
  severity?: string | null
  location?: string
  timestamp?: string
  risk_score?: number
  image_url?: string | null
}

export interface RiskTrendPoint {
  date: string
  average_risk: number
  high_risk_count: number
}

export interface DataCoverageStats {
  high_coverage_count: number
  moderate_coverage_count: number
  low_coverage_count: number
  average_coverage_pct: number
}

export interface DashboardData {
  kpi: KPIStats
  recent_events: AIEvent[]
  top_repairs: Repair[]
  interventions: Intervention[]
  risk_trend?: RiskTrendPoint[]
  data_coverage_stats?: DataCoverageStats
  recent_conflicts?: NearMiss[]
}

export interface DataQuality {
  last_observation: string | null
  last_observation_hours_ago: number
  observation_count: number
  hazard_count: number
  conflict_count: number
  gps_accuracy_meters: number
  model_confidence: number
  confidence_tier: string
  data_freshness: string
  coverage_percentage: number
  coverage_tier: string
  quality_warning?: string | null
  audit_disclaimer: string
}

export interface ContributingFactorDetail {
  score: number
  weight: number
  weighted_contribution: number
  label: string
}

export interface RiskSnapshot {
  id: number
  road_id: number
  timestamp: string
  risk_score: number
  confidence: number
  data_coverage: number
  hazard_severity: number
  traffic_exposure: number
  conflict_evidence: number
  vulnerable_exposure: number
  persistence: number
  road_importance: number
  contributing_factors: Record<string, any>
}

export interface EvidenceFile {
  id: number
  evidence_code: string
  road_id: number | null
  hazard_id: number | null
  conflict_id: number | null
  file_type: string
  file_url: string
  thumbnail_url: string | null
  captured_at: string
  metadata_json: Record<string, any>
  verified: boolean
  notes: string | null
}

export interface AuditLog {
  id: number
  timestamp: string
  actor: string
  action: string
  target_type: string
  target_id: number | null
  details: string | null
}

export interface RoadSegmentDossier {
  segment_id: number
  road_name: string
  road_type: string
  calculated_risk: number
  safe_city_score: number
  classification: string
  is_danger_zone: boolean
  confidence: number
  data_coverage: number
  traffic_exposure: string
  vulnerable_user_exposure: string
  repair_status: string
  suggested_intervention: string
  quality: DataQuality
  contributing_factors: Record<string, ContributingFactorDetail>
  hazards: Hazard[]
  conflicts: NearMiss[]
  evidence_timeline: EvidenceFile[]
  repairs: Repair[]
  risk_history: RiskSnapshot[]
  disclaimer: string
}

export interface DangerZone {
  id: number
  name: string
  latitude: number
  longitude: number
  risk_score: number
  hazard_count: number
  near_miss_count: number
  status: string
}

// ── AI Vision Types ──────────────────────────────────────────────────────────

export interface BoundingBox {
  x1: number
  y1: number
  x2: number
  y2: number
  width: number
  height: number
}

export interface PotholeDetectionItem {
  box: BoundingBox
  confidence: number
  class_name: string
  severity: RiskLevel
  risk_score: number
  is_demo: boolean
}

export interface PotholeAnalysisResult {
  is_demo_mode: boolean
  model_status: string
  status_message: string
  model_path: string
  pothole_count: number
  confidence: number
  visual_severity?: string
  contextual_severity?: string
  severity: RiskLevel | string
  risk_score: number
  risk_formula: string
  detections: PotholeDetectionItem[]
  original_image_url: string
  processed_image_url: string
  image_width?: number
  image_height?: number
  hazard_id: number | null
  road_id: number | null
  road_name: string | null
  evidence_id?: string | null
  latitude?: number | null
  longitude?: number | null
  gps_accuracy?: number
  is_simulated_gps?: boolean
  success?: boolean
  image_id?: string
  detection_count?: number
  average_confidence?: number
  processing_time_ms?: number
  processing_time_sec?: number
  highest_severity?: string
  traffic_exposure?: string
  vulnerable_users?: string
  persistence?: string
  risk_confidence?: string
  gps_source?: string
  is_precomputed_demo?: boolean
  timestamp?: string
  disclaimer?: string
}

export interface VideoTimelineItem {
  timestamp_str: string
  seconds: number
  status: string
  description?: string
  conflict?: boolean
  conflict_id?: string
  ttc?: number
}

export interface TrafficConflictDetail {
  conflict_id: string
  timestamp_str: string
  timestamp_seconds: number
  object_type_a?: string
  object_type_b?: string
  object_types: string[]
  track_ids: number[]
  ttc: number
  pet?: number | null
  minimum_distance?: number
  risk_level: RiskLevel
  event_severity?: string
  conflict_zone: string
  direction?: string | null
  frame_index: number
  confidence?: number
  snapshot_url: string | null
  review_status?: string
  disclaimer?: string
  drac?: number | null
  relative_speed_kmh?: number | null
  interaction_geometry?: string | null
  trajectory_convergence?: boolean | null
  why_conflict_explanation?: any
}

export interface ConflictHotspot {
  corridor_id: number
  corridor_name: string
  location_type: string
  conflict_count: number
  peak_period: string
  primary_interaction: string
  calculated_concern: string
  data_confidence: string
  mean_ttc: number
  label: string
}

export interface TrafficAnalysisResult {
  success?: boolean
  video_id?: string
  is_demo_mode: boolean
  is_demo_video?: boolean
  is_precomputed_demo?: boolean
  model_status: string
  status_message: string
  junction_id: number | null
  junction_name: string | null
  road_id?: number | null
  duration_seconds: number
  total_frames: number
  processed_frames: number
  processing_time_sec?: number
  tracked_objects_count: number
  object_class_counts: Record<string, number>
  near_miss_count: number
  near_misses: TrafficConflictDetail[]
  timeline?: VideoTimelineItem[]
  original_video_url: string
  processed_video_url: string | null
  conflict_snapshots: string[]
  created_near_miss_ids: number[]
  privacy_applied?: boolean
  privacy_notice?: string
  tracked_objects_summary?: any[]
  exposure_normalized_rates?: any
  traffic_model_meta?: any
  tracking_health_status?: string
  metric_calibration_status?: string
  tracking_health_summary?: {
    tracking_health_status: string
    status_message: string
    total_tracks: number
    multi_point_tracks: number
    tracks_gt_10_samples: number
    single_point_tracks: number
    pct_single_point: number
    median_track_duration_sec: number
    mean_samples_per_track: number
    metric_calibration_status: string
    camera_mode: string
    frames_decoded: number
    frames_analyzed: number
  }
}

export interface AIModelStatus {
  pothole_model_path: string
  pothole_model_loaded: boolean
  traffic_model_path: string
  traffic_model_loaded: boolean
  is_pothole_demo_mode: boolean
  is_traffic_demo_mode: boolean
  message: string
  model_name?: string
  model_version?: string
  framework?: string
  device?: string
  confidence_threshold?: number
  classes?: string[]
  engine?: string
  water_model_path?: string
  water_model_loaded?: boolean
  crack_model_path?: string
  crack_model_loaded?: boolean
  road_condition_capabilities?: Record<string, any>
}

// ── Road Condition Intelligence Types ────────────────────────────────────────

export interface WaterRegion {
  region_id: number
  area_pixels: number
  coverage_percent: number
  bbox: { x1: number; y1: number; x2: number; y2: number; width: number; height: number }
  severity: string
  polygon?: number[][]
}

export interface WaterFilledPotholeEvidence {
  pothole_confidence: number
  water_confidence: number
  overlap_ratio: number
  overlap_percent: number
  water_pixels_in_pothole: number
  total_pothole_pixels: number
  evidence_status: string
  fusion_formula: string
}

export interface RoadCrackDetectionItem {
  crack_type: 'longitudinal_crack' | 'transverse_crack' | 'alligator_crack' | string
  confidence: number
  raw_model_confidence?: number
  road_overlap_ratio?: number
  road_context_score?: number
  final_acceptance_decision?: string
  severity: RiskLevel | string
  risk_score: number
  bbox: { x1: number; y1: number; x2: number; y2: number; width: number; height: number }
  mask_polygon?: number[][]
  estimated_visible_area_sq_m?: number
}

export interface RoadConditionObservation {
  observation_id: string
  hazard_type: string
  confidence: number
  raw_model_confidence?: number
  road_overlap_ratio?: number
  road_context_score?: number
  final_acceptance_decision?: string
  severity: string
  risk_score: number
  bbox?: { x1: number; y1: number; x2: number; y2: number; width: number; height: number }
  mask_polygon?: number[][]
  coverage_percent?: number
  source_model: string
  model_version: string
  timestamp: string
  latitude?: number | null
  longitude?: number | null
  evidence_reference?: string | null
  fusion_evidence?: any
}

export interface RoadConditionAnalysisResult {
  success: boolean
  image_id: string
  total_hazards_count: number
  pothole_count: number
  water_regions_count: number
  crack_count: number
  longitudinal_crack_count: number
  transverse_crack_count: number
  alligator_crack_count: number
  water_filled_pothole_count: number
  water_coverage_percent: number
  water_severity: string
  highest_severity: string
  overall_risk_score: number
  risk_confidence: string
  observations: RoadConditionObservation[]
  water_regions: WaterRegion[]
  water_filled_potholes: any[]
  cracks: RoadCrackDetectionItem[]
  potholes: any[]
  disambiguation_audit?: any[]
  road_context_summary?: Record<string, any>
  original_image_url: string
  processed_image_url: string
  layer_urls: {
    fused: string
    potholes: string
    water: string
    cracks: string
  }
  model_status: Record<string, any>
  processing_time_ms: number
  evidence_id?: string
  hazard_id?: number | null
  road_id?: number | null
  road_name?: string | null
  latitude?: number | null
  longitude?: number | null
  gps_source?: string
  timestamp?: string
  disclaimer: string
}

// ── Road Safety Intelligence Layer Types ─────────────────────────────────────

export interface SafeCityFactors {
  pothole_risk: string
  junction_risk: string
  traffic_exposure: string
  pedestrian_exposure: string
  pothole_score: number
  junction_score: number
  traffic_score: number
  vulnerability_score: number
  report_count: number
  pothole_severity?: string
  hazard_frequency?: number
  conflict_frequency?: number
  conflict_severity?: string
  motorcycle_exposure?: string
  persistence_score?: number
  road_importance_score?: number
  data_confidence?: string
  main_contributing_factor?: string
}

export interface RoadSafetyEvaluation {
  road_id: number
  road_name: string
  danger_zone_score: number
  danger_zone_classification: 'LOW' | 'MEDIUM' | 'HIGH' | 'VERY HIGH' | string
  safe_city_score: number
  factors: SafeCityFactors
  conflict_count?: number
  hazard_count?: number
  traffic_exposure?: string
  main_contributing_factor?: string
  confidence?: string
  coverage?: string
  last_observed?: string | null
  is_hotspot?: boolean
  quality_warning?: string | null
  disclaimer: string
}

export type MunicipalRepairStatus =
  | 'New'
  | 'Verified'
  | 'High Priority'
  | 'Assigned'
  | 'Under Repair'
  | 'Completed'
  | 'NEW'
  | 'UNDER REVIEW'
  | 'VERIFIED'
  | 'REPAIR ASSIGNED'
  | 'REPAIRED'
  | 'POST-REPAIR MONITORING'
  | 'CLOSED'
  | string

export interface PrioritizedRepairItem {
  road_id: number
  road_name: string
  repair_id: number | null
  priority_level: 'LOW' | 'MEDIUM' | 'HIGH' | 'VERY HIGH' | string
  priority_rank: number
  urgency_score: number
  hazard_summary: string
  risk_score: number
  status: MunicipalRepairStatus
  assigned_to: string | null
  assigned_department?: string
  recommended_action?: string
  approval_status?: string
  reasons: string[]
  reason?: string
  suggested_action?: string
  evidence_ids?: string[]
  human_approval_required?: boolean
  created_at: string | null
}

export interface RouteWaypoint {
  latitude: number
  longitude: number
  name: string
  road_id?: number
  risk_score: number
}

export interface RouteOption {
  route_type: string
  route_key: string
  distance_km: number
  time_minutes: number
  pothole_risk: string
  junction_risk: string
  overall_calculated_risk: number
  risk_classification: string
  description: string
  waypoints: RouteWaypoint[]
  risk_penalty?: number
  route_cost?: number
  high_risk_segments_count?: number
  hazards_count?: number
  conflict_hotspots_count?: number
  explanation?: string
  maneuvers?: Array<{
    step_index: number
    instruction: string
    distance_meters: number
    maneuver_type: string
    hazard_warning?: string | null
    corridor_risk?: number
  }>
}

export interface RouteComparisonResult {
  origin_name: string
  destination_name: string
  routes: RouteOption[]
  neutral_advisory: string
}

export interface DecisionAuditRecord {
  evidence_id: string
  source: string
  date: string
  location: string
  ai_result: string
  confidence: number
  reviewer: string
  decision: string
  status: string
}

export interface SmartJunctionDisplayState {
  junction_id: number
  junction_name: string
  display_mode: 'NORMAL' | 'CAUTION' | 'HIGH RISK'
  signal_state: 'RED' | 'AMBER' | 'GREEN'
  countdown_seconds: number
  pedestrian_warning: boolean
  road_hazard_warning: boolean
  ai_risk_state: string
  risk_score: number
  active_near_misses_count: number
  recent_conflict: TrafficConflictDetail | null
}


// ══════════════════════════════════════════════════════════════════════════════
// CITIZEN ACCOUNT & COMPLAINT MANAGEMENT TYPES
// ══════════════════════════════════════════════════════════════════════════════

export type UserRole = 'CITIZEN' | 'INSPECTOR' | 'ENGINEER' | 'ADMIN' | 'SUPERUSER'

export interface User {
  id: number
  full_name: string
  email: string
  phone_number?: string | null
  role: UserRole
  is_active: boolean
  is_verified: boolean
  created_at: string
  last_login?: string | null
  profile_photo?: string | null
  preferred_language?: string
  notification_preferences?: {
    email?: boolean
    in_app?: boolean
  }
}

export interface AuthResponse {
  access_token: string
  token_type: string
  user: User
}

export type ComplaintStatus =
  | 'SUBMITTED'
  | 'ACKNOWLEDGED'
  | 'UNDER_REVIEW'
  | 'VERIFIED'
  | 'ASSIGNED'
  | 'IN_PROGRESS'
  | 'RESOLVED'
  | 'CITIZEN_CONFIRMATION'
  | 'CLOSED'
  | 'REJECTED'
  | 'DUPLICATE'
  | 'REOPENED'

export type ComplaintPriorityType = 'LOW' | 'MEDIUM' | 'HIGH' | 'CRITICAL'

export interface ComplaintCategory {
  id: number
  name: string
  code: string
  description?: string | null
  is_ai_supported: boolean
  is_active: boolean
}

export interface ComplaintEvidence {
  id: number
  complaint_id: number
  uploader_id: number
  file_type: string
  file_url: string
  filename: string
  mime_type: string
  file_size: number
  uploaded_at: string
  privacy_status: string
  notes?: string | null
}

export interface ComplaintStatusHistory {
  id: number
  complaint_id: number
  old_status?: string | null
  new_status: string
  changed_by_name: string
  changed_by_role: string
  comment?: string | null
  timestamp: string
}

export interface CitizenComplaint {
  id: number
  public_reference: string
  category: ComplaintCategory
  title: string
  description: string
  status: ComplaintStatus
  priority: ComplaintPriorityType
  location_lat: number
  location_lng: number
  location_address?: string | null
  assigned_department?: string | null
  created_at: string
  updated_at?: string | null
  resolved_at?: string | null
  closed_at?: string | null
  citizen_confirmed_at?: string | null
  resolution_summary?: string | null
  public_response?: string | null
  evidence_items: ComplaintEvidence[]
  status_history: ComplaintStatusHistory[]
  can_confirm_resolution: boolean
  can_reopen: boolean
}

export interface AdminComplaint extends CitizenComplaint {
  citizen_id: number
  citizen_name: string
  citizen_email: string
  citizen_phone?: string | null
  road_segment_id?: number | null
  road_name?: string | null
  assigned_to?: number | null
  assigned_to_name?: string | null
  source: string
  internal_notes?: string | null
  ai_linked: boolean
  ai_confidence?: number | null
  hazard_id?: number | null
  repair_id?: number | null
  duplicate_candidates_count: number
}

export interface AppNotification {
  id: number
  type: string
  title: string
  message: string
  related_complaint_id?: number | null
  is_read: boolean
  created_at: string
}

export interface CitizenDashboardData {
  welcome_name: string
  total_complaints: number
  active_complaints: number
  pending_review: number
  in_progress: number
  resolved: number
  closed: number
  recent_complaints: CitizenComplaint[]
  recent_notifications: AppNotification[]
}

export interface AdminDashboardData {
  total_complaints: number
  new_today: number
  under_review: number
  assigned: number
  in_progress: number
  resolved: number
  reopened: number
  closed: number
  ai_linked_count: number
  citizen_confirmed_count: number
  avg_resolution_hours: number
  status_distribution: Record<string, number>
  category_distribution: Record<string, number>
  priority_distribution: Record<string, number>
  timeline_trend: Array<{ date: string; submitted: number; resolved: number }>
  recent_complaints: AdminComplaint[]
}

export interface AIHazardCandidate {
  hazard_id: number
  road_id: number
  road_name?: string | null
  type: string
  severity: string
  risk_score: number
  distance_meters: number
  confidence: number
  image_path?: string | null
  detected_at: string
}

export interface DuplicateCandidate {
  complaint_id: number
  public_reference: string
  title: string
  status: string
  distance_meters: number
  similarity_score: number
  created_at: string
}

export interface StaffUser {
  id: number
  full_name: string
  email: string
  role: string
  department?: string | null
}
