"""
SafeCity Loop V2 — ORM Models
Complete database foundation supporting RoadSegment, HazardObservation,
ConflictEvent, RepairRecord, RiskSnapshot, EvidenceFile, and AuditLog.
Maintains 100% backward compatibility with existing queries.
"""
from datetime import datetime
from sqlalchemy import (
    Column, Integer, String, Float, DateTime, ForeignKey, Text, JSON, Boolean
)
from sqlalchemy.orm import relationship
from backend.database import Base


class Road(Base):
    """Represents a monitored road segment."""
    __tablename__ = "roads"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(200), nullable=False)
    latitude = Column(Float, nullable=False)
    longitude = Column(Float, nullable=False)
    road_type = Column(String(50), default="urban_arterial")       # urban_arterial, school_zone, transit_corridor, local_residential, industrial
    traffic_exposure = Column(String(20), default="MEDIUM")        # LOW/MEDIUM/HIGH/VERY HIGH
    vulnerability = Column(String(20), default="MEDIUM")           # LOW/MEDIUM/HIGH
    vulnerability_score = Column(Float, default=50.0)              # 0–100
    importance_score = Column(Float, default=50.0)                 # 0–100
    risk_score = Column(Float, default=0.0)                        # 0–100 calculated by RiskEngine
    risk_confidence = Column(Float, default=0.85)                  # 0.0–1.0
    safe_city_score = Column(Float, default=100.0)                 # 0–100 (100 - risk_score)
    data_coverage = Column(Float, default=75.0)                    # 0–100% (No data != safe)
    last_observed = Column(DateTime, default=datetime.utcnow)
    status = Column(String(50), default="monitored")               # monitored/at-risk/pending-repair/repaired
    geometry = Column(JSON, default=list)                          # Polyline coordinates [[lat, lon], ...]

    hazards = relationship("Hazard", back_populates="road", cascade="all, delete-orphan")
    conflicts = relationship("NearMiss", back_populates="road", cascade="all, delete-orphan")
    repairs = relationship("Repair", back_populates="road", cascade="all, delete-orphan")
    snapshots = relationship("RiskSnapshot", back_populates="road", cascade="all, delete-orphan")
    evidence_files = relationship("EvidenceFile", back_populates="road", cascade="all, delete-orphan")
    complaints = relationship("Complaint", back_populates="road")


# Alias for explicit domain naming
RoadSegment = Road


class Hazard(Base):
    """A detected hazard observation on a road segment."""
    __tablename__ = "hazards"

    id = Column(Integer, primary_key=True, index=True)
    road_id = Column(Integer, ForeignKey("roads.id"), nullable=False)
    type = Column(String(50), nullable=False)                      # pothole/near-miss/dangerous-junction/road-work/high-risk-zone
    severity = Column(String(20), default="MEDIUM")                # LOW/MEDIUM/HIGH/CRITICAL
    confidence = Column(Float, default=0.0)                        # 0.0–1.0 (AI detection confidence)
    risk_score = Column(Float, default=0.0)                        # 0–100
    latitude = Column(Float, nullable=False)
    longitude = Column(Float, nullable=False)
    gps_accuracy = Column(Float, default=4.5)                      # in meters
    detected_at = Column(DateTime, default=datetime.utcnow)
    status = Column(String(50), default="active")                  # active/resolved/monitoring
    image_path = Column(String(500), nullable=True)
    source = Column(String(50), default="AI Vision")               # "AI Vision" / "Citizen Report" / "Municipal Inspection"
    verified = Column(Boolean, default=False)
    evidence_id = Column(String(50), nullable=True)                # e.g. "EV-2026-0012"
    evidence_code = Column(String(50), nullable=True)              # e.g. "SC-H-1042"
    visual_severity = Column(String(20), default="HIGH")           # LOW/MEDIUM/HIGH/CRITICAL
    contextual_severity = Column(String(20), default="HIGH")       # LOW/MEDIUM/HIGH/CRITICAL
    direction = Column(String(50), nullable=True)                  # e.g. "Northbound Lane 1"

    road = relationship("Road", back_populates="hazards")
    complaints = relationship("Complaint", back_populates="hazard")


# Alias for explicit domain naming
HazardObservation = Hazard


class NearMiss(Base):
    """A traffic conflict event (near miss) observed on a road segment or junction."""
    __tablename__ = "near_misses"

    id = Column(Integer, primary_key=True, index=True)
    junction_id = Column(Integer, ForeignKey("junctions.id"), nullable=True)
    road_id = Column(Integer, ForeignKey("roads.id"), nullable=True)
    video_id = Column(String(100), nullable=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    object_types = Column(JSON, default=list)                      # e.g. ["vehicle", "pedestrian"]
    object_type_a = Column(String(50), default="vehicle")
    object_type_b = Column(String(50), default="pedestrian")
    ttc = Column(Float, nullable=True)                             # Time-to-collision in seconds (< 2.0s)
    pet = Column(Float, nullable=True)                             # Post-Encroachment Time in seconds
    minimum_distance = Column(Float, default=2.1)                  # in meters
    risk_level = Column(String(20), default="MEDIUM")              # LOW/MEDIUM/HIGH/CRITICAL
    event_severity = Column(String(20), default="HIGH")
    conflict_zone = Column(String(100), nullable=True)             # e.g. "Intersection Center Corridor"
    evidence_clip = Column(String(500), nullable=True)
    review_status = Column(String(50), default="pending_review")   # pending_review / reviewed / false_positive / actioned
    confidence = Column(Float, default=0.88)                       # AI detection/tracking confidence
    source = Column(String(100), default="YOLOv8 + ByteTrack")     # e.g. "YOLOv8 + ByteTrack (Local Edge)"
    direction = Column(String(100), nullable=True)                 # e.g. "Northbound ⟷ Pedestrian Crossing"
    created_at = Column(DateTime, default=datetime.utcnow)

    junction = relationship("Junction", back_populates="near_misses")
    road = relationship("Road", back_populates="conflicts")

    @property
    def event_id(self) -> int:
        return self.id

    @property
    def segment_id(self) -> Optional[int]:
        return self.road_id


# Alias for explicit domain naming
ConflictEvent = NearMiss


class Junction(Base):
    """A road junction being monitored for traffic conflicts."""
    __tablename__ = "junctions"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(200), nullable=False)
    latitude = Column(Float, nullable=False)
    longitude = Column(Float, nullable=False)
    risk_score = Column(Float, default=0.0)
    camera_id = Column(String(100), nullable=True)
    status = Column(String(50), default="active")

    near_misses = relationship("NearMiss", back_populates="junction", cascade="all, delete-orphan")


class Repair(Base):
    """A municipal repair task assigned to a road segment."""
    __tablename__ = "repairs"

    id = Column(Integer, primary_key=True, index=True)
    road_id = Column(Integer, ForeignKey("roads.id"), nullable=False)
    priority = Column(Integer, default=3)                          # 1 = highest
    reason = Column(Text, nullable=True)
    status = Column(String(50), default="pending")                 # pending/in-progress/completed
    assigned_to = Column(String(200), nullable=True)
    assigned_department = Column(String(200), default="Municipal Public Works")
    recommended_action = Column(String(200), default="Asphalt Resurfacing")
    approval_status = Column(String(50), default="approved")       # draft / pending_approval / approved / rejected
    repair_date = Column(DateTime, nullable=True)
    before_score = Column(Float, nullable=True)
    after_score = Column(Float, nullable=True)
    before_observations = Column(Integer, default=3)
    after_observations = Column(Integer, default=0)
    evidence_image = Column(String(500), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    road = relationship("Road", back_populates="repairs")
    complaints = relationship("Complaint", back_populates="repair")

    @property
    def before_risk(self) -> Optional[float]:
        return self.before_score

    @before_risk.setter
    def before_risk(self, val: Optional[float]):
        self.before_score = val

    @property
    def after_risk(self) -> Optional[float]:
        return self.after_score

    @after_risk.setter
    def after_risk(self, val: Optional[float]):
        self.after_score = val


# Alias for explicit domain naming
RepairRecord = Repair


class RiskSnapshot(Base):
    """Historical risk snapshot for trend analytics."""
    __tablename__ = "risk_snapshots"

    id = Column(Integer, primary_key=True, index=True)
    road_id = Column(Integer, ForeignKey("roads.id"), nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)
    risk_score = Column(Float, nullable=False)
    confidence = Column(Float, default=0.85)
    data_coverage = Column(Float, default=80.0)
    hazard_severity = Column(Float, default=0.0)
    traffic_exposure = Column(Float, default=0.0)
    conflict_evidence = Column(Float, default=0.0)
    vulnerable_exposure = Column(Float, default=0.0)
    persistence = Column(Float, default=0.0)
    road_importance = Column(Float, default=0.0)
    contributing_factors = Column(JSON, default=dict)

    road = relationship("Road", back_populates="snapshots")


class EvidenceFile(Base):
    """Centralized evidence file linking detections and conflicts to road segments."""
    __tablename__ = "evidence_files"

    id = Column(Integer, primary_key=True, index=True)
    evidence_code = Column(String(50), unique=True, index=True)     # e.g. "EV-2026-0012"
    road_id = Column(Integer, ForeignKey("roads.id"), nullable=True)
    hazard_id = Column(Integer, ForeignKey("hazards.id"), nullable=True)
    conflict_id = Column(Integer, ForeignKey("near_misses.id"), nullable=True)
    file_type = Column(String(50), nullable=False)                  # "image", "video_clip", "keyframe", "sensor"
    file_url = Column(String(500), nullable=False)
    thumbnail_url = Column(String(500), nullable=True)
    captured_at = Column(DateTime, default=datetime.utcnow)
    metadata_json = Column(JSON, default=dict)
    verified = Column(Boolean, default=False)
    notes = Column(Text, nullable=True)

    road = relationship("Road", back_populates="evidence_files")


class AuditLog(Base):
    """Municipal officer and system decision audit trail."""
    __tablename__ = "audit_logs"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    actor = Column(String(100), default="System AI")                # "Municipal Officer", "Road Authority", "System AI"
    action = Column(String(100), nullable=False)                    # "RISK_RECALCULATION", "HAZARD_DETECTED", "CONFLICT_REVIEWED", "REPAIR_STATUS_UPDATED"
    target_type = Column(String(50), nullable=False)                # "RoadSegment", "Hazard", "ConflictEvent", "Repair"
    target_id = Column(Integer, nullable=True)
    details = Column(Text, nullable=True)


class Intervention(Base):
    """Records a safety intervention and its measured impact."""
    __tablename__ = "interventions"

    id = Column(Integer, primary_key=True, index=True)
    road_id = Column(Integer, ForeignKey("roads.id"), nullable=True)
    name = Column(String(200), nullable=False)
    type = Column(String(100), nullable=True)                       # speed-bump/resurfacing/signage/lighting
    before_risk = Column(Float, nullable=True)
    after_risk = Column(Float, nullable=True)
    before_near_misses = Column(Integer, default=0)
    after_near_misses = Column(Integer, default=0)
    implemented_at = Column(DateTime, nullable=True)
    notes = Column(Text, nullable=True)


class CitizenReport(Base):
    """A hazard report submitted by a citizen."""
    __tablename__ = "citizen_reports"

    id = Column(Integer, primary_key=True, index=True)
    report_code = Column(String(50), nullable=True)                 # e.g. SC-DEMO-00128
    reporter_name = Column(String(200), nullable=True)
    type = Column(String(100), nullable=False)                      # pothole/flooding/debris/other
    description = Column(Text, nullable=True)
    latitude = Column(Float, nullable=True)
    longitude = Column(Float, nullable=True)
    severity = Column(String(20), default="MEDIUM")
    photo_url = Column(String(500), nullable=True)
    status = Column(String(50), default="AI Verification Pending")
    submitted_at = Column(DateTime, default=datetime.utcnow)


class VideoAnalysis(Base):
    """Log of processed traffic video sessions."""
    __tablename__ = "video_analyses"

    id = Column(Integer, primary_key=True, index=True)
    video_id = Column(String(100), unique=True, index=True)
    filename = Column(String(255), nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)
    road_id = Column(Integer, ForeignKey("roads.id"), nullable=True)
    junction_id = Column(Integer, ForeignKey("junctions.id"), nullable=True)
    frame_count = Column(Integer, default=0)
    processed_frames = Column(Integer, default=0)
    object_count = Column(Integer, default=0)
    conflict_count = Column(Integer, default=0)
    processing_time_sec = Column(Float, default=0.0)
    video_url = Column(String(500), nullable=True)
    processed_video_url = Column(String(500), nullable=True)
    summary_json = Column(JSON, default=dict)


# ══════════════════════════════════════════════════════════════════════════════
# SAFE CITY LOOP V2 — CITIZEN PORTAL & COMPLAINT WORKFLOW MODELS
# ══════════════════════════════════════════════════════════════════════════════

class UserRole:
    CITIZEN = "CITIZEN"
    INSPECTOR = "INSPECTOR"
    ENGINEER = "ENGINEER"
    ADMIN = "ADMIN"
    SUPERUSER = "SUPERUSER"
    ALL = [CITIZEN, INSPECTOR, ENGINEER, ADMIN, SUPERUSER]
    STAFF = [INSPECTOR, ENGINEER, ADMIN, SUPERUSER]


class ComplaintStatus:
    SUBMITTED = "SUBMITTED"
    ACKNOWLEDGED = "ACKNOWLEDGED"
    UNDER_REVIEW = "UNDER_REVIEW"
    VERIFIED = "VERIFIED"
    ASSIGNED = "ASSIGNED"
    IN_PROGRESS = "IN_PROGRESS"
    RESOLVED = "RESOLVED"
    CITIZEN_CONFIRMATION = "CITIZEN_CONFIRMATION"
    CLOSED = "CLOSED"
    REJECTED = "REJECTED"
    DUPLICATE = "DUPLICATE"
    REOPENED = "REOPENED"
    ALL = [
        SUBMITTED, ACKNOWLEDGED, UNDER_REVIEW, VERIFIED, ASSIGNED,
        IN_PROGRESS, RESOLVED, CITIZEN_CONFIRMATION, CLOSED, REJECTED,
        DUPLICATE, REOPENED
    ]


class ComplaintPriority:
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class ComplaintSource:
    CITIZEN = "CITIZEN"
    AI_DETECTION = "AI_DETECTION"
    ADMIN = "ADMIN"
    INSPECTOR = "INSPECTOR"
    SYSTEM = "SYSTEM"


class User(Base):
    """Authoritative user account supporting RBAC (Citizen, Inspector, Engineer, Admin, Superuser)."""
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    full_name = Column(String(200), nullable=False)
    email = Column(String(255), unique=True, index=True, nullable=False)
    phone_number = Column(String(50), nullable=True)
    password_hash = Column(String(255), nullable=False)
    role = Column(String(50), default=UserRole.CITIZEN, index=True, nullable=False)
    is_active = Column(Boolean, default=True, nullable=False)
    is_verified = Column(Boolean, default=True, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    last_login = Column(DateTime, nullable=True)
    profile_photo = Column(String(500), nullable=True)
    preferred_language = Column(String(20), default="en")
    notification_preferences = Column(JSON, default=lambda: {"email": True, "in_app": True})

    complaints = relationship("Complaint", back_populates="citizen", foreign_keys="Complaint.citizen_id", cascade="all, delete-orphan")
    assigned_complaints = relationship("Complaint", back_populates="assignee", foreign_keys="Complaint.assigned_to")
    notifications = relationship("Notification", back_populates="user", cascade="all, delete-orphan")
    evidence_uploads = relationship("ComplaintEvidence", back_populates="uploader")


class ComplaintCategory(Base):
    """Configurable municipal complaint category with AI detection capability flag."""
    __tablename__ = "complaint_categories"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), unique=True, nullable=False)
    code = Column(String(50), unique=True, nullable=False)
    description = Column(Text, nullable=True)
    is_ai_supported = Column(Boolean, default=False)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    complaints = relationship("Complaint", back_populates="category")


class Complaint(Base):
    """Normalized database-authoritative citizen and municipal complaint record."""
    __tablename__ = "complaints"

    id = Column(Integer, primary_key=True, index=True)
    public_reference = Column(String(50), unique=True, index=True, nullable=False)  # e.g. "SC-2026-000184"
    citizen_id = Column(Integer, ForeignKey("users.id"), index=True, nullable=False)
    category_id = Column(Integer, ForeignKey("complaint_categories.id"), index=True, nullable=False)
    title = Column(String(255), nullable=False)
    description = Column(Text, nullable=False)
    status = Column(String(50), default=ComplaintStatus.SUBMITTED, index=True, nullable=False)
    priority = Column(String(20), default=ComplaintPriority.MEDIUM, index=True, nullable=False)
    location_lat = Column(Float, nullable=False)
    location_lng = Column(Float, nullable=False)
    location_address = Column(String(500), nullable=True)
    road_segment_id = Column(Integer, ForeignKey("roads.id"), index=True, nullable=True)
    assigned_to = Column(Integer, ForeignKey("users.id"), index=True, nullable=True)
    assigned_department = Column(String(100), default="Public Works", nullable=True)
    source = Column(String(50), default=ComplaintSource.CITIZEN)

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, index=True, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, index=True)
    resolved_at = Column(DateTime, nullable=True)
    closed_at = Column(DateTime, nullable=True)
    citizen_confirmed_at = Column(DateTime, nullable=True)

    # Resolution & Municipal interaction
    resolution_summary = Column(Text, nullable=True)
    public_response = Column(Text, nullable=True)     # Visible to citizen
    internal_notes = Column(Text, nullable=True)      # Staff/admin only

    # SafeCity Loop Ecosystem Integration
    ai_linked = Column(Boolean, default=False)
    ai_confidence = Column(Float, nullable=True)
    hazard_id = Column(Integer, ForeignKey("hazards.id"), nullable=True)
    repair_id = Column(Integer, ForeignKey("repairs.id"), nullable=True)

    # Relationships
    citizen = relationship("User", foreign_keys=[citizen_id], back_populates="complaints")
    assignee = relationship("User", foreign_keys=[assigned_to], back_populates="assigned_complaints")
    category = relationship("ComplaintCategory", back_populates="complaints")
    road = relationship("Road", back_populates="complaints")
    hazard = relationship("Hazard", back_populates="complaints")
    repair = relationship("Repair", back_populates="complaints")
    evidence_items = relationship("ComplaintEvidence", back_populates="complaint", cascade="all, delete-orphan", order_by="ComplaintEvidence.uploaded_at.asc()")
    status_history = relationship("ComplaintStatusHistory", back_populates="complaint", cascade="all, delete-orphan", order_by="ComplaintStatusHistory.timestamp.asc()")


class ComplaintStatusHistory(Base):
    """Full chronological audit trail of all complaint status transitions."""
    __tablename__ = "complaint_status_histories"

    id = Column(Integer, primary_key=True, index=True)
    complaint_id = Column(Integer, ForeignKey("complaints.id"), index=True, nullable=False)
    old_status = Column(String(50), nullable=True)
    new_status = Column(String(50), nullable=False)
    changed_by_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    changed_by_name = Column(String(200), nullable=False)
    changed_by_role = Column(String(50), nullable=False)
    comment = Column(Text, nullable=True)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True, nullable=False)

    complaint = relationship("Complaint", back_populates="status_history")
    changed_by = relationship("User")


class ComplaintEvidence(Base):
    """Validated evidence media uploaded for a complaint."""
    __tablename__ = "complaint_evidences"

    id = Column(Integer, primary_key=True, index=True)
    complaint_id = Column(Integer, ForeignKey("complaints.id"), index=True, nullable=False)
    uploader_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    file_type = Column(String(50), nullable=False)      # image, video, document
    file_url = Column(String(500), nullable=False)
    filename = Column(String(255), nullable=False)
    mime_type = Column(String(100), nullable=False)
    file_size = Column(Integer, nullable=False)
    uploaded_at = Column(DateTime, default=datetime.utcnow, index=True, nullable=False)
    privacy_status = Column(String(50), default="processed")
    notes = Column(Text, nullable=True)

    complaint = relationship("Complaint", back_populates="evidence_items")
    uploader = relationship("User", back_populates="evidence_uploads")


class Notification(Base):
    """In-app notifications for citizens and municipal staff."""
    __tablename__ = "notifications"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), index=True, nullable=False)
    type = Column(String(50), nullable=False)  # status_update, assignment, municipal_response, system
    title = Column(String(200), nullable=False)
    message = Column(Text, nullable=False)
    related_complaint_id = Column(Integer, ForeignKey("complaints.id"), nullable=True)
    is_read = Column(Boolean, default=False, index=True, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, index=True, nullable=False)

    user = relationship("User", back_populates="notifications")
    complaint = relationship("Complaint")

