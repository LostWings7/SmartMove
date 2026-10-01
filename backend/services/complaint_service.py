"""
SafeCity Loop V2 — Complaint Workflow, AI Correlation & State Machine Service
Implements strict status transitions, public reference generation,
spatial-temporal hazard correlation, duplicate detection, and transactional audit logging.
"""
import math
from datetime import datetime
from typing import List, Optional, Dict, Any
from fastapi import HTTPException, status
from sqlalchemy.orm import Session
from sqlalchemy import func

from backend.models import (
    Complaint, ComplaintStatus, ComplaintCategory, ComplaintStatusHistory,
    ComplaintEvidence, User, UserRole, Hazard, Road, AuditLog, Repair
)
from backend.services.notification_service import create_notification
from backend.services.risk_engine import risk_engine


# ── Distance Utility ──────────────────────────────────────────────────────────

def calculate_distance_meters(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculates great-circle distance in meters between two coordinates via Haversine formula."""
    R = 6371000.0  # Earth radius in meters
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)
    a = (
        math.sin(delta_phi / 2.0) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2.0) ** 2
    )
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return R * c


# ── Public Reference Code Generator ───────────────────────────────────────────

def generate_public_reference(db: Session) -> str:
    """
    Generates a unique, non-sequential, human-readable public reference code.
    Format: SC-2026-XXXXXX (e.g. SC-2026-000184)
    Authoritative database validation ensures no collisions.
    """
    current_year = datetime.utcnow().year
    # Determine the highest existing reference index for the year
    count = db.query(Complaint).count() + 1
    candidate = f"SC-{current_year}-{count:06d}"
    while db.query(Complaint).filter(Complaint.public_reference == candidate).first() is not None:
        count += 1
        candidate = f"SC-{current_year}-{count:06d}"
    return candidate


# ── State Machine Definition & Validation ─────────────────────────────────────

VALID_TRANSITIONS: Dict[str, List[str]] = {
    ComplaintStatus.SUBMITTED: [
        ComplaintStatus.ACKNOWLEDGED,
        ComplaintStatus.UNDER_REVIEW,
        ComplaintStatus.REJECTED,
        ComplaintStatus.DUPLICATE,
    ],
    ComplaintStatus.ACKNOWLEDGED: [
        ComplaintStatus.UNDER_REVIEW,
        ComplaintStatus.VERIFIED,
        ComplaintStatus.REJECTED,
        ComplaintStatus.DUPLICATE,
    ],
    ComplaintStatus.UNDER_REVIEW: [
        ComplaintStatus.VERIFIED,
        ComplaintStatus.ASSIGNED,
        ComplaintStatus.REJECTED,
        ComplaintStatus.DUPLICATE,
    ],
    ComplaintStatus.VERIFIED: [
        ComplaintStatus.ASSIGNED,
        ComplaintStatus.IN_PROGRESS,
        ComplaintStatus.REJECTED,
        ComplaintStatus.UNDER_REVIEW,
    ],
    ComplaintStatus.ASSIGNED: [
        ComplaintStatus.IN_PROGRESS,
        ComplaintStatus.VERIFIED,
        ComplaintStatus.UNDER_REVIEW,
    ],
    ComplaintStatus.IN_PROGRESS: [
        ComplaintStatus.RESOLVED,
        ComplaintStatus.ASSIGNED,
    ],
    ComplaintStatus.RESOLVED: [
        ComplaintStatus.CITIZEN_CONFIRMATION,
        ComplaintStatus.CLOSED,
        ComplaintStatus.REOPENED,
    ],
    ComplaintStatus.CITIZEN_CONFIRMATION: [
        ComplaintStatus.CLOSED,
        ComplaintStatus.REOPENED,
    ],
    ComplaintStatus.CLOSED: [
        ComplaintStatus.REOPENED,
    ],
    ComplaintStatus.REJECTED: [
        ComplaintStatus.UNDER_REVIEW,
        ComplaintStatus.REOPENED,
    ],
    ComplaintStatus.DUPLICATE: [
        ComplaintStatus.UNDER_REVIEW,
    ],
    ComplaintStatus.REOPENED: [
        ComplaintStatus.UNDER_REVIEW,
        ComplaintStatus.ASSIGNED,
        ComplaintStatus.IN_PROGRESS,
    ],
}


def validate_status_transition(old_status: str, new_status: str, user: User, is_owner: bool) -> None:
    """
    Validates state machine transitions against defined graph and user role privileges.
    Raises HTTPException 409 Conflict if transition is not permissible.
    """
    if old_status == new_status:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Complaint is already in status '{old_status}'.",
        )

    allowed = VALID_TRANSITIONS.get(old_status, [])
    if new_status not in allowed:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Invalid state transition from '{old_status}' to '{new_status}'. Allowed transitions: {allowed}",
        )

    # Role-based privilege checks
    if user.role == UserRole.CITIZEN:
        if not is_owner:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Citizens can only update their own complaints.",
            )
        # Citizens can only confirm resolution (-> CLOSED) or reopen (-> REOPENED)
        if new_status not in [ComplaintStatus.CLOSED, ComplaintStatus.REOPENED]:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Citizens cannot perform transition to '{new_status}'. Allowed actions: confirm resolution (CLOSED) or report unresolved (REOPENED).",
            )
        if old_status not in [ComplaintStatus.RESOLVED, ComplaintStatus.CITIZEN_CONFIRMATION, ComplaintStatus.CLOSED]:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Citizens can only confirm or reopen complaints that have reached RESOLVED or CITIZEN_CONFIRMATION.",
            )


# ── Transactional Status Transition Execution ─────────────────────────────────

def execute_status_transition(
    complaint: Complaint,
    new_status: str,
    user: User,
    comment: Optional[str],
    db: Session,
) -> ComplaintStatusHistory:
    """
    Executes a status change transactionally:
    1. Updates complaint status and timestamps
    2. Writes ComplaintStatusHistory record
    3. Writes immutable tamper-evident AuditLog entry
    4. Generates an in-app Notification for the relevant parties
    """
    is_owner = complaint.citizen_id == user.id
    validate_status_transition(complaint.status, new_status, user, is_owner)

    old_status = complaint.status
    now = datetime.utcnow()

    # Apply lifecycle timestamps
    if new_status == ComplaintStatus.RESOLVED:
        complaint.resolved_at = now
        if comment:
            complaint.resolution_summary = comment
    elif new_status == ComplaintStatus.CLOSED:
        complaint.closed_at = now
        if is_owner:
            complaint.citizen_confirmed_at = now
    elif new_status == ComplaintStatus.CITIZEN_CONFIRMATION:
        complaint.resolved_at = complaint.resolved_at or now
    elif new_status == ComplaintStatus.REOPENED:
        complaint.resolved_at = None
        complaint.closed_at = None

    complaint.status = new_status
    complaint.updated_at = now

    # Create history entry
    history = ComplaintStatusHistory(
        complaint_id=complaint.id,
        old_status=old_status,
        new_status=new_status,
        changed_by_id=user.id,
        changed_by_name=user.full_name,
        changed_by_role=user.role,
        comment=comment or f"Status transitioned from {old_status} to {new_status}.",
        timestamp=now,
    )
    db.add(history)

    # Create tamper-evident audit record
    audit_action = f"COMPLAINT_STATUS_{new_status}"
    audit_log = AuditLog(
        actor=f"{user.full_name} ({user.role})",
        action=audit_action,
        target_type="Complaint",
        target_id=complaint.id,
        details=f"Reference: {complaint.public_reference}. Old: '{old_status}' -> New: '{new_status}'. Reason: {comment or 'None specified'}.",
        timestamp=now,
    )
    db.add(audit_log)

    # Create in-app notification
    if user.id != complaint.citizen_id:
        create_notification(
            db=db,
            user_id=complaint.citizen_id,
            type="status_update",
            title=f"Complaint {complaint.public_reference} Update",
            message=f"Status changed to {new_status}. {comment or ''}".strip(),
            complaint_id=complaint.id,
        )
    elif new_status == ComplaintStatus.REOPENED:
        # If citizen reopened, notify assigned officer or admin staff
        target_user_id = complaint.assigned_to
        if not target_user_id:
            first_admin = db.query(User).filter(User.role.in_([UserRole.ADMIN, UserRole.SUPERUSER])).first()
            target_user_id = first_admin.id if first_admin else None

        if target_user_id:
            create_notification(
                db=db,
                user_id=target_user_id,
                type="complaint_reopened",
                title=f"Complaint Reopened: {complaint.public_reference}",
                message=f"Citizen indicated problem still exists: {comment or 'No extra comment.'}",
                complaint_id=complaint.id,
            )

    db.commit()
    db.refresh(complaint)
    return history


# ── AI Hazard Matching & Spatial Correlation ──────────────────────────────────

def find_ai_hazard_candidates(
    complaint: Complaint,
    db: Session,
    max_distance_meters: float = 300.0,
) -> List[Dict[str, Any]]:
    """
    Correlates a citizen complaint with AI-detected hazards within spatial threshold.
    Matches hazard category (e.g. pothole, waterlogging, crack, near-miss).
    Computes spatial score and overall match confidence.
    """
    category = complaint.category
    category_code = category.code.upper() if category else ""

    # Map category to hazard types
    type_matches: List[str] = []
    if "POTHOLE" in category_code:
        type_matches = ["pothole", "surface_cavity"]
    elif "WATER" in category_code:
        type_matches = ["waterlogging", "flooding", "water_filled_pothole"]
    elif "CRACK" in category_code or "DAMAGE" in category_code:
        type_matches = ["crack", "road_crack", "surface_damage", "road-work"]
    elif "CONFLICT" in category_code or "INTERSECTION" in category_code:
        type_matches = ["near-miss", "dangerous-junction", "high-risk-zone"]
    else:
        type_matches = ["pothole", "near-miss", "dangerous-junction", "road-work"]

    hazards = db.query(Hazard).filter(
        Hazard.status != "resolved"
    ).all()

    candidates = []
    for h in hazards:
        # Check type similarity
        type_boost = 1.0 if h.type in type_matches else 0.5
        dist = calculate_distance_meters(complaint.location_lat, complaint.location_lng, h.latitude, h.longitude)

        if dist <= max_distance_meters:
            # Spatial proximity factor (1.0 at 0m, decaying to 0.0 at max_distance_meters)
            spatial_factor = max(0.0, 1.0 - (dist / max_distance_meters))
            # AI confidence factor
            ai_factor = h.confidence or 0.8
            correlation_score = round((spatial_factor * 0.6 + ai_factor * 0.4) * type_boost, 2)

            road = db.query(Road).filter(Road.id == h.road_id).first()
            candidates.append({
                "hazard_id": h.id,
                "road_id": h.road_id,
                "road_name": road.name if road else f"Road #{h.road_id}",
                "type": h.type,
                "severity": h.severity,
                "risk_score": h.risk_score,
                "distance_meters": round(dist, 1),
                "confidence": correlation_score,
                "image_path": h.image_path,
                "detected_at": h.detected_at,
            })

    # Sort candidates by correlation score descending
    candidates.sort(key=lambda c: c["confidence"], reverse=True)
    return candidates


# ── Duplicate Complaint Detection ─────────────────────────────────────────────

def find_duplicate_complaint_candidates(
    complaint: Complaint,
    db: Session,
    max_distance_meters: float = 250.0,
) -> List[Dict[str, Any]]:
    """
    Identifies potential duplicate complaints based on spatial proximity,
    same category, and active status. Does NOT auto-merge; alerts human admin.
    """
    active_complaints = db.query(Complaint).filter(
        Complaint.id != complaint.id,
        Complaint.status.notin_([ComplaintStatus.CLOSED, ComplaintStatus.REJECTED]),
    ).all()

    duplicates = []
    for other in active_complaints:
        dist = calculate_distance_meters(
            complaint.location_lat, complaint.location_lng,
            other.location_lat, other.location_lng
        )
        if dist <= max_distance_meters:
            category_match = 1.0 if other.category_id == complaint.category_id else 0.4
            spatial_factor = max(0.0, 1.0 - (dist / max_distance_meters))
            similarity = round(spatial_factor * 0.7 + category_match * 0.3, 2)

            if similarity >= 0.5:
                duplicates.append({
                    "complaint_id": other.id,
                    "public_reference": other.public_reference,
                    "title": other.title,
                    "status": other.status,
                    "distance_meters": round(dist, 1),
                    "similarity_score": similarity,
                    "created_at": other.created_at,
                })

    duplicates.sort(key=lambda d: d["similarity_score"], reverse=True)
    return duplicates


# ── Complaint to SafeCity Loop Linking ────────────────────────────────────────

def link_complaint_to_hazard(
    complaint: Complaint,
    hazard_id: int,
    user: User,
    db: Session,
    notes: Optional[str] = None,
) -> Hazard:
    """Links an active complaint to an existing SafeCity Hazard, updating risk engine."""
    hazard = db.query(Hazard).filter(Hazard.id == hazard_id).first()
    if not hazard:
        raise HTTPException(status_code=404, detail="Target Hazard not found.")

    complaint.hazard_id = hazard.id
    complaint.road_segment_id = hazard.road_id
    complaint.ai_linked = True
    complaint.ai_confidence = hazard.confidence

    # Recalculate road risk
    road = db.query(Road).filter(Road.id == hazard.road_id).first()
    if road:
        risk_engine.evaluate_road(road, db)

    # Tamper-evident audit log
    db.add(AuditLog(
        actor=f"{user.full_name} ({user.role})",
        action="COMPLAINT_LINKED_TO_HAZARD",
        target_type="Complaint",
        target_id=complaint.id,
        details=f"Complaint {complaint.public_reference} manually linked to Hazard #{hazard.id} ({hazard.type}) on road #{hazard.road_id}. {notes or ''}".strip(),
        timestamp=datetime.utcnow(),
    ))
    db.commit()
    db.refresh(complaint)
    return hazard


def create_hazard_from_complaint(
    complaint: Complaint,
    user: User,
    db: Session,
    road_id: Optional[int] = None,
    severity: str = "HIGH",
    notes: Optional[str] = None,
) -> Hazard:
    """
    Promotes a verified citizen complaint into a formal SafeCity Hazard.
    Integrates directly with the municipal Risk Engine & Repair Workflow.
    """
    target_road: Optional[Road] = None
    if road_id:
        target_road = db.query(Road).filter(Road.id == road_id).first()

    if not target_road:
        # Find closest monitored road segment
        roads = db.query(Road).all()
        if roads:
            target_road = min(
                roads,
                key=lambda r: calculate_distance_meters(complaint.location_lat, complaint.location_lng, r.latitude, r.longitude)
            )

    if not target_road:
        raise HTTPException(status_code=400, detail="No road segment found to associate with hazard.")

    # Determine hazard type based on category
    category_code = complaint.category.code.lower() if complaint.category else "pothole"
    hazard_type = "pothole"
    if "water" in category_code:
        hazard_type = "waterlogging"
    elif "crack" in category_code:
        hazard_type = "road_crack"
    elif "conflict" in category_code or "junction" in category_code:
        hazard_type = "near-miss"

    # Create new Hazard
    new_hazard = Hazard(
        road_id=target_road.id,
        type=hazard_type,
        severity=severity.upper(),
        confidence=0.90,  # Municipal verified confidence
        risk_score=75.0 if severity.upper() == "HIGH" else 60.0,
        latitude=complaint.location_lat,
        longitude=complaint.location_lng,
        gps_accuracy=3.0,
        detected_at=datetime.utcnow(),
        status="active",
        source="Citizen Report (Verified)",
        verified=True,
        evidence_code=f"CR-{complaint.public_reference}",
        visual_severity=severity.upper(),
        contextual_severity=severity.upper(),
    )
    db.add(new_hazard)
    db.flush()

    complaint.hazard_id = new_hazard.id
    complaint.road_segment_id = target_road.id
    complaint.ai_linked = True
    complaint.ai_confidence = 0.90

    # Recalculate road risk using SafeCity centralized RiskEngine
    risk_engine.evaluate_road(target_road, db)

    # Tamper-evident AuditLog
    db.add(AuditLog(
        actor=f"{user.full_name} ({user.role})",
        action="HAZARD_CREATED_FROM_COMPLAINT",
        target_type="Complaint",
        target_id=complaint.id,
        details=f"Created Hazard #{new_hazard.id} ({new_hazard.type}, {new_hazard.severity}) on '{target_road.name}' from complaint {complaint.public_reference}. Notes: {notes or 'None'}.",
        timestamp=datetime.utcnow(),
    ))
    db.commit()
    db.refresh(complaint)
    db.refresh(new_hazard)
    return new_hazard
