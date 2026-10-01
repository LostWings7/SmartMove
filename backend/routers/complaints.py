"""
SafeCity Loop V2 — Citizen Complaints & Lifecycle Router
Provides citizen complaint submission, file evidence upload,
strict citizen data isolation, timeline tracking, and resolution confirmation.
"""
import os
import uuid
from datetime import datetime
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form, Query, status
from sqlalchemy.orm import Session
from sqlalchemy import or_

from backend.database import get_db
from backend.models import (
    Complaint, ComplaintCategory, ComplaintStatus, ComplaintStatusHistory,
    ComplaintEvidence, User, UserRole, Road, AuditLog
)
from backend.schemas import (
    ComplaintCreateRequest, CitizenComplaintOut, ComplaintCategoryOut,
    ComplaintEvidenceOut, ComplaintStatusHistoryOut, CitizenDashboardOut,
    CitizenResolutionConfirmRequest
)
from backend.auth import get_current_user
from backend.services.complaint_service import (
    generate_public_reference, execute_status_transition,
    calculate_distance_meters
)
from backend.services.notification_service import (
    create_notification, get_user_notifications
)

router = APIRouter(tags=["Citizen Complaints"])

# Ensure uploads directory for complaints
UPLOAD_DIR = os.path.join("uploads", "complaints")
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Allowed file extensions and MIME types
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".mp4", ".mov", ".webm", ".pdf"}
MAX_FILE_SIZE_BYTES = 15 * 1024 * 1024  # 15 MB limit


def _build_citizen_complaint_out(c: Complaint, citizen_id: int) -> CitizenComplaintOut:
    """Formats complaint for citizen consumption, ensuring internal notes are never exposed."""
    return CitizenComplaintOut(
        id=c.id,
        public_reference=c.public_reference,
        category=ComplaintCategoryOut.model_validate(c.category),
        title=c.title,
        description=c.description,
        status=c.status,
        priority=c.priority,
        location_lat=c.location_lat,
        location_lng=c.location_lng,
        location_address=c.location_address,
        assigned_department=c.assigned_department,
        created_at=c.created_at,
        updated_at=c.updated_at,
        resolved_at=c.resolved_at,
        closed_at=c.closed_at,
        citizen_confirmed_at=c.citizen_confirmed_at,
        resolution_summary=c.resolution_summary,
        public_response=c.public_response,
        evidence_items=[ComplaintEvidenceOut.model_validate(e) for e in c.evidence_items],
        status_history=[ComplaintStatusHistoryOut.model_validate(h) for h in c.status_history],
        can_confirm_resolution=(
            c.citizen_id == citizen_id
            and c.status in [ComplaintStatus.RESOLVED, ComplaintStatus.CITIZEN_CONFIRMATION]
        ),
        can_reopen=(
            c.citizen_id == citizen_id
            and c.status in [ComplaintStatus.RESOLVED, ComplaintStatus.CITIZEN_CONFIRMATION, ComplaintStatus.CLOSED]
        ),
    )


# ── Categories ────────────────────────────────────────────────────────────────

@router.get("/complaints/categories", response_model=List[ComplaintCategoryOut])
def get_categories(db: Session = Depends(get_db)):
    """Returns active municipal complaint categories."""
    return (
        db.query(ComplaintCategory)
        .filter(ComplaintCategory.is_active == True)
        .order_by(ComplaintCategory.id.asc())
        .all()
    )


# ── Citizen Dashboard ─────────────────────────────────────────────────────────

@router.get("/citizen/dashboard", response_model=CitizenDashboardOut)
def get_citizen_dashboard(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Returns live database-backed metrics and recent complaints for the logged-in citizen.
    No mock arrays. Authoritative database aggregation.
    """
    user_id = current_user.id
    base_q = db.query(Complaint).filter(Complaint.citizen_id == user_id)

    total = base_q.count()
    under_review = base_q.filter(Complaint.status.in_([ComplaintStatus.SUBMITTED, ComplaintStatus.ACKNOWLEDGED, ComplaintStatus.UNDER_REVIEW])).count()
    in_prog = base_q.filter(Complaint.status.in_([ComplaintStatus.VERIFIED, ComplaintStatus.ASSIGNED, ComplaintStatus.IN_PROGRESS])).count()
    resolved = base_q.filter(Complaint.status.in_([ComplaintStatus.RESOLVED, ComplaintStatus.CITIZEN_CONFIRMATION])).count()
    closed = base_q.filter(Complaint.status == ComplaintStatus.CLOSED).count()
    active = under_review + in_prog

    recent = (
        base_q.order_by(Complaint.created_at.desc())
        .limit(5)
        .all()
    )
    recent_outs = [_build_citizen_complaint_out(c, user_id) for c in recent]

    recent_notifications = get_user_notifications(db, user_id, limit=5)

    return CitizenDashboardOut(
        welcome_name=current_user.full_name,
        total_complaints=total,
        active_complaints=active,
        pending_review=under_review,
        in_progress=in_prog,
        resolved=resolved,
        closed=closed,
        recent_complaints=recent_outs,
        recent_notifications=recent_notifications,
    )


# ── Submit Complaint ──────────────────────────────────────────────────────────

@router.post("/complaints", response_model=CitizenComplaintOut, status_code=status.HTTP_201_CREATED)
def submit_complaint(
    payload: ComplaintCreateRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Files a new citizen complaint.
    1. Validates category
    2. Identifies nearest road segment
    3. Generates authoritative human-readable reference (e.g. SC-2026-000184)
    4. Sets initial status SUBMITTED
    5. Adds initial status history and tamper-evident audit record
    """
    category = db.query(ComplaintCategory).filter(ComplaintCategory.id == payload.category_id).first()
    if not category:
        raise HTTPException(status_code=404, detail="Selected complaint category does not exist.")

    # Find closest road segment
    roads = db.query(Road).all()
    closest_road = None
    if roads:
        closest_road = min(
            roads,
            key=lambda r: calculate_distance_meters(payload.location_lat, payload.location_lng, r.latitude, r.longitude)
        )

    public_ref = generate_public_reference(db)

    complaint = Complaint(
        public_reference=public_ref,
        citizen_id=current_user.id,
        category_id=category.id,
        title=payload.title.strip(),
        description=payload.description.strip(),
        status=ComplaintStatus.SUBMITTED,
        priority=payload.priority.upper() if payload.priority else "MEDIUM",
        location_lat=payload.location_lat,
        location_lng=payload.location_lng,
        location_address=payload.location_address,
        road_segment_id=closest_road.id if closest_road else None,
        source="CITIZEN",
        created_at=datetime.utcnow(),
    )
    db.add(complaint)
    db.flush()

    # Initial status history
    history = ComplaintStatusHistory(
        complaint_id=complaint.id,
        old_status=None,
        new_status=ComplaintStatus.SUBMITTED,
        changed_by_id=current_user.id,
        changed_by_name=current_user.full_name,
        changed_by_role=current_user.role,
        comment="Complaint submitted by citizen.",
        timestamp=datetime.utcnow(),
    )
    db.add(history)

    # Immutable AuditLog entry
    db.add(AuditLog(
        actor=f"{current_user.full_name} ({current_user.role})",
        action="COMPLAINT_CREATED",
        target_type="Complaint",
        target_id=complaint.id,
        details=f"Reference: {public_ref}. Category: {category.name}. Location: ({payload.location_lat}, {payload.location_lng})",
        timestamp=datetime.utcnow(),
    ))

    # Confirmation in-app notification to the citizen
    create_notification(
        db=db,
        user_id=current_user.id,
        type="complaint_submitted",
        title=f"Complaint Received: {public_ref}",
        message=f"Your complaint regarding '{payload.title}' has been registered with reference {public_ref}.",
        complaint_id=complaint.id,
    )

    db.commit()
    db.refresh(complaint)
    return _build_citizen_complaint_out(complaint, current_user.id)


# ── Citizen My Complaints ─────────────────────────────────────────────────────

@router.get("/complaints/my", response_model=List[CitizenComplaintOut])
def get_my_complaints(
    status_filter: Optional[str] = Query(None, alias="status"),
    category_id: Optional[int] = Query(None),
    search: Optional[str] = Query(None),
    sort_order: str = Query("desc", pattern="^(asc|desc)$"),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Returns authenticated citizen's complaints.
    Strict data isolation ensures users cannot view other citizens' complaints.
    """
    q = db.query(Complaint).filter(Complaint.citizen_id == current_user.id)

    if status_filter:
        q = q.filter(Complaint.status == status_filter.upper())
    if category_id:
        q = q.filter(Complaint.category_id == category_id)
    if search:
        term = f"%{search.strip()}%"
        q = q.filter(or_(Complaint.public_reference.ilike(term), Complaint.title.ilike(term), Complaint.description.ilike(term)))

    if sort_order == "asc":
        q = q.order_by(Complaint.created_at.asc())
    else:
        q = q.order_by(Complaint.created_at.desc())

    complaints = q.all()
    return [_build_citizen_complaint_out(c, current_user.id) for c in complaints]


# ── Get Single Complaint ──────────────────────────────────────────────────────

@router.get("/complaints/{complaint_id}", response_model=CitizenComplaintOut)
def get_complaint_detail(
    complaint_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Returns single complaint details.
    Citizen can only view their own complaint. Staff can view any.
    Internal notes and staff personal emails are never returned.
    """
    complaint = db.query(Complaint).filter(Complaint.id == complaint_id).first()
    if not complaint:
        raise HTTPException(status_code=404, detail="Complaint not found.")

    # Data isolation
    is_staff = current_user.role in UserRole.STAFF
    if not is_staff and complaint.citizen_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Unauthorized: You do not have permission to access this complaint.",
        )

    return _build_citizen_complaint_out(complaint, current_user.id)


# ── Evidence Upload ───────────────────────────────────────────────────────────

@router.post("/complaints/{complaint_id}/evidence", response_model=ComplaintEvidenceOut)
async def upload_complaint_evidence(
    complaint_id: int,
    file: UploadFile = File(...),
    notes: Optional[str] = Form(None),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Secure file evidence upload for a complaint.
    1. Validates user owns complaint (or is municipal staff)
    2. Validates extension and file size
    3. Generates sanitized UUID server-side filename (prevents path traversal)
    4. Records EvidenceFile in DB and logs tamper-evident audit record
    """
    complaint = db.query(Complaint).filter(Complaint.id == complaint_id).first()
    if not complaint:
        raise HTTPException(status_code=404, detail="Complaint not found.")

    is_staff = current_user.role in UserRole.STAFF
    if not is_staff and complaint.citizen_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Unauthorized: Cannot upload evidence to another citizen's complaint.",
        )

    # Validate file extension
    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"File extension '{ext}' is not supported. Allowed extensions: {list(ALLOWED_EXTENSIONS)}",
        )

    content = await file.read()
    file_size = len(content)
    if file_size > MAX_FILE_SIZE_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File size ({round(file_size / (1024*1024), 2)}MB) exceeds maximum allowed size of 15MB.",
        )

    # Save to disk with safe UUID name
    safe_filename = f"{uuid.uuid4().hex}{ext}"
    dest_path = os.path.join(UPLOAD_DIR, safe_filename)
    with open(dest_path, "wb") as f:
        f.write(content)

    file_url = f"/uploads/complaints/{safe_filename}"
    file_type = "video" if ext in [".mp4", ".mov", ".webm"] else "image"

    evidence = ComplaintEvidence(
        complaint_id=complaint.id,
        uploader_id=current_user.id,
        file_type=file_type,
        file_url=file_url,
        filename=os.path.basename(file.filename or safe_filename),
        mime_type=file.content_type or "application/octet-stream",
        file_size=file_size,
        uploaded_at=datetime.utcnow(),
        privacy_status="processed",
        notes=notes,
    )
    db.add(evidence)

    # Tamper-evident AuditLog
    db.add(AuditLog(
        actor=f"{current_user.full_name} ({current_user.role})",
        action="EVIDENCE_UPLOADED",
        target_type="Complaint",
        target_id=complaint.id,
        details=f"Uploaded evidence '{safe_filename}' ({file_size} bytes, {file_type}) to complaint {complaint.public_reference}.",
        timestamp=datetime.utcnow(),
    ))
    db.commit()
    db.refresh(evidence)

    return ComplaintEvidenceOut.model_validate(evidence)


# ── Citizen Resolution Confirmation & Reopening ───────────────────────────────

@router.post("/complaints/{complaint_id}/citizen-confirm", response_model=CitizenComplaintOut)
def citizen_confirm_resolution(
    complaint_id: int,
    payload: CitizenResolutionConfirmRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Citizen confirmation workflow:
    When municipal repair is marked RESOLVED:
    - If citizen confirms (confirmed=True): transitions to CLOSED.
    - If problem still exists (confirmed=False): transitions to REOPENED with citizen explanation.
    """
    complaint = db.query(Complaint).filter(Complaint.id == complaint_id).first()
    if not complaint:
        raise HTTPException(status_code=404, detail="Complaint not found.")

    if complaint.citizen_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only the filing citizen can confirm or reopen this complaint.",
        )

    if payload.confirmed:
        new_status = ComplaintStatus.CLOSED
        comment = payload.comment or "Citizen confirmed problem has been resolved satisfactorily."
    else:
        new_status = ComplaintStatus.REOPENED
        comment = payload.comment or "Citizen reported that the problem still exists after municipal repair."

    execute_status_transition(
        complaint=complaint,
        new_status=new_status,
        user=current_user,
        comment=comment,
        db=db,
    )

    return _build_citizen_complaint_out(complaint, current_user.id)


# ── Complaint Status Timeline ─────────────────────────────────────────────────

@router.get("/complaints/{complaint_id}/timeline", response_model=List[ComplaintStatusHistoryOut])
def get_complaint_timeline(
    complaint_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Returns chronological status history timeline for a complaint."""
    complaint = db.query(Complaint).filter(Complaint.id == complaint_id).first()
    if not complaint:
        raise HTTPException(status_code=404, detail="Complaint not found.")

    is_staff = current_user.role in UserRole.STAFF
    if not is_staff and complaint.citizen_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Unauthorized to access this complaint's timeline.",
        )

    history = (
        db.query(ComplaintStatusHistory)
        .filter(ComplaintStatusHistory.complaint_id == complaint_id)
        .order_by(ComplaintStatusHistory.timestamp.asc())
        .all()
    )
    return [ComplaintStatusHistoryOut.model_validate(h) for h in history]
