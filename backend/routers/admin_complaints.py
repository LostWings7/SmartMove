"""
SafeCity Loop V2 — Admin & Municipal Complaints Router
Full operational complaint management, staff assignment, public responses,
internal notes, AI correlation, hazard creation, work order linking, analytics, and CSV export.
"""
import io
import csv
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session
from sqlalchemy import or_, func

from backend.database import get_db
from backend.models import (
    Complaint, ComplaintCategory, ComplaintStatus, ComplaintStatusHistory,
    ComplaintEvidence, User, UserRole, Hazard, Road, AuditLog, Repair
)
from backend.schemas import (
    AdminComplaintOut, ComplaintCategoryOut, ComplaintEvidenceOut,
    ComplaintStatusHistoryOut, AdminDashboardOut, ComplaintStatusUpdateRequest,
    ComplaintAssignRequest, ComplaintPublicResponseRequest,
    ComplaintInternalNoteRequest, ComplaintLinkHazardRequest,
    ComplaintCreateHazardRequest, ComplaintLinkRepairRequest,
    DuplicateCandidateOut, AIHazardCandidateOut, StaffUserOut
)
from backend.auth import (
    require_staff_user, require_admin_user, get_current_user
)
from backend.services.complaint_service import (
    execute_status_transition, find_ai_hazard_candidates,
    find_duplicate_complaint_candidates, link_complaint_to_hazard,
    create_hazard_from_complaint
)
from backend.services.notification_service import create_notification

router = APIRouter(prefix="/admin", tags=["Admin Complaints & Operations"])


def _build_admin_complaint_out(c: Complaint, db: Session) -> AdminComplaintOut:
    """Builds administrative complaint detail containing operational and internal fields."""
    assignee_name = c.assignee.full_name if c.assignee else None
    road_name = c.road.name if c.road else None

    # Count potential duplicates
    dups = find_duplicate_complaint_candidates(c, db)

    return AdminComplaintOut(
        id=c.id,
        public_reference=c.public_reference,
        citizen_id=c.citizen_id,
        citizen_name=c.citizen.full_name if c.citizen else "Unknown Citizen",
        citizen_email=c.citizen.email if c.citizen else "N/A",
        citizen_phone=c.citizen.phone_number if c.citizen else None,
        category=ComplaintCategoryOut.model_validate(c.category),
        title=c.title,
        description=c.description,
        status=c.status,
        priority=c.priority,
        location_lat=c.location_lat,
        location_lng=c.location_lng,
        location_address=c.location_address,
        road_segment_id=c.road_segment_id,
        road_name=road_name,
        assigned_to=c.assigned_to,
        assigned_to_name=assignee_name,
        assigned_department=c.assigned_department,
        source=c.source or "CITIZEN",
        created_at=c.created_at,
        updated_at=c.updated_at,
        resolved_at=c.resolved_at,
        closed_at=c.closed_at,
        citizen_confirmed_at=c.citizen_confirmed_at,
        resolution_summary=c.resolution_summary,
        public_response=c.public_response,
        internal_notes=c.internal_notes,
        ai_linked=c.ai_linked,
        ai_confidence=c.ai_confidence,
        hazard_id=c.hazard_id,
        repair_id=c.repair_id,
        evidence_items=[ComplaintEvidenceOut.model_validate(e) for e in c.evidence_items],
        status_history=[ComplaintStatusHistoryOut.model_validate(h) for h in c.status_history],
        duplicate_candidates_count=len(dups),
    )


# ── Admin Dashboard ───────────────────────────────────────────────────────────

@router.get("/dashboard", response_model=AdminDashboardOut)
def get_admin_dashboard(
    current_user: User = Depends(require_staff_user),
    db: Session = Depends(get_db),
):
    """
    Authoritative database metrics for the Municipal Command Center.
    Queries real SQLite tables. Zero static arrays or mocked telemetry.
    """
    now = datetime.utcnow()
    today_start = datetime(now.year, now.month, now.day)

    total = db.query(Complaint).count()
    new_today = db.query(Complaint).filter(Complaint.created_at >= today_start).count()
    under_review = db.query(Complaint).filter(Complaint.status.in_([ComplaintStatus.SUBMITTED, ComplaintStatus.ACKNOWLEDGED, ComplaintStatus.UNDER_REVIEW])).count()
    assigned = db.query(Complaint).filter(Complaint.status == ComplaintStatus.ASSIGNED).count()
    in_progress = db.query(Complaint).filter(Complaint.status == ComplaintStatus.IN_PROGRESS).count()
    resolved = db.query(Complaint).filter(Complaint.status.in_([ComplaintStatus.RESOLVED, ComplaintStatus.CITIZEN_CONFIRMATION])).count()
    reopened = db.query(Complaint).filter(Complaint.status == ComplaintStatus.REOPENED).count()
    closed = db.query(Complaint).filter(Complaint.status == ComplaintStatus.CLOSED).count()

    ai_linked_count = db.query(Complaint).filter(Complaint.ai_linked == True).count()
    citizen_confirmed_count = db.query(Complaint).filter(Complaint.citizen_confirmed_at != None).count()

    # Calculate average resolution time for resolved complaints
    resolved_complaints = db.query(Complaint).filter(Complaint.resolved_at != None).all()
    avg_hours = 0.0
    if resolved_complaints:
        diffs = [(c.resolved_at - c.created_at).total_seconds() / 3600.0 for c in resolved_complaints if c.resolved_at > c.created_at]
        if diffs:
            avg_hours = round(sum(diffs) / len(diffs), 1)

    # Status distribution
    status_counts = (
        db.query(Complaint.status, func.count(Complaint.id))
        .group_by(Complaint.status)
        .all()
    )
    status_dist = {s: c for s, c in status_counts}

    # Category distribution
    cat_counts = (
        db.query(ComplaintCategory.name, func.count(Complaint.id))
        .join(Complaint, Complaint.category_id == ComplaintCategory.id)
        .group_by(ComplaintCategory.name)
        .all()
    )
    cat_dist = {name: c for name, c in cat_counts}

    # Priority distribution
    priority_counts = (
        db.query(Complaint.priority, func.count(Complaint.id))
        .group_by(Complaint.priority)
        .all()
    )
    prio_dist = {p: c for p, c in priority_counts}

    # 7-day timeline trend
    timeline_trend = []
    for day_offset in range(6, -1, -1):
        d = today_start - timedelta(days=day_offset)
        d_end = d + timedelta(days=1)
        cnt = db.query(Complaint).filter(Complaint.created_at >= d, Complaint.created_at < d_end).count()
        res_cnt = db.query(Complaint).filter(Complaint.resolved_at >= d, Complaint.resolved_at < d_end).count()
        timeline_trend.append({
            "date": d.strftime("%d %b"),
            "submitted": cnt,
            "resolved": res_cnt,
        })

    # Recent complaints
    recent = (
        db.query(Complaint)
        .order_by(Complaint.created_at.desc())
        .limit(10)
        .all()
    )
    recent_outs = [_build_admin_complaint_out(c, db) for c in recent]

    return AdminDashboardOut(
        total_complaints=total,
        new_today=new_today,
        under_review=under_review,
        assigned=assigned,
        in_progress=in_progress,
        resolved=resolved,
        reopened=reopened,
        closed=closed,
        ai_linked_count=ai_linked_count,
        citizen_confirmed_count=citizen_confirmed_count,
        avg_resolution_hours=avg_hours,
        status_distribution=status_dist,
        category_distribution=cat_dist,
        priority_distribution=prio_dist,
        timeline_trend=timeline_trend,
        recent_complaints=recent_outs,
    )


# ── Complaints List (Search, Filter, Sort, Paginate) ──────────────────────────

@router.get("/complaints")
def list_admin_complaints(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    status: Optional[str] = Query(None),
    category_id: Optional[int] = Query(None),
    priority: Optional[str] = Query(None),
    assigned_to: Optional[int] = Query(None),
    ai_linked: Optional[bool] = Query(None),
    search: Optional[str] = Query(None),
    sort_by: str = Query("created_at", pattern="^(created_at|updated_at|priority|status|public_reference)$"),
    sort_order: str = Query("desc", pattern="^(asc|desc)$"),
    current_user: User = Depends(require_staff_user),
    db: Session = Depends(get_db),
):
    """
    Returns filtered, searched, sorted, and paginated complaints for municipal staff.
    """
    q = db.query(Complaint)

    if status:
        q = q.filter(Complaint.status == status.upper())
    if category_id:
        q = q.filter(Complaint.category_id == category_id)
    if priority:
        q = q.filter(Complaint.priority == priority.upper())
    if assigned_to:
        q = q.filter(Complaint.assigned_to == assigned_to)
    if ai_linked is not None:
        q = q.filter(Complaint.ai_linked == ai_linked)

    if search:
        term = f"%{search.strip()}%"
        q = q.join(User, Complaint.citizen_id == User.id).filter(
            or_(
                Complaint.public_reference.ilike(term),
                Complaint.title.ilike(term),
                Complaint.description.ilike(term),
                Complaint.location_address.ilike(term),
                User.full_name.ilike(term),
                User.email.ilike(term),
            )
        )

    total = q.count()

    # Sorting
    sort_col = getattr(Complaint, sort_by, Complaint.created_at)
    if sort_order == "asc":
        q = q.order_by(sort_col.asc())
    else:
        q = q.order_by(sort_col.desc())

    offset = (page - 1) * page_size
    items = q.offset(offset).limit(page_size).all()

    items_out = [_build_admin_complaint_out(c, db) for c in items]
    total_pages = (total + page_size - 1) // page_size

    return {
        "items": items_out,
        "total": total,
        "page": page,
        "page_size": page_size,
        "total_pages": total_pages,
    }


# ── Single Complaint Operational View ─────────────────────────────────────────

@router.get("/complaints/{complaint_id}")
def get_admin_complaint_detail(
    complaint_id: int,
    current_user: User = Depends(require_staff_user),
    db: Session = Depends(get_db),
):
    """
    Returns full complaint details including duplicate candidates,
    AI correlation candidates, and linked repair/hazard records.
    """
    complaint = db.query(Complaint).filter(Complaint.id == complaint_id).first()
    if not complaint:
        raise HTTPException(status_code=404, detail="Complaint not found.")

    out = _build_admin_complaint_out(complaint, db)
    ai_candidates = find_ai_hazard_candidates(complaint, db)
    duplicate_candidates = find_duplicate_complaint_candidates(complaint, db)

    return {
        "complaint": out,
        "ai_hazard_candidates": ai_candidates,
        "duplicate_candidates": duplicate_candidates,
    }


# ── Update Status (State Machine) ─────────────────────────────────────────────

@router.post("/complaints/{complaint_id}/status", response_model=AdminComplaintOut)
def update_complaint_status(
    complaint_id: int,
    payload: ComplaintStatusUpdateRequest,
    current_user: User = Depends(require_staff_user),
    db: Session = Depends(get_db),
):
    """Performs validated state machine transition with tamper-evident audit record."""
    complaint = db.query(Complaint).filter(Complaint.id == complaint_id).first()
    if not complaint:
        raise HTTPException(status_code=404, detail="Complaint not found.")

    execute_status_transition(
        complaint=complaint,
        new_status=payload.new_status.upper(),
        user=current_user,
        comment=payload.comment,
        db=db,
    )

    return _build_admin_complaint_out(complaint, db)


# ── Assign Staff / Department ─────────────────────────────────────────────────

@router.post("/complaints/{complaint_id}/assign", response_model=AdminComplaintOut)
def assign_complaint(
    complaint_id: int,
    payload: ComplaintAssignRequest,
    current_user: User = Depends(require_staff_user),
    db: Session = Depends(get_db),
):
    """Assigns complaint to a staff member or department."""
    complaint = db.query(Complaint).filter(Complaint.id == complaint_id).first()
    if not complaint:
        raise HTTPException(status_code=404, detail="Complaint not found.")

    assignee: Optional[User] = None
    if payload.assigned_to:
        assignee = db.query(User).filter(User.id == payload.assigned_to).first()
        if not assignee:
            raise HTTPException(status_code=404, detail="Assigned staff user not found.")
        complaint.assigned_to = assignee.id

    if payload.assigned_department:
        complaint.assigned_department = payload.assigned_department

    # Auto transition to ASSIGNED if currently UNDER_REVIEW or VERIFIED
    if complaint.status in [ComplaintStatus.SUBMITTED, ComplaintStatus.UNDER_REVIEW, ComplaintStatus.VERIFIED]:
        complaint.status = ComplaintStatus.ASSIGNED

    complaint.updated_at = datetime.utcnow()

    # Log history & audit
    db.add(ComplaintStatusHistory(
        complaint_id=complaint.id,
        old_status=complaint.status,
        new_status=ComplaintStatus.ASSIGNED,
        changed_by_id=current_user.id,
        changed_by_name=current_user.full_name,
        changed_by_role=current_user.role,
        comment=f"Assigned to {assignee.full_name if assignee else 'team'} ({complaint.assigned_department}). {payload.notes or ''}".strip(),
        timestamp=datetime.utcnow(),
    ))

    db.add(AuditLog(
        actor=f"{current_user.full_name} ({current_user.role})",
        action="COMPLAINT_ASSIGNED",
        target_type="Complaint",
        target_id=complaint.id,
        details=f"Assigned to user #{complaint.assigned_to} ({complaint.assigned_department}). Notes: {payload.notes or 'None'}.",
        timestamp=datetime.utcnow(),
    ))

    # Notify assignee if set
    if assignee and assignee.id != current_user.id:
        create_notification(
            db=db,
            user_id=assignee.id,
            type="assignment",
            title=f"New Assignment: {complaint.public_reference}",
            message=f"You have been assigned complaint '{complaint.title}' ({complaint.priority} priority).",
            complaint_id=complaint.id,
        )

    db.commit()
    db.refresh(complaint)
    return _build_admin_complaint_out(complaint, db)


# ── Public Response ───────────────────────────────────────────────────────────

@router.post("/complaints/{complaint_id}/public-response", response_model=AdminComplaintOut)
def add_public_response(
    complaint_id: int,
    payload: ComplaintPublicResponseRequest,
    current_user: User = Depends(require_staff_user),
    db: Session = Depends(get_db),
):
    """
    Sets the official municipal response visible to the citizen.
    Notifies citizen immediately via in-app notification.
    """
    complaint = db.query(Complaint).filter(Complaint.id == complaint_id).first()
    if not complaint:
        raise HTTPException(status_code=404, detail="Complaint not found.")

    complaint.public_response = payload.public_response.strip()
    complaint.updated_at = datetime.utcnow()

    # Tamper-evident AuditLog
    db.add(AuditLog(
        actor=f"{current_user.full_name} ({current_user.role})",
        action="PUBLIC_RESPONSE_ADDED",
        target_type="Complaint",
        target_id=complaint.id,
        details=f"Municipal response posted to citizen on {complaint.public_reference}.",
        timestamp=datetime.utcnow(),
    ))

    # In-app notification to the citizen
    create_notification(
        db=db,
        user_id=complaint.citizen_id,
        type="municipal_response",
        title=f"Municipal Response: {complaint.public_reference}",
        message=f"The municipality has posted a response: {payload.public_response[:120]}...",
        complaint_id=complaint.id,
    )

    db.commit()
    db.refresh(complaint)
    return _build_admin_complaint_out(complaint, db)


# ── Internal Notes ────────────────────────────────────────────────────────────

@router.post("/complaints/{complaint_id}/internal-note", response_model=AdminComplaintOut)
def add_internal_note(
    complaint_id: int,
    payload: ComplaintInternalNoteRequest,
    current_user: User = Depends(require_staff_user),
    db: Session = Depends(get_db),
):
    """Adds staff-only internal note. Strictly hidden from citizen endpoints."""
    complaint = db.query(Complaint).filter(Complaint.id == complaint_id).first()
    if not complaint:
        raise HTTPException(status_code=404, detail="Complaint not found.")

    timestamp_str = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    new_entry = f"[{timestamp_str} - {current_user.full_name} ({current_user.role})]: {payload.internal_note.strip()}"

    if complaint.internal_notes:
        complaint.internal_notes = f"{complaint.internal_notes}\n{new_entry}"
    else:
        complaint.internal_notes = new_entry

    complaint.updated_at = datetime.utcnow()

    # Tamper-evident AuditLog
    db.add(AuditLog(
        actor=f"{current_user.full_name} ({current_user.role})",
        action="INTERNAL_NOTE_ADDED",
        target_type="Complaint",
        target_id=complaint.id,
        details="Added confidential staff internal note.",
        timestamp=datetime.utcnow(),
    ))

    db.commit()
    db.refresh(complaint)
    return _build_admin_complaint_out(complaint, db)


# ── AI Hazard Linking & Creation ──────────────────────────────────────────────

@router.post("/complaints/{complaint_id}/link-hazard", response_model=AdminComplaintOut)
def link_hazard(
    complaint_id: int,
    payload: ComplaintLinkHazardRequest,
    current_user: User = Depends(require_staff_user),
    db: Session = Depends(get_db),
):
    """Links complaint to an existing AI-detected Hazard."""
    complaint = db.query(Complaint).filter(Complaint.id == complaint_id).first()
    if not complaint:
        raise HTTPException(status_code=404, detail="Complaint not found.")

    link_complaint_to_hazard(complaint, payload.hazard_id, current_user, db, payload.notes)
    return _build_admin_complaint_out(complaint, db)


@router.post("/complaints/{complaint_id}/create-hazard", response_model=AdminComplaintOut)
def create_hazard(
    complaint_id: int,
    payload: ComplaintCreateHazardRequest,
    current_user: User = Depends(require_staff_user),
    db: Session = Depends(get_db),
):
    """Promotes verified complaint into a formal SafeCity Hazard in the risk loop."""
    complaint = db.query(Complaint).filter(Complaint.id == complaint_id).first()
    if not complaint:
        raise HTTPException(status_code=404, detail="Complaint not found.")

    create_hazard_from_complaint(
        complaint=complaint,
        user=current_user,
        db=db,
        road_id=payload.road_id,
        severity=payload.severity or "HIGH",
        notes=payload.notes,
    )
    return _build_admin_complaint_out(complaint, db)


# ── Link to Work Order / Repair ───────────────────────────────────────────────

@router.post("/complaints/{complaint_id}/link-repair", response_model=AdminComplaintOut)
def link_repair_work_order(
    complaint_id: int,
    payload: ComplaintLinkRepairRequest,
    current_user: User = Depends(require_staff_user),
    db: Session = Depends(get_db),
):
    """Links complaint directly to a municipal Repair work order."""
    complaint = db.query(Complaint).filter(Complaint.id == complaint_id).first()
    if not complaint:
        raise HTTPException(status_code=404, detail="Complaint not found.")

    repair = db.query(Repair).filter(Repair.id == payload.repair_id).first()
    if not repair:
        raise HTTPException(status_code=404, detail="Repair record not found.")

    complaint.repair_id = repair.id
    complaint.updated_at = datetime.utcnow()

    db.add(AuditLog(
        actor=f"{current_user.full_name} ({current_user.role})",
        action="COMPLAINT_LINKED_TO_REPAIR",
        target_type="Complaint",
        target_id=complaint.id,
        details=f"Linked complaint {complaint.public_reference} to Repair #{repair.id} on road #{repair.road_id}. {payload.notes or ''}".strip(),
        timestamp=datetime.utcnow(),
    ))
    db.commit()
    db.refresh(complaint)
    return _build_admin_complaint_out(complaint, db)


# ── Staff List for Assignment Dropdowns ────────────────────────────────────────

@router.get("/staff", response_model=List[StaffUserOut])
def get_staff_users(
    current_user: User = Depends(require_staff_user),
    db: Session = Depends(get_db),
):
    """Returns municipal staff users available for complaint assignment."""
    staff = (
        db.query(User)
        .filter(User.role.in_(UserRole.STAFF), User.is_active == True)
        .order_by(User.full_name.asc())
        .all()
    )
    return [
        StaffUserOut(
            id=s.id,
            full_name=s.full_name,
            email=s.email,
            role=s.role,
            department="Public Works" if s.role == UserRole.ENGINEER else "Traffic Enforcement",
        )
        for s in staff
    ]


# ── Export CSV Report ─────────────────────────────────────────────────────────

@router.get("/complaints/export")
def export_complaints_csv(
    status: Optional[str] = Query(None),
    category_id: Optional[int] = Query(None),
    current_user: User = Depends(require_staff_user),
    db: Session = Depends(get_db),
):
    """Exports complaint records as CSV for municipal audit and executive reporting."""
    q = db.query(Complaint)
    if status:
        q = q.filter(Complaint.status == status.upper())
    if category_id:
        q = q.filter(Complaint.category_id == category_id)

    complaints = q.order_by(Complaint.created_at.desc()).all()

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow([
        "Reference", "Title", "Category", "Citizen Name", "Citizen Email",
        "Status", "Priority", "Latitude", "Longitude", "Address",
        "Road Segment", "Assigned To", "Department", "Created At",
        "Resolved At", "AI Linked", "Public Response"
    ])

    for c in complaints:
        writer.writerow([
            c.public_reference,
            c.title,
            c.category.name if c.category else "N/A",
            c.citizen.full_name if c.citizen else "N/A",
            c.citizen.email if c.citizen else "N/A",
            c.status,
            c.priority,
            c.location_lat,
            c.location_lng,
            c.location_address or "",
            c.road.name if c.road else "",
            c.assignee.full_name if c.assignee else "",
            c.assigned_department or "",
            c.created_at.isoformat() if c.created_at else "",
            c.resolved_at.isoformat() if c.resolved_at else "",
            "YES" if c.ai_linked else "NO",
            c.public_response or "",
        ])

    output.seek(0)
    filename = f"SafeCity_Complaints_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.csv"
    return StreamingResponse(
        io.BytesIO(output.getvalue().encode("utf-8")),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )
