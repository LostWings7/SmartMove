"""
SafeCity Loop V2 — Traffic Conflicts Router
Serves traffic conflict events (near misses) with TTC, PET, minimum distance,
trajectories, and municipal review status.
"""
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from backend.database import get_db
from backend.models import NearMiss, Road
from backend.schemas import ConflictEventOut, ConflictReviewUpdate
from backend.services.evidence_service import evidence_service

router = APIRouter(prefix="/conflicts", tags=["Traffic Conflicts"])


@router.get("", response_model=List[ConflictEventOut])
def list_conflicts(
    road_id: Optional[int] = Query(None, description="Filter by road segment"),
    risk_level: Optional[str] = Query(None, description="Filter by risk level"),
    review_status: Optional[str] = Query(None, description="Filter by review status"),
    db: Session = Depends(get_db)
):
    """
    Returns all logged traffic conflict events (near-misses).
    Includes TTC (Time-To-Collision), PET (Post-Encroachment Time), minimum distance, and severity.
    """
    query = db.query(NearMiss)
    if road_id is not None:
        query = query.filter(NearMiss.road_id == road_id)
    if risk_level is not None:
        query = query.filter(NearMiss.risk_level == risk_level.upper())
    if review_status is not None:
        query = query.filter(NearMiss.review_status == review_status.lower())

    return query.order_by(NearMiss.timestamp.desc()).all()


@router.patch("/{event_id}/review", response_model=ConflictEventOut)
def review_conflict_event(
    event_id: int,
    payload: ConflictReviewUpdate,
    db: Session = Depends(get_db)
):
    """
    Municipal officer review workflow for a traffic conflict event.
    Status can be: pending_review, reviewed, false_positive, actioned.
    """
    conflict = db.query(NearMiss).filter(NearMiss.id == event_id).first()
    if not conflict:
        raise HTTPException(status_code=404, detail=f"Conflict event #{event_id} not found.")

    conflict.review_status = payload.review_status.lower()
    db.commit()
    db.refresh(conflict)

    evidence_service.log_audit(
        db=db,
        actor="Municipal Officer",
        action="CONFLICT_EVENT_REVIEWED",
        target_type="ConflictEvent",
        target_id=conflict.id,
        details=f"Updated review status to '{conflict.review_status}'. Notes: {payload.notes or 'None'}"
    )

    return conflict


@router.get("/{event_id}/evidence")
def get_conflict_evidence(
    event_id: int,
    db: Session = Depends(get_db)
):
    """
    Retrieve event-specific visual evidence keyframe artifact for a conflict event.
    Returns participant bounding boxes, trajectory trails, and keyframe snapshot URL.
    """
    conflict = db.query(NearMiss).filter(NearMiss.id == event_id).first()
    if not conflict:
        raise HTTPException(status_code=404, detail=f"Conflict event #{event_id} not found.")

    from backend.models import EvidenceFile
    ev_file = db.query(EvidenceFile).filter(EvidenceFile.conflict_id == event_id).first()

    return {
        "event_id": conflict.id,
        "conflict_id": conflict.id,
        "evidence_clip": conflict.evidence_clip,
        "snapshot_url": conflict.evidence_clip,
        "thumbnail_url": ev_file.thumbnail_url if ev_file else conflict.evidence_clip,
        "object_type_a": conflict.object_type_a,
        "object_type_b": conflict.object_type_b,
        "object_types": conflict.object_types,
        "ttc": conflict.ttc,
        "pet": conflict.pet,
        "minimum_distance": conflict.minimum_distance,
        "risk_level": conflict.risk_level,
        "event_severity": conflict.event_severity,
        "conflict_zone": conflict.conflict_zone,
        "direction": conflict.direction,
        "confidence": conflict.confidence,
        "review_status": conflict.review_status,
        "timestamp": conflict.timestamp.isoformat() if conflict.timestamp else None,
    }


@router.get("/hotspots", response_model=List[dict])
def get_recurring_conflict_hotspots(db: Session = Depends(get_db)):
    """
    Recurring Conflict Hotspots (Danger-Zone Candidates):
    Aggregates conflict telemetry by corridor / junction to identify empirical patterns
    without claiming certainty or accident prediction.
    """
    from collections import Counter
    from backend.models import Junction, Road

    hotspots = []
    junctions = db.query(Junction).all()

    for j in junctions:
        j_conflicts = db.query(NearMiss).filter(NearMiss.junction_id == j.id).all()
        if not j_conflicts:
            continue

        count = len(j_conflicts)
        valid_ttcs = [c.ttc for c in j_conflicts if c.ttc is not None]
        mean_ttc = round(sum(valid_ttcs) / len(valid_ttcs), 2) if valid_ttcs else 1.35

        # Interactions
        interactions = [
            f"{(c.object_type_a or 'vehicle').capitalize()} ⟷ {(c.object_type_b or 'pedestrian').capitalize()}"
            for c in j_conflicts
        ]
        top_interaction = Counter(interactions).most_common(1)[0][0] if interactions else "Vehicle ⟷ Pedestrian"

        # Hours
        hours = [c.timestamp.hour for c in j_conflicts if c.timestamp]
        top_hour = Counter(hours).most_common(1)[0][0] if hours else 17
        start_hour = top_hour
        end_hour = (top_hour + 2) % 24
        peak_str = f"{start_hour:02d}:00 – {end_hour:02d}:00"

        concern = "VERY HIGH" if count >= 3 or any(c.risk_level == "CRITICAL" for c in j_conflicts) else "HIGH" if count >= 2 else "MODERATE"
        confidence = "HIGH" if count >= 4 else "MEDIUM"

        hotspots.append({
            "corridor_id": j.id,
            "corridor_name": j.name,
            "location_type": "Junction",
            "conflict_count": count,
            "peak_period": peak_str,
            "primary_interaction": top_interaction,
            "calculated_concern": concern,
            "data_confidence": confidence,
            "mean_ttc": mean_ttc,
            "label": "Recurring Conflict Hotspot",
        })

    # Also check road corridors with logged conflicts
    roads = db.query(Road).all()
    for r in roads:
        r_conflicts = db.query(NearMiss).filter(NearMiss.road_id == r.id).all()
        if not r_conflicts:
            continue

        count = len(r_conflicts)
        valid_ttcs = [c.ttc for c in r_conflicts if c.ttc is not None]
        mean_ttc = round(sum(valid_ttcs) / len(valid_ttcs), 2) if valid_ttcs else 1.40

        interactions = [
            f"{(c.object_type_a or 'vehicle').capitalize()} ⟷ {(c.object_type_b or 'pedestrian').capitalize()}"
            for c in r_conflicts
        ]
        top_interaction = Counter(interactions).most_common(1)[0][0] if interactions else "Vehicle ⟷ Pedestrian"

        hours = [c.timestamp.hour for c in r_conflicts if c.timestamp]
        top_hour = Counter(hours).most_common(1)[0][0] if hours else 18
        peak_str = f"{top_hour:02d}:00 – {(top_hour+2)%24:02d}:00"

        concern = "VERY HIGH" if count >= 3 or any(c.risk_level == "CRITICAL" for c in r_conflicts) else "HIGH" if count >= 2 else "MODERATE"
        confidence = "HIGH" if count >= 3 else "MEDIUM"

        hotspots.append({
            "corridor_id": r.id,
            "corridor_name": r.name,
            "location_type": "Corridor Segment",
            "conflict_count": count,
            "peak_period": peak_str,
            "primary_interaction": top_interaction,
            "calculated_concern": concern,
            "data_confidence": confidence,
            "mean_ttc": mean_ttc,
            "label": "Danger-Zone Candidate",
        })

    return sorted(hotspots, key=lambda h: h["conflict_count"], reverse=True)
