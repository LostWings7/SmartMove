"""
SafeCity Loop V2 — Deterministic Demo Data Seeder
Populates a realistic, cohesive smart city road safety network.
⚠️ ALL DATA IS DETERMINISTIC DEMO DATA — NOT REAL MUNICIPAL DATA.
Locations: School Road, Market Junction, Hospital Road, Industrial Road, Main Road Corridor, Residential Lane.
"""
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from backend.models import (
    Road, Hazard, NearMiss, Junction, Repair, Intervention,
    CitizenReport, RiskSnapshot, EvidenceFile, AuditLog,
    User, UserRole, ComplaintCategory, Complaint, ComplaintStatus,
    ComplaintPriority, ComplaintStatusHistory, ComplaintEvidence, Notification
)
from backend.database import Base, engine, SessionLocal
from backend.auth import hash_password


def seed_all(db: Session, force_reseed: bool = False) -> None:
    """
    Deterministically seeds demo data if tables are empty or if force_reseed is True.
    Zero random generation — completely deterministic on every run.
    """
    if not force_reseed and db.query(Road).count() > 0:
        seed_categories(db)
        seed_demo_users(db)
        seed_demo_complaints(db)
        return

    print("[SEED] Seeding DETERMINISTIC SafeCity Loop V2 Demo Data...")

    # Recreate tables to ensure any new columns exist if force_reseed
    if force_reseed:
        db.rollback()
        Base.metadata.drop_all(bind=engine)
        Base.metadata.create_all(bind=engine)

    now = datetime(2026, 9, 23, 12, 0, 0)

    # ── 1. Roads (RoadSegments) ───────────────────────────────────────────────
    roads_seed = [
        {
            "id": 1,
            "name": "School Road [DEMO]",
            "latitude": 51.5074,
            "longitude": -0.1278,
            "road_type": "school_zone",
            "traffic_exposure": "HIGH",
            "vulnerability": "HIGH",
            "vulnerability_score": 92.0,
            "importance_score": 95.0,
            "risk_score": 87.5,
            "risk_confidence": 0.91,
            "safe_city_score": 12.5,
            "data_coverage": 88.0,
            "last_observed": now - timedelta(hours=4),
            "status": "pending-repair",
            "geometry": [[51.5065, -0.1285], [51.5074, -0.1278], [51.5085, -0.1270]],
        },
        {
            "id": 2,
            "name": "Market Junction [DEMO]",
            "latitude": 51.5120,
            "longitude": -0.1190,
            "road_type": "urban_arterial",
            "traffic_exposure": "VERY HIGH",
            "vulnerability": "MEDIUM",
            "vulnerability_score": 68.0,
            "importance_score": 85.0,
            "risk_score": 76.0,
            "risk_confidence": 0.88,
            "safe_city_score": 24.0,
            "data_coverage": 92.0,
            "last_observed": now - timedelta(hours=2),
            "status": "at-risk",
            "geometry": [[51.5110, -0.1200], [51.5120, -0.1190], [51.5130, -0.1180]],
        },
        {
            "id": 3,
            "name": "Hospital Road [DEMO]",
            "latitude": 51.5030,
            "longitude": -0.1350,
            "road_type": "transit_corridor",
            "traffic_exposure": "MEDIUM",
            "vulnerability": "HIGH",
            "vulnerability_score": 82.0,
            "importance_score": 90.0,
            "risk_score": 64.0,
            "risk_confidence": 0.86,
            "safe_city_score": 36.0,
            "data_coverage": 84.0,
            "last_observed": now - timedelta(hours=8),
            "status": "monitored",
            "geometry": [[51.5020, -0.1360], [51.5030, -0.1350], [51.5040, -0.1340]],
        },
        {
            "id": 4,
            "name": "Industrial Road [DEMO]",
            "latitude": 51.5200,
            "longitude": -0.1100,
            "road_type": "industrial",
            "traffic_exposure": "HIGH",
            "vulnerability": "LOW",
            "vulnerability_score": 35.0,
            "importance_score": 60.0,
            "risk_score": 53.0,
            "risk_confidence": 0.79,
            "safe_city_score": 47.0,
            "data_coverage": 65.0,
            "last_observed": now - timedelta(hours=28),
            "status": "monitored",
            "geometry": [[51.5190, -0.1110], [51.5200, -0.1100], [51.5215, -0.1085]],
        },
        {
            "id": 5,
            "name": "Main Road Corridor [DEMO]",
            "latitude": 51.4980,
            "longitude": -0.1400,
            "road_type": "urban_arterial",
            "traffic_exposure": "MEDIUM",
            "vulnerability": "LOW",
            "vulnerability_score": 25.0,
            "importance_score": 75.0,
            "risk_score": 36.5,
            "risk_confidence": 0.82,
            "safe_city_score": 63.5,
            "data_coverage": 70.0,
            "last_observed": now - timedelta(hours=14),
            "status": "monitored",
            "geometry": [[51.4965, -0.1415], [51.4980, -0.1400], [51.4995, -0.1385]],
        },
        {
            "id": 6,
            "name": "Residential Lane [DEMO]",
            "latitude": 51.5150,
            "longitude": -0.1320,
            "road_type": "local_residential",
            "traffic_exposure": "LOW",
            "vulnerability": "LOW",
            "vulnerability_score": 20.0,
            "importance_score": 35.0,
            "risk_score": 19.0,
            "risk_confidence": 0.65,
            "safe_city_score": 81.0,
            "data_coverage": 42.0,
            "last_observed": now - timedelta(days=6),
            "status": "monitored",
            "geometry": [[51.5140, -0.1330], [51.5150, -0.1320], [51.5160, -0.1310]],
        },
    ]

    roads = []
    for rd in roads_seed:
        r = Road(**rd)
        db.add(r)
        roads.append(r)
    db.commit()

    # ── 2. Junctions ──────────────────────────────────────────────────────────
    junctions_seed = [
        {"id": 1, "name": "Market Junction North [DEMO]", "latitude": 51.5125, "longitude": -0.1185, "risk_score": 82.0, "camera_id": "CAM-001", "status": "active"},
        {"id": 2, "name": "School Road Crossing [DEMO]", "latitude": 51.5080, "longitude": -0.1270, "risk_score": 91.0, "camera_id": "CAM-002", "status": "active"},
        {"id": 3, "name": "Hospital Entrance [DEMO]", "latitude": 51.5035, "longitude": -0.1345, "risk_score": 62.0, "camera_id": "CAM-003", "status": "active"},
        {"id": 4, "name": "Industrial Gate [DEMO]", "latitude": 51.5205, "longitude": -0.1095, "risk_score": 54.0, "camera_id": "CAM-004", "status": "active"},
    ]
    junctions = []
    for jd in junctions_seed:
        j = Junction(**jd)
        db.add(j)
        junctions.append(j)
    db.commit()

    # ── 3. Hazards (HazardObservations) ───────────────────────────────────────
    hazards_seed = [
        # School Road
        {"road_id": 1, "type": "pothole", "severity": "CRITICAL", "visual_severity": "CRITICAL", "contextual_severity": "CRITICAL", "confidence": 0.94, "risk_score": 92.0, "latitude": 51.5076, "longitude": -0.1276, "gps_accuracy": 3.8, "detected_at": now - timedelta(hours=4), "status": "active", "source": "AI Vision", "verified": True, "evidence_id": "EV-2026-0001", "evidence_code": "SC-H-1001", "direction": "Northbound Lane 1"},
        {"road_id": 1, "type": "pothole", "severity": "HIGH", "visual_severity": "HIGH", "contextual_severity": "CRITICAL", "confidence": 0.89, "risk_score": 84.0, "latitude": 51.5072, "longitude": -0.1280, "gps_accuracy": 4.1, "detected_at": now - timedelta(hours=18), "status": "active", "source": "AI Vision", "verified": True, "evidence_id": "EV-2026-0002", "evidence_code": "SC-H-1002", "direction": "Northbound Lane 2"},
        {"road_id": 1, "type": "dangerous-junction", "severity": "HIGH", "visual_severity": "HIGH", "contextual_severity": "HIGH", "confidence": 0.92, "risk_score": 86.0, "latitude": 51.5080, "longitude": -0.1271, "gps_accuracy": 3.2, "detected_at": now - timedelta(days=1), "status": "active", "source": "Municipal Inspection", "verified": True, "evidence_id": "EV-2026-0003", "evidence_code": "SC-H-1003", "direction": "Intersection Crossing"},

        # Market Junction
        {"road_id": 2, "type": "pothole", "severity": "HIGH", "visual_severity": "HIGH", "contextual_severity": "HIGH", "confidence": 0.87, "risk_score": 78.0, "latitude": 51.5122, "longitude": -0.1188, "gps_accuracy": 4.5, "detected_at": now - timedelta(hours=6), "status": "active", "source": "AI Vision", "verified": True, "evidence_id": "EV-2026-0004", "evidence_code": "SC-H-1004", "direction": "Eastbound Bus Bay"},
        {"road_id": 2, "type": "road-work", "severity": "MEDIUM", "visual_severity": "MEDIUM", "contextual_severity": "HIGH", "confidence": 0.78, "risk_score": 62.0, "latitude": 51.5118, "longitude": -0.1192, "gps_accuracy": 5.0, "detected_at": now - timedelta(days=2), "status": "active", "source": "Citizen Report", "verified": False, "evidence_id": "EV-2026-0005", "evidence_code": "SC-H-1005", "direction": "Westbound Shoulder"},

        # Hospital Road
        {"road_id": 3, "type": "pothole", "severity": "MEDIUM", "visual_severity": "MEDIUM", "contextual_severity": "HIGH", "confidence": 0.85, "risk_score": 65.0, "latitude": 51.5032, "longitude": -0.1348, "gps_accuracy": 4.2, "detected_at": now - timedelta(hours=12), "status": "active", "source": "AI Vision", "verified": True, "evidence_id": "EV-2026-0006", "evidence_code": "SC-H-1006", "direction": "Ambulance Lane"},
        {"road_id": 3, "type": "near-miss", "severity": "HIGH", "visual_severity": "HIGH", "contextual_severity": "HIGH", "confidence": 0.88, "risk_score": 74.0, "latitude": 51.5036, "longitude": -0.1344, "gps_accuracy": 3.5, "detected_at": now - timedelta(days=1), "status": "active", "source": "AI Vision", "verified": True, "evidence_id": "EV-2026-0007", "evidence_code": "SC-H-1007", "direction": "Emergency Bay Crossing"},

        # Industrial Road
        {"road_id": 4, "type": "pothole", "severity": "HIGH", "visual_severity": "HIGH", "contextual_severity": "MEDIUM", "confidence": 0.82, "risk_score": 72.0, "latitude": 51.5202, "longitude": -0.1098, "gps_accuracy": 5.5, "detected_at": now - timedelta(hours=30), "status": "active", "source": "Citizen Report", "verified": True, "evidence_id": "EV-2026-0008", "evidence_code": "SC-H-1008", "direction": "Freight Lane"},

        # Main Road Corridor
        {"road_id": 5, "type": "pothole", "severity": "LOW", "visual_severity": "LOW", "contextual_severity": "LOW", "confidence": 0.79, "risk_score": 38.0, "latitude": 51.4982, "longitude": -0.1398, "gps_accuracy": 4.8, "detected_at": now - timedelta(days=3), "status": "monitoring", "source": "AI Vision", "verified": False, "evidence_id": "EV-2026-0009", "evidence_code": "SC-H-1009", "direction": "Southbound"},

        # Residential Lane
        {"road_id": 6, "type": "pothole", "severity": "LOW", "visual_severity": "LOW", "contextual_severity": "LOW", "confidence": 0.71, "risk_score": 28.0, "latitude": 51.5152, "longitude": -0.1318, "gps_accuracy": 6.2, "detected_at": now - timedelta(days=7), "status": "monitoring", "source": "Citizen Report", "verified": False, "evidence_id": "EV-2026-0010", "evidence_code": "SC-H-1010", "direction": "Cul-de-sac"},
    ]
    for hd in hazards_seed:
        db.add(Hazard(**hd))
    db.commit()

    # ── 4. Conflict Events (NearMisses) ───────────────────────────────────────
    conflicts_seed = [
        {
            "junction_id": 2,
            "road_id": 1,
            "video_id": "VID-2026-0922-01",
            "timestamp": now - timedelta(hours=3),
            "object_types": ["vehicle", "pedestrian"],
            "object_type_a": "car",
            "object_type_b": "pedestrian",
            "ttc": 1.15,
            "pet": 1.42,
            "minimum_distance": 1.8,
            "risk_level": "CRITICAL",
            "event_severity": "CRITICAL",
            "conflict_zone": "Pedestrian Crossing Corridor",
            "direction": "Northbound ⟷ Crosswalk",
            "confidence": 0.93,
            "source": "YOLOv8 + ByteTrack (Local Edge)",
            "review_status": "pending_review",
            "evidence_clip": "/uploads/traffic/sample_conflict_school.mp4",
        },
        {
            "junction_id": 1,
            "road_id": 2,
            "video_id": "VID-2026-0922-02",
            "timestamp": now - timedelta(hours=7),
            "object_types": ["motorcycle", "pedestrian"],
            "object_type_a": "motorcycle",
            "object_type_b": "pedestrian",
            "ttc": 1.45,
            "pet": 1.85,
            "minimum_distance": 2.2,
            "risk_level": "HIGH",
            "event_severity": "HIGH",
            "conflict_zone": "Intersection Center Merge",
            "direction": "Eastbound Turn ⟷ Median Crossing",
            "confidence": 0.90,
            "source": "YOLOv8 + ByteTrack (Local Edge)",
            "review_status": "reviewed",
            "evidence_clip": "/uploads/traffic/sample_conflict_market.mp4",
        },
        {
            "junction_id": 3,
            "road_id": 3,
            "video_id": "VID-2026-0921-01",
            "timestamp": now - timedelta(hours=19),
            "object_types": ["car", "pedestrian"],
            "object_type_a": "car",
            "object_type_b": "pedestrian",
            "ttc": 1.58,
            "pet": 2.10,
            "minimum_distance": 2.9,
            "risk_level": "HIGH",
            "event_severity": "HIGH",
            "conflict_zone": "Ambulance Bay Egress",
            "direction": "Southbound ⟷ Hospital Walkway",
            "confidence": 0.88,
            "source": "YOLOv8 + ByteTrack (Local Edge)",
            "review_status": "actioned",
            "evidence_clip": "/uploads/traffic/sample_conflict_hospital.mp4",
        },
        {
            "junction_id": 4,
            "road_id": 4,
            "video_id": "VID-2026-0920-01",
            "timestamp": now - timedelta(days=2),
            "object_types": ["truck", "car"],
            "object_type_a": "truck",
            "object_type_b": "car",
            "ttc": 1.82,
            "pet": 2.30,
            "minimum_distance": 3.4,
            "risk_level": "MEDIUM",
            "event_severity": "MEDIUM",
            "conflict_zone": "Industrial Gate T-Junction",
            "direction": "Westbound Turn ⟷ Gate Entry",
            "confidence": 0.86,
            "source": "YOLOv8 + ByteTrack (Local Edge)",
            "review_status": "reviewed",
            "evidence_clip": "/uploads/traffic/sample_conflict_industrial.mp4",
        },
    ]
    for cd in conflicts_seed:
        db.add(NearMiss(**cd))
    db.commit()

    # ── 5. Repairs (RepairRecords) ────────────────────────────────────────────
    repairs_seed = [
        {
            "road_id": 1,
            "priority": 1,
            "reason": "Severe pothole cluster in active school zone with 92% pedestrian vulnerability score",
            "status": "in-progress",
            "assigned_to": "Metro Asphalt Unit Alpha",
            "assigned_department": "Municipal Road Maintenance",
            "recommended_action": "Milling & Full-Depth Asphalt Infill",
            "approval_status": "approved",
            "repair_date": now + timedelta(days=1),
            "before_score": 87.5,
            "after_score": 28.0,
            "before_observations": 3,
            "after_observations": 0,
            "created_at": now - timedelta(days=2),
        },
        {
            "road_id": 2,
            "priority": 2,
            "reason": "Arterial junction surface fatigue causing lateral swerving across active lanes",
            "status": "pending",
            "assigned_to": "Contractor Rapid Patch Team",
            "assigned_department": "District Highways Division",
            "recommended_action": "Thermal Patch Resurfacing & Anti-Skid Re-texture",
            "approval_status": "approved",
            "repair_date": now + timedelta(days=3),
            "before_score": 76.0,
            "after_score": 32.0,
            "before_observations": 2,
            "after_observations": 0,
            "created_at": now - timedelta(days=1),
        },
        {
            "road_id": 3,
            "priority": 3,
            "reason": "Pavement depression near emergency corridor access",
            "status": "pending",
            "assigned_to": "City Maintenance Crew 4",
            "assigned_department": "Municipal Road Maintenance",
            "recommended_action": "Localized Micro-Surfacing",
            "approval_status": "pending_approval",
            "repair_date": now + timedelta(days=5),
            "before_score": 64.0,
            "after_score": 25.0,
            "before_observations": 2,
            "after_observations": 0,
            "created_at": now - timedelta(hours=16),
        },
        {
            "road_id": 4,
            "priority": 4,
            "reason": "Heavy axle load structural rutting on freight access corridor",
            "status": "completed",
            "assigned_to": "Heavy Freight Logistics Infrastructure Team",
            "assigned_department": "Port & Industrial Roads Board",
            "recommended_action": "Reinforced Concrete Base Replacement",
            "approval_status": "approved",
            "repair_date": now - timedelta(days=4),
            "before_score": 84.0,
            "after_score": 53.0,
            "before_observations": 4,
            "after_observations": 1,
            "created_at": now - timedelta(days=14),
        },
    ]
    for rd in repairs_seed:
        db.add(Repair(**rd))
    db.commit()

    # ── 6. Interventions (Before / After Evidence) ────────────────────────────
    interventions_seed = [
        {
            "road_id": 4,
            "name": "Industrial Gate Heavy Vehicle Surface Reconstruction [DEMO]",
            "type": "resurfacing",
            "before_risk": 84.0,
            "after_risk": 53.0,
            "before_near_misses": 6,
            "after_near_misses": 1,
            "implemented_at": now - timedelta(days=4),
            "notes": "Replaced fatigued asphalt top layer with polymer-modified bitumen. Reduced heavy brake slippage near access gates.",
        },
        {
            "road_id": 5,
            "name": "Main Road Pedestrian Island & High-Visibility Crossing [DEMO]",
            "type": "signage",
            "before_risk": 72.0,
            "after_risk": 36.5,
            "before_near_misses": 8,
            "after_near_misses": 2,
            "implemented_at": now - timedelta(days=21),
            "notes": "Installed central refuge island and solar-illuminated pedestrian signage. Noticeable drop in mid-block crossing conflicts.",
        },
        {
            "road_id": 6,
            "name": "Residential Lane Chicanes & 20mph Traffic Calming [DEMO]",
            "type": "speed-bump",
            "before_risk": 58.0,
            "after_risk": 19.0,
            "before_near_misses": 4,
            "after_near_misses": 0,
            "implemented_at": now - timedelta(days=35),
            "notes": "Added dual chicanes to eliminate rat-running shortcuts between commercial avenues.",
        },
    ]
    for idata in interventions_seed:
        db.add(Intervention(**idata))
    db.commit()

    # ── 7. Evidence Files ─────────────────────────────────────────────────────
    evidence_seed = [
        {
            "evidence_code": "EV-2026-0001",
            "road_id": 1,
            "hazard_id": 1,
            "file_type": "image",
            "file_url": "/uploads/potholes/school_rd_crater_01.jpg",
            "thumbnail_url": "/uploads/potholes/school_rd_crater_01.jpg",
            "captured_at": now - timedelta(hours=4),
            "metadata_json": {"bounding_boxes": [[120, 85, 260, 195]], "confidence": 0.94, "severity": "CRITICAL"},
            "verified": True,
            "notes": "Deep crater located directly in pedestrian walking path of school crossing.",
        },
        {
            "evidence_code": "EV-2026-0002",
            "road_id": 1,
            "hazard_id": 2,
            "file_type": "image",
            "file_url": "/uploads/potholes/school_rd_crater_02.jpg",
            "thumbnail_url": "/uploads/potholes/school_rd_crater_02.jpg",
            "captured_at": now - timedelta(hours=18),
            "metadata_json": {"bounding_boxes": [[90, 110, 210, 220]], "confidence": 0.89, "severity": "HIGH"},
            "verified": True,
            "notes": "Spalling road surface with loose gravel scatter.",
        },
        {
            "evidence_code": "EV-2026-0003",
            "road_id": 2,
            "conflict_id": 2,
            "file_type": "video_clip",
            "file_url": "/uploads/traffic/market_crossing_clip.mp4",
            "thumbnail_url": "/uploads/traffic/market_crossing_thumb.jpg",
            "captured_at": now - timedelta(hours=7),
            "metadata_json": {"ttc": 1.45, "minimum_distance_m": 2.2, "object_types": ["vehicle", "cyclist"]},
            "verified": True,
            "notes": "Left-turning van came within 2.2m of oncoming cyclist.",
        },
        {
            "evidence_code": "EV-2026-0004",
            "road_id": 3,
            "hazard_id": 6,
            "file_type": "image",
            "file_url": "/uploads/potholes/hospital_rd_pothole.jpg",
            "thumbnail_url": "/uploads/potholes/hospital_rd_pothole.jpg",
            "captured_at": now - timedelta(hours=12),
            "metadata_json": {"bounding_boxes": [[140, 95, 240, 180]], "confidence": 0.85, "severity": "MEDIUM"},
            "verified": True,
            "notes": "Depression on ambulance curb approach.",
        },
    ]
    for ev in evidence_seed:
        db.add(EvidenceFile(**ev))
    db.commit()

    # ── 8. Risk Snapshots (Historical Trend Analytics) ────────────────────────
    # 7-day risk trajectory for School Road, Market Junction, Hospital Road
    for day_offset in range(7, -1, -1):
        snap_time = now - timedelta(days=day_offset)
        # School Road trending slightly upwards due to defect growth
        db.add(RiskSnapshot(
            road_id=1,
            timestamp=snap_time,
            risk_score=round(78.0 + (7 - day_offset) * 1.35, 1),
            confidence=0.91,
            data_coverage=88.0,
            hazard_severity=92.0,
            traffic_exposure=80.0,
            conflict_evidence=90.0,
            vulnerable_exposure=92.0,
            persistence=85.0,
            road_importance=95.0,
            contributing_factors={"hazard": 92.0, "vulnerability": 92.0}
        ))
        # Market Junction stable high
        db.add(RiskSnapshot(
            road_id=2,
            timestamp=snap_time,
            risk_score=round(74.0 + (7 - day_offset) * 0.28, 1),
            confidence=0.88,
            data_coverage=92.0,
            hazard_severity=78.0,
            traffic_exposure=95.0,
            conflict_evidence=82.0,
            vulnerable_exposure=68.0,
            persistence=70.0,
            road_importance=85.0,
            contributing_factors={"traffic": 95.0, "conflict": 82.0}
        ))
        # Hospital Road stable moderate
        db.add(RiskSnapshot(
            road_id=3,
            timestamp=snap_time,
            risk_score=round(62.0 + (7 - day_offset) * 0.28, 1),
            confidence=0.86,
            data_coverage=84.0,
            hazard_severity=65.0,
            traffic_exposure=50.0,
            conflict_evidence=74.0,
            vulnerable_exposure=82.0,
            persistence=60.0,
            road_importance=90.0,
            contributing_factors={"vulnerability": 82.0}
        ))
    db.commit()

    # ── 9. Audit Logs ─────────────────────────────────────────────────────────
    audit_seed = [
        {"timestamp": now - timedelta(hours=2), "actor": "Municipal Officer (E. Vance)", "action": "REPAIR_WORK_ORDER_ISSUED", "target_type": "RepairRecord", "target_id": 1, "details": "Approved Milling & Asphalt infill work order for School Road [DEMO]"},
        {"timestamp": now - timedelta(hours=4), "actor": "Vision Pipeline (YOLOv8)", "action": "CRITICAL_HAZARD_FLAGGED", "target_type": "HazardObservation", "target_id": 1, "details": "Detected CRITICAL pothole (confidence: 94%) on School Road [DEMO]"},
        {"timestamp": now - timedelta(hours=7), "actor": "Conflict Engine (ByteTrack)", "action": "NEAR_MISS_CONFLICT_RECORDED", "target_type": "ConflictEvent", "target_id": 2, "details": "TTC 1.45s conflict flagged between vehicle and cyclist at Market Junction [DEMO]"},
        {"timestamp": now - timedelta(hours=14), "actor": "Risk Engine", "action": "RISK_SCORE_RECOMPUTED", "target_type": "RoadSegment", "target_id": 1, "details": "Updated School Road calculated risk score to 87.5/100 (HIGH RISK)"},
        {"timestamp": now - timedelta(days=1), "actor": "Road Authority Admin", "action": "INTERVENTION_EVALUATION_LOGGED", "target_type": "Intervention", "target_id": 1, "details": "Logged before/after outcome for Industrial Road resurfacing"},
    ]
    for al in audit_seed:
        db.add(AuditLog(**al))
    db.commit()

    # ── 10. Citizen Reports ───────────────────────────────────────────────────
    reports_seed = [
        {
            "report_code": "SC-DEMO-00128",
            "reporter_name": "Aanya Sharma",
            "type": "pothole",
            "description": "Deep crater developing on northbound lane near school crossing bus stop",
            "latitude": 51.5074,
            "longitude": -0.1278,
            "severity": "HIGH",
            "status": "AI Verified",
            "submitted_at": now - timedelta(hours=5),
        },
        {
            "report_code": "SC-DEMO-00129",
            "reporter_name": "Marcus Chen",
            "type": "dangerous-junction",
            "description": "Vehicles regularly fail to yield to pedestrians on zebra crossing during morning rush",
            "latitude": 51.5120,
            "longitude": -0.1190,
            "severity": "HIGH",
            "status": "Under Review",
            "submitted_at": now - timedelta(hours=22),
        },
        {
            "report_code": "SC-DEMO-00130",
            "reporter_name": "Sarah Jenkins",
            "type": "pothole",
            "description": "Sunken utility trench creating tire impact near hospital access",
            "latitude": 51.5030,
            "longitude": -0.1350,
            "severity": "MEDIUM",
            "status": "AI Verification Pending",
            "submitted_at": now - timedelta(days=2),
        },
    ]
    for rep in reports_seed:
        db.add(CitizenReport(**rep))
    db.commit()

    print(f"[SEED] Successfully seeded: {len(roads_seed)} roads, {len(junctions_seed)} junctions, "
          f"{len(hazards_seed)} hazards, {len(conflicts_seed)} conflicts, {len(repairs_seed)} repairs, "
          f"{len(evidence_seed)} evidence files, and {len(audit_seed)} audit logs.")

    seed_categories(db)
    seed_demo_users(db)
    seed_demo_complaints(db)


def seed_categories(db: Session) -> None:
    if db.query(ComplaintCategory).count() > 0:
        return
    categories = [
        {"name": "Pothole", "code": "POTHOLE", "description": "Road surface cavity, pit, or deep depression posing tire and suspension danger.", "is_ai_supported": True},
        {"name": "Waterlogging", "code": "WATERLOGGING", "description": "Standing water accumulation, ponding, or clogged drainage channels on roadway.", "is_ai_supported": True},
        {"name": "Road Crack", "code": "ROAD_CRACK", "description": "Longitudinal, transverse, or alligator pattern asphalt structural cracking.", "is_ai_supported": True},
        {"name": "Road Surface Damage", "code": "ROAD_SURFACE_DAMAGE", "description": "Severe raveling, rutting, potholes, or widespread pavement disintegration.", "is_ai_supported": False},
        {"name": "Traffic Conflict", "code": "TRAFFIC_CONFLICT", "description": "Recurring surrogate near-miss events, unsafe crossing conflicts, or abrupt braking.", "is_ai_supported": True},
        {"name": "Dangerous Intersection", "code": "DANGEROUS_INTERSECTION", "description": "Uncontrolled junction, poor sightline, blind turns, or pedestrian hazard zone.", "is_ai_supported": True},
        {"name": "Blocked Road", "code": "BLOCKED_ROAD", "description": "Vehicle breakdown, unauthorized road closure, or impassable obstacle.", "is_ai_supported": False},
        {"name": "Damaged Road Infrastructure", "code": "DAMAGED_INFRASTRUCTURE", "description": "Broken guardrails, bent signage, damaged median barriers, or signal heads.", "is_ai_supported": False},
        {"name": "Street/Traffic Obstruction", "code": "STREET_OBSTRUCTION", "description": "Encroaching construction debris, fallen trees, or sightline impediments.", "is_ai_supported": False},
        {"name": "Other", "code": "OTHER", "description": "Other civic highway or urban mobility hazard.", "is_ai_supported": False},
    ]
    for cat in categories:
        db.add(ComplaintCategory(**cat))
    db.commit()
    print(f"[SEED] Seeded {len(categories)} authoritative complaint categories.")


def seed_demo_users(db: Session) -> None:
    if db.query(User).count() > 0:
        return
    users = [
        {
            "full_name": "Priya Sharma [DEMO CITIZEN]",
            "email": "citizen_demo@safecity.gov",
            "phone_number": "+91 98450 12345",
            "password_hash": hash_password("CitizenPass123!"),
            "role": UserRole.CITIZEN,
            "is_active": True,
            "is_verified": True,
        },
        {
            "full_name": "Marcus Chen [DEMO INSPECTOR]",
            "email": "inspector_demo@safecity.gov",
            "phone_number": "+91 98450 23456",
            "password_hash": hash_password("InspectorPass123!"),
            "role": UserRole.INSPECTOR,
            "is_active": True,
            "is_verified": True,
        },
        {
            "full_name": "Elena Rostova [DEMO ENGINEER]",
            "email": "engineer_demo@safecity.gov",
            "phone_number": "+91 98450 34567",
            "password_hash": hash_password("EngineerPass123!"),
            "role": UserRole.ENGINEER,
            "is_active": True,
            "is_verified": True,
        },
        {
            "full_name": "David Vance [DEMO ADMIN]",
            "email": "admin_demo@safecity.gov",
            "phone_number": "+91 98450 45678",
            "password_hash": hash_password("AdminPass123!"),
            "role": UserRole.ADMIN,
            "is_active": True,
            "is_verified": True,
        },
        {
            "full_name": "Chief Commissioner Rao [DEMO SUPERUSER]",
            "email": "superuser_demo@safecity.gov",
            "phone_number": "+91 98450 56789",
            "password_hash": hash_password("SuperPass123!"),
            "role": UserRole.SUPERUSER,
            "is_active": True,
            "is_verified": True,
        },
    ]
    for u in users:
        db.add(User(**u))
    db.commit()
    print(f"[SEED] Seeded {len(users)} RBAC demo accounts (Citizen, Inspector, Engineer, Admin, Superuser).")


def seed_demo_complaints(db: Session) -> None:
    if db.query(Complaint).count() > 0:
        return

    citizen = db.query(User).filter(User.role == UserRole.CITIZEN).first()
    inspector = db.query(User).filter(User.role == UserRole.INSPECTOR).first()
    engineer = db.query(User).filter(User.role == UserRole.ENGINEER).first()
    admin = db.query(User).filter(User.role == UserRole.ADMIN).first()

    if not citizen:
        return

    pothole_cat = db.query(ComplaintCategory).filter(ComplaintCategory.code == "POTHOLE").first()
    water_cat = db.query(ComplaintCategory).filter(ComplaintCategory.code == "WATERLOGGING").first()
    crack_cat = db.query(ComplaintCategory).filter(ComplaintCategory.code == "ROAD_CRACK").first()

    school_road = db.query(Road).filter(Road.id == 1).first()
    market_junc = db.query(Road).filter(Road.id == 2).first()
    hospital_road = db.query(Road).filter(Road.id == 3).first()

    now = datetime(2026, 9, 23, 14, 0, 0)

    # 1. Active pothole complaint
    c1 = Complaint(
        public_reference="SC-2026-000184",
        citizen_id=citizen.id,
        category_id=pothole_cat.id if pothole_cat else 1,
        title="Severe Pothole Cluster near Primary School Crossing",
        description="Multiple deep potholes formed right outside the school main gate. Vehicles swerving sharply into oncoming pedestrian path.",
        status=ComplaintStatus.UNDER_REVIEW,
        priority=ComplaintPriority.HIGH,
        location_lat=51.5074,
        location_lng=-0.1278,
        location_address="School Road Corridor, East Gate Section",
        road_segment_id=school_road.id if school_road else 1,
        assigned_to=inspector.id if inspector else None,
        assigned_department="Municipal Road Engineering",
        source="CITIZEN",
        ai_linked=True,
        ai_confidence=0.88,
        hazard_id=1,
        created_at=now - timedelta(hours=6),
        updated_at=now - timedelta(hours=2),
    )
    db.add(c1)
    db.flush()

    db.add(ComplaintStatusHistory(
        complaint_id=c1.id,
        old_status=None,
        new_status=ComplaintStatus.SUBMITTED,
        changed_by_id=citizen.id,
        changed_by_name=citizen.full_name,
        changed_by_role=citizen.role,
        comment="Complaint submitted with photographic evidence.",
        timestamp=now - timedelta(hours=6),
    ))
    db.add(ComplaintStatusHistory(
        complaint_id=c1.id,
        old_status=ComplaintStatus.SUBMITTED,
        new_status=ComplaintStatus.ACKNOWLEDGED,
        changed_by_id=admin.id if admin else citizen.id,
        changed_by_name=admin.full_name if admin else "System",
        changed_by_role=UserRole.ADMIN,
        comment="Complaint received and queued for field inspection.",
        timestamp=now - timedelta(hours=4),
    ))
    db.add(ComplaintStatusHistory(
        complaint_id=c1.id,
        old_status=ComplaintStatus.ACKNOWLEDGED,
        new_status=ComplaintStatus.UNDER_REVIEW,
        changed_by_id=inspector.id if inspector else citizen.id,
        changed_by_name=inspector.full_name if inspector else "Inspector",
        changed_by_role=UserRole.INSPECTOR,
        comment="AI Vision cross-reference identified candidate pothole observation #1. Verifying coordinates.",
        timestamp=now - timedelta(hours=2),
    ))

    # Evidence for c1
    db.add(ComplaintEvidence(
        complaint_id=c1.id,
        uploader_id=citizen.id,
        file_type="image",
        file_url="/uploads/potholes/demo_pothole_1.jpg",
        filename="school_road_pothole_cluster.jpg",
        mime_type="image/jpeg",
        file_size=245800,
        uploaded_at=now - timedelta(hours=6),
        privacy_status="processed",
        notes="Citizen upload: deep surface cavity right in front of school bus pickup lane.",
    ))

    # 2. Resolved waterlogging complaint with citizen confirmation pending
    c2 = Complaint(
        public_reference="SC-2026-000171",
        citizen_id=citizen.id,
        category_id=water_cat.id if water_cat else 2,
        title="Monsoon Waterlogging & Submerged Zebra Crosswalk",
        description="Drain inlet blocked by construction debris. 15cm deep standing water submerges pedestrian crossing.",
        status=ComplaintStatus.RESOLVED,
        priority=ComplaintPriority.HIGH,
        location_lat=51.5120,
        location_lng=-0.1190,
        location_address="Market Junction West Approach",
        road_segment_id=market_junc.id if market_junc else 2,
        assigned_to=engineer.id if engineer else None,
        assigned_department="Water & Drainage Division",
        source="CITIZEN",
        created_at=now - timedelta(days=2),
        updated_at=now - timedelta(hours=3),
        resolved_at=now - timedelta(hours=3),
        public_response="Rapid Drainage Crew deployed to flush blocked stormwater catch basin. Standing water cleared and flow re-established.",
        resolution_summary="Storm catch basin cleared. Resurfacing of damaged shoulder completed.",
    )
    db.add(c2)
    db.flush()

    db.add(ComplaintStatusHistory(
        complaint_id=c2.id,
        old_status=None,
        new_status=ComplaintStatus.SUBMITTED,
        changed_by_id=citizen.id,
        changed_by_name=citizen.full_name,
        changed_by_role=citizen.role,
        comment="Citizen report submitted.",
        timestamp=now - timedelta(days=2),
    ))
    db.add(ComplaintStatusHistory(
        complaint_id=c2.id,
        old_status=ComplaintStatus.SUBMITTED,
        new_status=ComplaintStatus.VERIFIED,
        changed_by_id=inspector.id if inspector else citizen.id,
        changed_by_name=inspector.full_name if inspector else "Inspector",
        changed_by_role=UserRole.INSPECTOR,
        comment="Verified 15cm flooding. Priority set to HIGH.",
        timestamp=now - timedelta(days=1, hours=18),
    ))
    db.add(ComplaintStatusHistory(
        complaint_id=c2.id,
        old_status=ComplaintStatus.VERIFIED,
        new_status=ComplaintStatus.IN_PROGRESS,
        changed_by_id=engineer.id if engineer else citizen.id,
        changed_by_name=engineer.full_name if engineer else "Engineer",
        changed_by_role=UserRole.ENGINEER,
        comment="Drain cleaning jetting vehicle dispatched.",
        timestamp=now - timedelta(hours=8),
    ))
    db.add(ComplaintStatusHistory(
        complaint_id=c2.id,
        old_status=ComplaintStatus.IN_PROGRESS,
        new_status=ComplaintStatus.RESOLVED,
        changed_by_id=engineer.id if engineer else citizen.id,
        changed_by_name=engineer.full_name if engineer else "Engineer",
        changed_by_role=UserRole.ENGINEER,
        comment="Catch basin flushed and clear water flow verified. Awaiting citizen confirmation.",
        timestamp=now - timedelta(hours=3),
    ))

    # 3. In-progress road crack
    c3 = Complaint(
        public_reference="SC-2026-000165",
        citizen_id=citizen.id,
        category_id=crack_cat.id if crack_cat else 3,
        title="Longitudinal Surface Crack Widening along Transit Lane",
        description="Fissure extending over 12 meters near hospital ambulance turn-in. Motorcycles slipping during rain.",
        status=ComplaintStatus.IN_PROGRESS,
        priority=ComplaintPriority.MEDIUM,
        location_lat=51.5030,
        location_lng=-0.1350,
        location_address="Hospital Road Northbound Corridor",
        road_segment_id=hospital_road.id if hospital_road else 3,
        assigned_to=engineer.id if engineer else None,
        assigned_department="Municipal Public Works",
        source="CITIZEN",
        created_at=now - timedelta(days=3),
        updated_at=now - timedelta(hours=5),
        public_response="Cracking sealed with bituminous hot-pour filler. Final curing in progress.",
    )
    db.add(c3)
    db.flush()

    db.add(ComplaintStatusHistory(
        complaint_id=c3.id,
        old_status=None,
        new_status=ComplaintStatus.SUBMITTED,
        changed_by_id=citizen.id,
        changed_by_name=citizen.full_name,
        changed_by_role=citizen.role,
        comment="Submitted by citizen.",
        timestamp=now - timedelta(days=3),
    ))
    db.add(ComplaintStatusHistory(
        complaint_id=c3.id,
        old_status=ComplaintStatus.SUBMITTED,
        new_status=ComplaintStatus.ASSIGNED,
        changed_by_id=admin.id if admin else citizen.id,
        changed_by_name=admin.full_name if admin else "Admin",
        changed_by_role=UserRole.ADMIN,
        comment="Assigned to Elena Rostova (Engineer).",
        timestamp=now - timedelta(days=2),
    ))
    db.add(ComplaintStatusHistory(
        complaint_id=c3.id,
        old_status=ComplaintStatus.ASSIGNED,
        new_status=ComplaintStatus.IN_PROGRESS,
        changed_by_id=engineer.id if engineer else citizen.id,
        changed_by_name=engineer.full_name if engineer else "Engineer",
        changed_by_role=UserRole.ENGINEER,
        comment="Hot-pour bitumen crack sealing started.",
        timestamp=now - timedelta(hours=5),
    ))

    # Notifications for citizen
    db.add(Notification(
        user_id=citizen.id,
        type="municipal_response",
        title="Municipal Response on SC-2026-000171",
        message="Your waterlogging complaint has been marked RESOLVED. Please inspect and confirm resolution.",
        related_complaint_id=c2.id,
        is_read=False,
        created_at=now - timedelta(hours=3),
    ))
    db.add(Notification(
        user_id=citizen.id,
        type="status_update",
        title="Status Update on SC-2026-000184",
        message="Your pothole complaint is now UNDER REVIEW by Inspector Marcus Chen.",
        related_complaint_id=c1.id,
        is_read=True,
        created_at=now - timedelta(hours=2),
    ))

    db.commit()
    print("[SEED] Seeded 3 comprehensive demo complaints with status histories, evidence, and citizen notifications.")


if __name__ == "__main__":
    db = SessionLocal()
    try:
        seed_all(db, force_reseed=True)
    finally:
        db.close()
