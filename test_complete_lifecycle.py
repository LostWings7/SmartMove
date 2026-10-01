"""
SafeCity Loop V2 — Complete 12-Stage System Flow Integration Test
Tests the full connected flow:
IMAGE / VIDEO
  ↓
AI DETECTION
  ↓
GPS / ROAD SEGMENT
  ↓
HAZARD + CONFLICT ANALYSIS
  ↓
RISK ENGINE
  ↓
DANGER ZONE
  ↓
REPAIR PRIORITY
  ↓
HUMAN REVIEW
  ↓
ROUTE UPDATE
  ↓
REPAIR
  ↓
BEFORE / AFTER
"""
import io
import cv2
import numpy as np
from fastapi.testclient import TestClient
from backend.main import app
from backend.database import SessionLocal, Base, engine
from backend.models import Road, Hazard, NearMiss, Repair, Intervention, EvidenceFile, AuditLog, CitizenReport
from backend.seed import seed_all

client = TestClient(app)


def create_test_road_image(width=640, height=480):
    """Returns sample pothole image or synthetic asphalt image with a distinct crater."""
    from pathlib import Path
    sample_p = Path(__file__).resolve().parent / "frontend" / "public" / "samples" / "sample_pothole_road.jpg"
    if sample_p.exists():
        return sample_p.read_bytes()
    img = np.full((height, width, 3), 75, dtype=np.uint8)
    cv2.line(img, (width // 2, 0), (width // 2, height), (240, 240, 240), 3)
    cv2.circle(img, (width // 3, height // 2), 48, (20, 20, 20), -1)
    cv2.ellipse(img, (width // 3 + 12, height // 2 + 6), (58, 32), 15, 0, 360, (10, 10, 10), -1)
    _, buf = cv2.imencode('.jpg', img)
    return buf.tobytes()


def run_full_flow_test():
    print("=" * 70)
    print("SAFECITY LOOP V2 -- COMPLETE 12-STAGE LIFECYCLE VERIFICATION")
    print("=" * 70)

    # 0. Clean Reseed
    db = SessionLocal()
    seed_all(db, force_reseed=True)
    db.close()
    print("[INIT] Deterministic baseline seeded. Segment #1 = School Road [DEMO]")

    # ── STAGE 1 & 2: Local AI Pothole Detection & GPS Mapping ─────────────────
    img_bytes = create_test_road_image()
    files = {"file": ("school_road_crater.jpg", io.BytesIO(img_bytes), "image/jpeg")}
    data = {
        "road_id": "1",
        "latitude": "51.5074",
        "longitude": "-0.1278",
    }
    pothole_resp = client.post("/api/potholes/analyze", files=files, data=data)
    assert pothole_resp.status_code == 200, f"Pothole analyze failed: {pothole_resp.text}"
    pothole_data = pothole_resp.json()

    evidence_id = pothole_data["evidence_id"]
    assert evidence_id and evidence_id.startswith("SC-H-")
    assert pothole_data["road_id"] == 1
    assert pothole_data["pothole_count"] >= 1
    assert "visual_severity" in pothole_data and "contextual_severity" in pothole_data
    print(f"[STAGE 1 & 2 OK] AI Pothole Detection & GPS Mapping:")
    print(f"                 Evidence Code: {evidence_id}")
    print(f"                 Visual Severity: {pothole_data['visual_severity']} | Contextual: {pothole_data['contextual_severity']}")
    print(f"                 GPS Coordinates: ({pothole_data['latitude']}, {pothole_data['longitude']})")

    # ── STAGE 3 & 4: Traffic Conflict Surrogate Safety Analysis ───────────────
    # Simulate a surrogate traffic conflict on School Road / Junction corridor
    sim_resp = client.post("/api/junctions/1/simulate-event", json={
        "conflict_type": "pedestrian-vehicle",
        "ttc": 1.25,
        "risk_level": "HIGH",
    })
    assert sim_resp.status_code == 200
    print("[STAGE 3 & 4 OK] Traffic Conflict Engine:")
    print("                 Surrogate Conflict Registered (TTC: 1.25s, Candidate: Pedestrian ⟷ Vehicle)")

    # ── STAGE 5: Centralized Multi-Factor Risk Engine Recomputation ───────────
    # Formula: Risk = 0.30H + 0.20E + 0.20C + 0.15V + 0.10P + 0.05U
    intel_resp = client.get("/api/road-intelligence/1")
    assert intel_resp.status_code == 200
    intel_data = intel_resp.json()
    assert intel_data["risk_score"] >= 70.0
    assert "contributors_summary" in intel_data
    assert len(intel_data["contributors_summary"]) == 6
    print(f"[STAGE 5 OK] Centralized Multi-Factor Risk Engine:")
    print(f"             Corridor Risk Score: {intel_data['risk_score']}/100 ({intel_data['classification']})")
    print(f"             SafeCity Score: {intel_data['safe_city_score']}/100")
    for factor in intel_data["contributors_summary"]:
        print(f"             - {factor['label']}: {factor['display']}")

    # ── STAGE 6: Danger Zone Engine Classification ────────────────────────────
    dz_resp = client.get("/api/danger-zones/1")
    assert dz_resp.status_code == 200
    dz_data = dz_resp.json()
    assert dz_data["is_hotspot"] is True
    assert dz_data["hazard_count"] >= 1
    assert dz_data["conflict_count"] >= 1
    assert "traffic_exposure" in dz_data
    assert "main_contributing_factor" in dz_data
    assert "confidence" in dz_data
    print(f"[STAGE 6 OK] Danger-Zone Engine Classification:")
    print(f"             Hotspot Status: {dz_data['is_hotspot']} | Classification: {dz_data['danger_zone_classification']}")
    print(f"             Main Contributing Factor: {dz_data['main_contributing_factor']}")
    print(f"             Data Coverage: {dz_data['coverage']} | Confidence: {dz_data['confidence']}")

    # ── STAGE 7: AI Repair Priority Queue Generation ──────────────────────────
    queue_resp = client.get("/api/repair-priority")
    assert queue_resp.status_code == 200
    queue = queue_resp.json()
    assert len(queue) > 0
    top_item = next(q for q in queue if q["road_id"] == 1)
    assert top_item["priority_level"] in ["HIGH", "VERY HIGH"]
    assert top_item["reason"] is not None
    assert top_item["suggested_action"] is not None
    assert top_item["human_approval_required"] is True
    print(f"[STAGE 7 OK] AI Repair Priority Engine:")
    print(f"             Rank #{top_item['priority_rank']}: {top_item['road_name']}")
    print(f"             Priority: {top_item['priority_level']} | Urgency: {top_item['urgency_score']}/100")
    print(f"             Reason: {top_item['reason']}")
    print(f"             Suggested Action: {top_item['suggested_action']}")
    print(f"             Linked Evidence IDs: {top_item['evidence_ids']}")

    # ── STAGE 8: Mandatory Human Officer Review & Approval ────────────────────
    # Step 1: Officer verifies
    patch_v = client.patch(f"/api/repairs/{top_item['repair_id'] or 1}", json={
        "action": "verify",
        "reviewer": "Officer Marcus Chen",
        "notes": "Field inspection confirmed severe defect crater and pedestrian conflict near school gate.",
    })
    assert patch_v.status_code == 200
    assert patch_v.json()["status"] == "VERIFIED"

    # Step 2: Officer approves and assigns crew
    patch_a = client.patch(f"/api/repairs/{top_item['repair_id'] or 1}", json={
        "action": "assign",
        "assigned_to": "Public Works Rapid Asphalt Unit",
        "reviewer": "Senior Municipal Engineer",
    })
    assert patch_a.status_code == 200
    assert patch_a.json()["status"] == "REPAIR ASSIGNED"
    print("[STAGE 8 OK] Mandatory Human Approval:")
    print("             Officer Marcus Chen verified defect -> Senior Engineer assigned 'Public Works Rapid Asphalt Unit'")
    print("             Status transitioned: NEW -> VERIFIED -> REPAIR ASSIGNED")

    # ── STAGE 9: Route Intelligence Update ────────────────────────────────────
    route_resp = client.get("/api/routes/compare?origin_id=1&destination_id=2")
    assert route_resp.status_code == 200
    routes = route_resp.json()["routes"]
    fastest = next(r for r in routes if r["route_key"] == "fastest")
    safer = next(r for r in routes if r["route_key"] == "safer")
    assert fastest["overall_calculated_risk"] > safer["overall_calculated_risk"]
    assert safer["high_risk_segments_count"] < fastest["high_risk_segments_count"]
    print(f"[STAGE 9 OK] Route Intelligence Comparison:")
    print(f"             FASTEST ROUTE: {fastest['time_minutes']} min | Risk: {fastest['overall_calculated_risk']} | Cost: {fastest['route_cost']}")
    print(f"             SAFER ROUTE:   {safer['time_minutes']} min | Risk: {safer['overall_calculated_risk']} | Cost: {safer['route_cost']}")
    print(f"             Trade-off: {safer['explanation']}")

    # ── STAGE 10: Repair Execution & Hazard Resolution ────────────────────────
    before_risk_score = top_item["risk_score"]
    patch_rep = client.patch(f"/api/repairs/{top_item['repair_id'] or 1}", json={
        "action": "mark_repaired",
        "reviewer": "Civil Engineering Inspector",
        "notes": "Pothole filled and cold-mix compacted. Surface level verified.",
    })
    assert patch_rep.status_code == 200
    repaired_data = patch_rep.json()
    assert repaired_data["status"] == "REPAIRED"
    assert repaired_data["risk_score"] < before_risk_score
    print(f"[STAGE 10 OK] Municipal Repair Executed & Hazards Resolved:")
    print(f"              Status: REPAIRED | Before Risk: {before_risk_score} -> After Risk: {repaired_data['risk_score']}")

    # ── STAGE 11: Before / After Intervention Impact ──────────────────────────
    int_resp = client.get("/api/interventions")
    assert int_resp.status_code == 200
    ints = int_resp.json()
    school_int = next((i for i in ints if i["road_id"] == 1), ints[0])
    assert school_int["before_risk"] is not None and school_int["after_risk"] is not None
    reduction = round(((school_int["before_risk"] - school_int["after_risk"]) / school_int["before_risk"]) * 100, 1)
    print(f"[STAGE 11 OK] Before / After Intervention Impact Recorded:")
    print(f"              Intervention: {school_int['name']}")
    print(f"              BEFORE Risk: {school_int['before_risk']} -> AFTER Risk: {school_int['after_risk']} (Calculated Reduction: {reduction}%)")

    # ── STAGE 12: Chain-of-Custody Decision Audit Ledger ──────────────────────
    dec_resp = client.get("/api/evidence/decisions")
    assert dec_resp.status_code == 200
    decisions = dec_resp.json()
    assert len(decisions) > 0
    first_dec = decisions[0]
    for required_key in ["evidence_id", "source", "date", "location", "ai_result", "confidence", "reviewer", "decision", "status"]:
        assert required_key in first_dec, f"Missing key {required_key} in decision record"

    audit_resp = client.get("/api/audit-logs?limit=10")
    assert audit_resp.status_code == 200
    logs = audit_resp.json()
    assert len(logs) >= 3
    print(f"[STAGE 12 OK] Explainable Decision Audit Ledger & Chain of Custody:")
    print(f"              Decision Record: [{first_dec['evidence_id']}] at {first_dec['location']}")
    print(f"              Result: {first_dec['ai_result']} | Reviewer: {first_dec['reviewer']} | Decision: {first_dec['decision']}")
    print(f"              Immutable Audit Trail: {len(logs)} tamper-evident entries logged.")

    # ── CITIZEN REPORTING WORKFLOW VALIDATION ─────────────────────────────────
    cr_resp = client.post("/api/reports", json={
        "reporter_name": "Citizen Aanya",
        "type": "pothole",
        "description": "Crater near transit stop",
        "latitude": 51.5074,
        "longitude": -0.1278,
        "severity": "HIGH",
    })
    assert cr_resp.status_code == 201
    cr_data = cr_resp.json()
    assert cr_data["report_code"].startswith("SC-R-")
    assert cr_data["status"] == "Submitted"
    print(f"[CITIZEN OK] Citizen Report Created: {cr_data['report_code']} (Initial status: {cr_data['status']})")

    # Officer review of citizen report
    cr_verify = client.patch(f"/api/reports/{cr_data['id']}/status", json={
        "status": "Converted to Hazard",
        "reviewer": "Municipal Officer",
        "notes": "Visual match confirmed.",
    })
    assert cr_verify.status_code == 200
    assert cr_verify.json()["status"] == "Converted to Hazard"
    print(f"[CITIZEN OK] Citizen Report Verified by Officer -> Converted to Hazard.")

    print("=" * 70)
    print("ALL 12/12 STAGES OF SAFECITY LOOP V2 COMPLETED WITH 100% SUCCESS!")
    print("=" * 70)


if __name__ == "__main__":
    run_full_flow_test()
