"""
SafeCity Loop V2 — Voice Agent Phase 1 Test Suite
Verifies:
1. LiveKit and Google Gemini Realtime agent structure & tool registry
2. Authoritative SafeCity read-only database tools (dashboard, complaints, hazards, conflicts, routes)
3. FRIDAY natural voice queries:
   - "Friday, what page am I on?"
   - "Friday, show my complaints."
   - "Friday, what is the status of complaint SC-XXXX?"
   - "Friday, show the active traffic conflicts."
   - "Friday, what are the dangerous areas?"
   - "Friday, what is the current road condition?"
4. Wake/idle state transitions ("sleep now", "friday")
5. RBAC security enforcement (no arbitrary actions, citizen vs staff permissions)
6. FastAPI endpoints (/api/voice/health, /api/voice/tools, /api/voice/query, /api/voice/context)
"""
import pytest
from fastapi.testclient import TestClient

from backend.main import app
from backend.database import SessionLocal
from backend.models import User, UserRole, Complaint, Road, Hazard, NearMiss
from backend.seed import seed_all
from backend.voice.context import SafeCityContext, context_manager
from backend.voice.permissions import VoicePermissions
from backend.voice.tools import (
    get_current_user,
    get_current_page,
    get_dashboard_summary,
    get_my_complaints,
    get_complaint_details,
    get_complaint_status,
    get_active_hazards,
    get_hazard_details,
    get_traffic_summary,
    get_traffic_conflicts,
    get_conflict_details,
    get_road_conditions,
    get_repair_status,
    get_route_summary,
    get_danger_zones,
    safecity_voice_tools,
    get_tool_registry,
)
from backend.voice.friday_agent import FridayAgent, get_friday_agent


@pytest.fixture(scope="module", autouse=True)
def setup_database():
    """Ensure database has fresh baseline seed data."""
    db = SessionLocal()
    try:
        seed_all(db)
    finally:
        db.close()


@pytest.fixture
def client():
    return TestClient(app)


# ── 1. Tool Registry & LiveKit Agent Initialization ───────────────────────────

def test_01_tool_registry_and_livekit_tools():
    """Verifies that all 15 SafeCity read-only tools are registered and valid."""
    tools = get_tool_registry()
    assert len(tools) >= 15
    assert "get_current_user" in tools
    assert "get_current_page" in tools
    assert "get_dashboard_summary" in tools
    assert "get_my_complaints" in tools
    assert "get_complaint_details" in tools
    assert "get_complaint_status" in tools
    assert "get_active_hazards" in tools
    assert "get_hazard_details" in tools
    assert "get_traffic_summary" in tools
    assert "get_traffic_conflicts" in tools
    assert "get_conflict_details" in tools
    assert "get_road_conditions" in tools
    assert "get_repair_status" in tools
    assert "get_route_summary" in tools
    assert "get_danger_zones" in tools

    # Check LiveKit FunctionTools
    assert len(safecity_voice_tools) >= 15
    tool_names = [t.info.name for t in safecity_voice_tools]
    assert "get_dashboard_summary" in tool_names
    assert "get_traffic_conflicts" in tool_names


# ── 2. User & UI Context Tools ────────────────────────────────────────────────

def test_02_get_current_user_and_page():
    """Verifies get_current_user and get_current_page behave accurately."""
    ctx = SafeCityContext(
        client_id="test_session",
        current_page="/citizen/complaints",
        current_page_title="My Citizen Complaints",
        current_user={"id": 1, "full_name": "Aanya Sharma", "role": "CITIZEN"},
        user_role="CITIZEN",
        is_authenticated=True,
        selected_complaint="SC-2026-000001",
    )

    user_res = get_current_user(context=ctx)
    assert user_res["success"] is True
    assert user_res["authenticated"] is True
    assert "Aanya Sharma" in user_res["voice_summary"]
    assert "CITIZEN" in user_res["voice_summary"]

    page_res = get_current_page(context=ctx)
    assert page_res["success"] is True
    assert page_res["current_page"] == "/citizen/complaints"
    assert "My Citizen Complaints" in page_res["voice_summary"]
    assert "SC-2026-000001" in page_res["voice_summary"]


# ── 3. Database Read-Only Intelligence Tools ──────────────────────────────────

def test_03_get_dashboard_summary_real_data():
    """Verifies dashboard summary queries actual database records."""
    res = get_dashboard_summary()
    assert res["success"] is True
    assert res["roads_monitored"] >= 5
    assert res["active_hazards"] >= 5
    assert res["traffic_conflicts_count"] >= 3
    assert res["pending_repairs_count"] >= 1
    assert res["average_city_risk"] > 0.0
    assert "citywide overview" in res["voice_summary"].lower()


def test_04_get_my_complaints_real_data():
    """Verifies citizen complaints retrieval."""
    res = get_my_complaints(limit=5)
    assert res["success"] is True
    assert res["total_found"] >= 1
    assert len(res["complaints"]) >= 1
    top = res["complaints"][0]
    assert top["reference"].startswith("SC-")
    assert top["status"] in ["SUBMITTED", "ACKNOWLEDGED", "UNDER_REVIEW", "VERIFIED"]
    assert "complaint" in res["voice_summary"].lower()


def test_05_get_complaint_status_and_details():
    """Verifies complaint lookup by reference code."""
    db = SessionLocal()
    try:
        complaint = db.query(Complaint).first()
        assert complaint is not None
        ref = complaint.public_reference

        # Test details
        detail_res = get_complaint_details(complaint_reference=ref, user_role="ADMIN")
        assert detail_res["success"] is True
        assert detail_res["reference"] == ref
        assert detail_res["title"] == complaint.title
        assert detail_res["status"] == complaint.status

        # Test status
        status_res = get_complaint_status(complaint_reference=ref, user_role="ADMIN")
        assert status_res["success"] is True
        assert status_res["reference"] == ref
        assert status_res["status"] == complaint.status
        assert ref in status_res["voice_summary"]
    finally:
        db.close()


def test_06_get_active_hazards_and_details():
    """Verifies active road hazards and defect telemetry."""
    db = SessionLocal()
    try:
        hazards_res = get_active_hazards(limit=5)
        assert hazards_res["success"] is True
        assert hazards_res["total_found"] >= 1
        top_h = hazards_res["hazards"][0]
        assert "pothole" in top_h["type"].lower() or "defect" in top_h["type"].lower()

        detail_res = get_hazard_details(hazard_id=top_h["id"])
        assert detail_res["success"] is True
        assert detail_res["id"] == top_h["id"]
        assert detail_res["confidence"] > 0.5
        assert "hazard" in detail_res["voice_summary"].lower()
    finally:
        db.close()


def test_07_get_traffic_summary_and_conflicts():
    """Verifies traffic conflict surrogate safety events (TTC, clearance)."""
    sum_res = get_traffic_summary()
    assert sum_res["success"] is True
    assert sum_res["total_conflicts"] >= 1
    assert "traffic telemetry reports" in sum_res["voice_summary"].lower()

    conflicts_res = get_traffic_conflicts(limit=5)
    assert conflicts_res["success"] is True
    assert conflicts_res["total_found"] >= 1
    top_c = conflicts_res["conflicts"][0]
    assert top_c["ttc"] is not None or top_c["minimum_distance_m"] is not None

    detail_res = get_conflict_details(conflict_id=top_c["id"])
    assert detail_res["success"] is True
    assert detail_res["id"] == top_c["id"]
    assert "clearance" in detail_res["voice_summary"].lower()


def test_08_get_road_conditions_and_danger_zones():
    """Verifies surface condition inspection and danger zones."""
    cond_res = get_road_conditions(location="School Road")
    assert cond_res["success"] is True
    assert len(cond_res["roads"]) >= 1
    assert "School Road" in cond_res["roads"][0]["name"]
    assert "risk is" in cond_res["voice_summary"].lower()

    danger_res = get_danger_zones(limit=5)
    assert danger_res["success"] is True
    assert len(danger_res["danger_zones"]) >= 1
    assert "danger zone" in danger_res["voice_summary"].lower()


def test_09_get_route_summary():
    """Verifies route intelligence (Fastest vs. Safer Route comparison)."""
    route_res = get_route_summary()
    assert route_res["success"] is True
    assert "fastest_route" in route_res
    assert "safer_route" in route_res
    assert route_res["risk_reduction_pct"] > 0.0
    assert "safer route" in route_res["voice_summary"].lower()


# ── 4. FRIDAY Natural Voice Queries (Phase 1 Acceptance Criteria) ──────────────

def test_10_friday_voice_queries():
    """
    Tests the exact voice utterances specified in Phase 1:
    - 'Friday, what page am I on?'
    - 'Friday, show my complaints.'
    - 'Friday, what is the status of complaint SC-XXXX?'
    - 'Friday, show the active traffic conflicts.'
    - 'Friday, what are the dangerous areas?'
    - 'Friday, what is the current road condition?'
    """
    agent = get_friday_agent()
    # Reset wake state
    agent.awaiting_wake = False

    ctx = SafeCityContext(
        client_id="query_test_session",
        current_page="/map",
        current_page_title="Interactive SafeCity GIS Hazard Map",
        current_user={"id": 1, "full_name": "Aanya Sharma", "role": "CITIZEN"},
        user_role="CITIZEN",
        is_authenticated=True,
    )

    # 1. Page query
    r1 = agent.process_query("Friday, what page am I on?", context=ctx)
    assert r1["success"] is True
    assert "interactive safecity gis hazard map" in r1["reply"].lower() or "/map" in r1["reply"].lower()
    assert r1["tools_called"][0]["tool"] == "get_current_page"

    # 2. My complaints query
    r2 = agent.process_query("Friday, show my complaints.", context=ctx)
    assert r2["success"] is True
    assert "complaint" in r2["reply"].lower()
    assert r2["tools_called"][0]["tool"] == "get_my_complaints"
    assert any(a["action"] == "navigate_to_page" for a in r2["frontend_actions"])

    # 3. Complaint status query
    r3 = agent.process_query("Friday, what is the status of complaint SC-2026-000001?", context=ctx)
    assert r3["success"] is True
    assert "sc-2026-000001" in r3["reply"].lower()
    assert r3["tools_called"][0]["tool"] == "get_complaint_status"

    # 4. Traffic conflicts query
    r4 = agent.process_query("Friday, show the active traffic conflicts.", context=ctx)
    assert r4["success"] is True
    assert "traffic conflict" in r4["reply"].lower()
    assert r4["tools_called"][0]["tool"] == "get_traffic_conflicts"

    # 5. Dangerous areas query
    r5 = agent.process_query("Friday, what are the dangerous areas?", context=ctx)
    assert r5["success"] is True
    assert "danger zone" in r5["reply"].lower() or "high-concern" in r5["reply"].lower()
    assert r5["tools_called"][0]["tool"] == "get_danger_zones"

    # 6. Road condition query
    r6 = agent.process_query("Friday, what is the current road condition on School Road?", context=ctx)
    assert r6["success"] is True
    assert "school road" in r6["reply"].lower()
    assert "risk" in r6["reply"].lower()
    assert r6["tools_called"][0]["tool"] == "get_road_conditions"


def test_11_friday_sleep_and_wake_behavior():
    """Verifies sleep command ('sleep now') and wake word ('Friday') handling."""
    agent = get_friday_agent()
    agent.awaiting_wake = False

    # Sleep command
    r_sleep = agent.process_query("FRIDAY, sleep now.")
    assert r_sleep["is_idle"] is True
    assert "going idle" in r_sleep["reply"].lower()

    # Query while idle without wake word
    r_idle = agent.process_query("Show my complaints.")
    assert r_idle["is_idle"] is True
    assert "idle" in r_idle["reply"].lower()

    # Wake word
    r_wake = agent.process_query("Friday, wake up!")
    assert r_wake["is_idle"] is False
    assert "ready" in r_wake["reply"].lower() or "right here" in r_wake["reply"].lower()


# ── 5. Security & RBAC Enforcement ────────────────────────────────────────────

def test_12_rbac_and_action_allowlist():
    """Verifies unauthorized browser actions and page access are strictly rejected."""
    # 1. Arbitrary browser automation is blocked
    is_valid, err = VoicePermissions.validate_action("click", {"x": 100, "y": 200}, "ADMIN")
    assert is_valid is False
    assert "not in the safe allowlist" in err

    is_valid, err = VoicePermissions.validate_action("eval_js", {"code": "alert(1)"}, "SUPERUSER")
    assert is_valid is False

    # 2. Citizens cannot navigate to administrative console
    is_valid, err = VoicePermissions.validate_action(
        "navigate_to_page", {"path": "/admin/dashboard"}, "CITIZEN"
    )
    assert is_valid is False
    assert "not authorized" in err

    # 3. Staff can navigate to administrative console
    is_valid, err = VoicePermissions.validate_action(
        "navigate_to_page", {"path": "/admin/dashboard"}, "ADMIN"
    )
    assert is_valid is True

    # 4. Safe citizen actions are permitted
    is_valid, err = VoicePermissions.validate_action(
        "navigate_to_page", {"path": "/citizen/complaints"}, "CITIZEN"
    )
    assert is_valid is True

    is_valid, err = VoicePermissions.validate_action(
        "open_complaint", {"reference": "SC-2026-000001"}, "CITIZEN"
    )
    assert is_valid is True


# ── 6. FastAPI Voice Endpoints ────────────────────────────────────────────────

def test_13_api_voice_endpoints(client):
    """Verifies all FastAPI voice endpoints work over HTTP."""
    # Health endpoint
    res_health = client.get("/api/voice/health")
    assert res_health.status_code == 200
    data = res_health.json()
    assert data["status"] == "operational"
    assert data["agent_name"] == "FRIDAY"
    assert data["registered_tools_count"] >= 15

    # Tools endpoint
    res_tools = client.get("/api/voice/tools")
    assert res_tools.status_code == 200
    t_data = res_tools.json()
    assert t_data["count"] >= 15

    # Context update & get
    res_ctx_update = client.post("/api/voice/context", json={
        "session_id": "api_test_session",
        "current_page": "/road-intelligence",
        "selected_road": "School Road",
    })
    assert res_ctx_update.status_code == 200
    assert res_ctx_update.json()["context"]["current_page"] == "/road-intelligence"

    res_ctx_get = client.get("/api/voice/context?session_id=api_test_session")
    assert res_ctx_get.status_code == 200
    assert res_ctx_get.json()["context"]["selected_road"] == "School Road"

    # Query endpoint
    res_query = client.post("/api/voice/query", json={
        "session_id": "api_test_session",
        "query": "Friday, what is the current road condition?",
    })
    assert res_query.status_code == 200
    q_data = res_query.json()
    assert q_data["success"] is True
    assert "risk" in q_data["reply"].lower() or "hazard" in q_data["reply"].lower()
    assert len(q_data["tools_called"]) >= 1
