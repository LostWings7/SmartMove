# SafeCity Loop V2 — Voice Agent Phase 2 Test Suite
# Tests voice-controlled frontend actions, RPC validation, navigation intents,
# real conflict data retrieval, map focus, and RBAC enforcement.

import pytest
from fastapi.testclient import TestClient
from backend.main import app
from backend.database import get_db, SessionLocal
from backend.voice.permissions import VoicePermissions, PAGE_ALLOWLIST, ALLOWLISTED_ACTIONS
from backend.voice.context import SafeCityContext, context_manager
from backend.voice.friday_agent import FridayAgent
from backend.voice.tools import get_conflict_details, get_complaint_details

client = TestClient(app)


# ── Test 1: PAGE_ALLOWLIST Validation & Resolution ────────────────────────────

def test_page_allowlist_validation():
    """Verify authorized pages resolve correctly and unauthorized/arbitrary pages are rejected."""
    # Valid semantic and direct paths
    assert VoicePermissions.validate_page("dashboard") == "/"
    assert VoicePermissions.validate_page("my-complaints") == "/citizen/complaints"
    assert VoicePermissions.validate_page("/traffic-conflicts") == "/traffic-conflicts"
    assert VoicePermissions.validate_page("ai-vision") == "/ai-vision"
    assert VoicePermissions.validate_page("road-intelligence") == "/road-intelligence"
    assert VoicePermissions.validate_page("map") == "/map"
    assert VoicePermissions.validate_page("repairs") == "/repair"
    assert VoicePermissions.validate_page("routes") == "/routes"
    assert VoicePermissions.validate_page("danger-zones") == "/danger-zones"

    # Reject arbitrary/malicious paths
    assert VoicePermissions.validate_page("https://google.com") is None
    assert VoicePermissions.validate_page("eval(code)") is None
    assert VoicePermissions.validate_page("<script>") is None
    assert VoicePermissions.validate_page("/bin/sh") is None
    assert VoicePermissions.validate_page("unknown_fake_page") is None


# ── Test 2: Role-Based Page Access Control (RBAC) ─────────────────────────────

def test_role_based_page_access():
    """Verify citizens are blocked from admin routes while staff/admins are permitted."""
    admin_routes = ["/admin/dashboard", "/admin/complaints", "/admin/users"]

    for route in admin_routes:
        # Citizens must NOT be allowed
        assert not VoicePermissions.can_access_page("CITIZEN", route), f"Citizen should not access {route}"

        # Staff roles MUST be allowed
        for role in ["INSPECTOR", "ENGINEER", "ADMIN", "SUPERUSER"]:
            assert VoicePermissions.can_access_page(role, route), f"{role} should be allowed to access {route}"

    # Public/Citizen routes accessible by all
    assert VoicePermissions.can_access_page("CITIZEN", "/citizen/complaints")
    assert VoicePermissions.can_access_page("CITIZEN", "/map")
    assert VoicePermissions.can_access_page("CITIZEN", "/traffic-conflicts")


# ── Test 3: Frontend Action Allowlist & Rejection ─────────────────────────────

def test_action_allowlist_enforcement():
    """Verify allowlisted actions pass and arbitrary browser commands are rejected."""
    # Allowlisted actions
    for action in [
        "navigate_to_page", "open_complaint", "open_hazard", "open_conflict",
        "open_evidence", "focus_map", "highlight_hazard", "highlight_complaint",
        "seek_video", "open_route", "show_route_comparison", "show_danger_zones",
        "show_status_history"
    ]:
        assert VoicePermissions.is_allowlisted_action(action), f"Action '{action}' should be allowlisted"

    # Prohibited arbitrary browser automation
    for bad_action in [
        "click", "click_element", "type", "type_text", "eval", "eval_js",
        "execute_script", "press_key", "hover", "drag_and_drop", "screenshot"
    ]:
        assert not VoicePermissions.is_allowlisted_action(bad_action), f"Action '{bad_action}' must be rejected"


# ── Test 4: Action Argument Validation ────────────────────────────────────────

def test_action_argument_validation():
    """Verify strict schema validation for action arguments."""
    # focus_map: valid coordinates
    ok, err = VoicePermissions.validate_action_arguments("focus_map", {"lat": 51.5074, "lng": -0.1278, "zoom": 16})
    assert ok, f"Valid focus_map failed: {err}"

    # focus_map: invalid coordinates (lat > 90)
    ok, err = VoicePermissions.validate_action_arguments("focus_map", {"lat": 150.0, "lng": 0.0})
    assert not ok and "Latitude" in err

    # seek_video: valid non-negative seconds
    ok, err = VoicePermissions.validate_action_arguments("seek_video", {"timestamp_seconds": 14.5})
    assert ok, f"Valid seek_video failed: {err}"

    # seek_video: invalid negative seconds
    ok, err = VoicePermissions.validate_action_arguments("seek_video", {"timestamp_seconds": -5.0})
    assert not ok and "negative" in err

    # open_hazard: valid positive ID
    ok, err = VoicePermissions.validate_action_arguments("open_hazard", {"id": 3})
    assert ok

    # open_hazard: invalid ID
    ok, err = VoicePermissions.validate_action_arguments("open_hazard", {"id": -1})
    assert not ok


# ── Test 5: Complaint Navigation Workflow ─────────────────────────────────────

def test_complaint_navigation_workflow():
    """Test full cycle: 'Open my complaints' -> 'Open complaint SC-...' -> 'Status history' -> 'Evidence'."""
    ctx = SafeCityContext(
        current_user={"id": 1, "full_name": "Citizen User", "role": "CITIZEN"},
        user_role="CITIZEN"
    )
    agent = FridayAgent(session_id="test_complaint_flow")
    db = SessionLocal()

    try:
        # Step 1: Open my complaints
        r1 = agent.process_query("Open my complaints", context=ctx, db=db)
        assert r1["success"]
        actions1 = r1["frontend_actions"]
        assert any(a["action"] == "navigate_to_page" and a["payload"]["path"] == "/citizen/complaints" for a in actions1)

        # Step 2: Open specific complaint
        r2 = agent.process_query("Open complaint SC-2026-000184", context=ctx, db=db)
        assert r2["success"]
        actions2 = r2["frontend_actions"]
        assert any(a["action"] == "open_complaint" for a in actions2)
        assert "SC-2026-000184" in r2["reply"]

        # Step 3: Show status history
        r3 = agent.process_query("Show me the status history", context=ctx, db=db)
        assert r3["success"]
        actions3 = r3["frontend_actions"]
        assert any(a["action"] == "show_status_history" for a in actions3)

        # Step 4: Open complaint evidence
        r4 = agent.process_query("Open the evidence for that complaint", context=ctx, db=db)
        assert r4["success"]
        actions4 = r4["frontend_actions"]
        assert any(a["action"] == "open_evidence" for a in actions4)

    finally:
        db.close()


# ── Test 6: Traffic Navigation & Video Seeking ────────────────────────────────

def test_traffic_navigation_and_video_seek():
    """Test full cycle: 'Open traffic conflicts' -> 'Show me the critical conflict' -> 'Take me to that moment in the video'."""
    agent = FridayAgent(session_id="test_traffic_flow")
    db = SessionLocal()

    try:
        # Step 1: Open traffic conflicts
        r1 = agent.process_query("Open traffic conflicts", db=db)
        assert r1["success"]
        assert any(a["action"] == "navigate_to_page" and a["payload"]["path"] == "/traffic-conflicts" for a in r1["frontend_actions"])

        # Step 2: Show critical conflict
        r2 = agent.process_query("Show me the critical conflict", db=db)
        assert r2["success"]
        assert any(a["action"] == "open_conflict" for a in r2["frontend_actions"])
        assert "TTC" in r2["reply"] or "traffic conflict" in r2["reply"].lower()

        # Step 3: Show its evidence
        r3 = agent.process_query("Show me its evidence", db=db)
        assert r3["success"]
        assert any(a["action"] == "open_evidence" for a in r3["frontend_actions"])

        # Step 4: Seek video to exact occurrence
        r4 = agent.process_query("Take me to that moment in the video", db=db)
        assert r4["success"]
        seek_actions = [a for a in r4["frontend_actions"] if a["action"] == "seek_video"]
        assert len(seek_actions) == 1
        payload = seek_actions[0]["payload"]
        assert "timestamp_seconds" in payload
        assert isinstance(payload["timestamp_seconds"], (int, float))
        assert payload["timestamp_seconds"] >= 0
        assert "clip_url" in payload
        assert "Seeking video playback" in r4["reply"]

    finally:
        db.close()


# ── Test 7: AI Vision Navigation Intents ──────────────────────────────────────

def test_ai_vision_navigation_intents():
    """Verify switching to pothole detections and waterlogging analysis."""
    agent = FridayAgent(session_id="test_vision_flow")
    db = SessionLocal()

    try:
        # Potholes view
        rp = agent.process_query("Show me the pothole detections", db=db)
        assert rp["success"]
        assert any(
            a["action"] == "navigate_to_page" and a["payload"]["path"] == "/ai-vision" and a["payload"].get("tab") == "potholes"
            for a in rp["frontend_actions"]
        )

        # Waterlogging view
        rw = agent.process_query("Open the waterlogging analysis", db=db)
        assert rw["success"]
        assert any(
            a["action"] == "navigate_to_page" and a["payload"].get("tab") == "waterlogging"
            for a in rw["frontend_actions"]
        )

        # Hazard evidence
        rh = agent.process_query("Show me the evidence for this hazard", db=db)
        assert rh["success"]
        assert any(a["action"] == "open_evidence" for a in rh["frontend_actions"])

    finally:
        db.close()


# ── Test 8: Map Control & Telemetry Actions ───────────────────────────────────

def test_map_control_actions():
    """Verify GIS map focus, entity highlighting, and danger zone display."""
    agent = FridayAgent(session_id="test_map_flow")
    db = SessionLocal()

    try:
        # Focus on School Road
        rf = agent.process_query("Focus the map on School Road", db=db)
        assert rf["success"]
        focus_actions = [a for a in rf["frontend_actions"] if a["action"] == "focus_map"]
        assert len(focus_actions) == 1
        p = focus_actions[0]["payload"]
        assert p["lat"] == pytest.approx(51.5074, 0.01)
        assert p["lng"] == pytest.approx(-0.1278, 0.01)
        assert p["zoom"] == 16

        # Highlight hazard #1
        rh = agent.process_query("Highlight hazard #1", db=db)
        assert rh["success"]
        assert any(a["action"] == "highlight_hazard" and a["payload"]["id"] == 1 for a in rh["frontend_actions"])

        # Highlight complaint
        rc = agent.process_query("Highlight complaint SC-2026-000184", db=db)
        assert rc["success"]
        assert any(a["action"] == "highlight_complaint" and a["payload"]["reference"] == "SC-2026-000184" for a in rc["frontend_actions"])

        # Show danger zones
        rd = agent.process_query("Show danger zones", db=db)
        assert rd["success"]
        assert any(a["action"] == "show_danger_zones" for a in rd["frontend_actions"])

        # Show route comparison
        rr = agent.process_query("Show route comparison", db=db)
        assert rr["success"]
        assert any(a["action"] == "show_route_comparison" for a in rr["frontend_actions"])

    finally:
        db.close()


# ── Test 9: Voice Query API Router Integration ────────────────────────────────

def test_voice_query_api_endpoint():
    """Verify /api/voice/query produces valid schema and allowlisted frontend actions."""
    res = client.post("/api/voice/query", json={
        "session_id": "test_api_query_session",
        "query": "Open traffic conflicts",
    })
    assert res.status_code == 200
    data = res.json()
    assert data["success"]
    assert "reply" in data
    assert isinstance(data["frontend_actions"], list)
    assert any(a["action"] == "navigate_to_page" for a in data["frontend_actions"])
    assert isinstance(data["tools_called"], list)


def test_voice_query_api_video_seek():
    """Verify /api/voice/query video seeking returns exact timestamp and clip."""
    # First select a conflict
    client.post("/api/voice/query", json={
        "session_id": "test_video_seek_session",
        "query": "Show me the critical conflict",
    })

    # Then seek video
    res = client.post("/api/voice/query", json={
        "session_id": "test_video_seek_session",
        "query": "Take me to that moment in the video",
    })
    assert res.status_code == 200
    data = res.json()
    assert data["success"]
    assert any(a["action"] == "seek_video" for a in data["frontend_actions"])
    seek_action = next(a for a in data["frontend_actions"] if a["action"] == "seek_video")
    assert seek_action["payload"]["timestamp_seconds"] >= 0
    assert "clip_url" in seek_action["payload"]
