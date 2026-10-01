"""
SafeCity Loop V2 — Dedicated FRIDAY Voice Recitation Test Suite
Verifies verbatim recitation of turn-by-turn navigation clues, zero hallucination,
allowlisted actions, agent natural language intents, permissions, and REST endpoints.
"""
import pytest
from fastapi.testclient import TestClient
from backend.main import app
from backend.database import SessionLocal
from backend.voice.navigation import VoiceNavigationManager, navigation_manager
from backend.voice.tools import get_tool_registry, read_navigation_directions
from backend.voice.permissions import VoicePermissions, ALLOWLISTED_ACTIONS, TOOL_METADATA
from backend.voice.friday_agent import get_friday_agent
from backend.voice.context import context_manager


@pytest.fixture(scope="module")
def db_session():
    db = SessionLocal()
    yield db
    db.close()


@pytest.fixture(scope="module")
def client():
    return TestClient(app)


# ── 1. Tool 33 Execution & Verbatim Clue Verification ─────────────────────────

def test_tool_33_idle_behavior(db_session):
    """Tool 33 must fail gracefully when navigation is not active."""
    mgr = VoiceNavigationManager()
    sid = "test_recite_idle"

    res = mgr.get_navigation_directions(session_id=sid)
    assert res["success"] is False
    assert res["status"] == "IDLE"
    assert "start navigation first" in res["error"].lower()
    assert "no active navigation route" in res["voice_summary"].lower()


def test_tool_33_active_directions_match_route_engine(db_session):
    """Tool 33 must return exact displayed navigation steps matching RouteEngine verbatim."""
    mgr = VoiceNavigationManager()
    sid = "test_recite_active"

    sess = mgr.start_navigation(session_id=sid, route_key="safer", db=db_session)
    assert sess.status == "ACTIVE"
    assert len(sess.steps) >= 3

    res = mgr.get_navigation_directions(session_id=sid)
    assert res["success"] is True
    assert res["status"] == "ACTIVE"
    assert res["total"] == len(sess.steps)
    assert len(res["directions"]) == len(sess.steps)

    # Verify each direction matches step instruction verbatim (Zero Hallucination)
    for i, d in enumerate(res["directions"]):
        assert d["index"] == sess.steps[i].step_index
        assert d["text"] == sess.steps[i].instruction
        assert d["maneuver_type"] == sess.steps[i].maneuver_type
        assert d["distance_meters"] == sess.steps[i].distance_meters
        assert d["hazard_warning"] == sess.steps[i].hazard_warning


def test_tool_33_custom_start_index(db_session):
    """Tool 33 should support starting recitation from an arbitrary valid index."""
    mgr = VoiceNavigationManager()
    sid = "test_recite_index"

    sess = mgr.start_navigation(session_id=sid, route_key="fastest", db=db_session)
    assert len(sess.steps) >= 2

    res = mgr.get_navigation_directions(session_id=sid, start_index=1)
    assert res["success"] is True
    assert res["start_index"] == 1
    assert "step 2" in res["voice_summary"].lower()


# ── 2. Allowlisted Actions & Permission Engine ────────────────────────────────

def test_permissions_recitation_actions_registered():
    """Verify all 3 recitation actions are strictly allowlisted in permissions."""
    assert "read_navigation_directions" in ALLOWLISTED_ACTIONS
    assert "read_next_navigation_direction" in ALLOWLISTED_ACTIONS
    assert "stop_navigation_speech" in ALLOWLISTED_ACTIONS


def test_permissions_tool_33_metadata():
    """Verify Tool 33 metadata is registered with CITIZEN role requirement."""
    meta = TOOL_METADATA.get("read_navigation_directions")
    assert meta is not None
    assert meta["required_role"] == "CITIZEN"
    assert meta["is_write"] is False
    assert meta["requires_confirmation"] is False


def test_permissions_validate_action():
    """Validate action allowlist checks for recitation actions."""
    # Valid actions for Citizen
    ok1, err1 = VoicePermissions.validate_action("read_navigation_directions", {}, "CITIZEN")
    assert ok1 is True
    assert err1 is None

    ok2, err2 = VoicePermissions.validate_action("read_next_navigation_direction", {}, "CITIZEN")
    assert ok2 is True
    assert err2 is None

    ok3, err3 = VoicePermissions.validate_action("stop_navigation_speech", {}, "CITIZEN")
    assert ok3 is True
    assert err3 is None

    # Invalid negative start_index
    bad_idx, err_idx = VoicePermissions.validate_action("read_navigation_directions", {"start_index": -1}, "CITIZEN")
    assert bad_idx is False
    assert "greater than or equal to 0" in err_idx

    # Prohibited arbitrary browser automation action
    bad_act, err_act = VoicePermissions.validate_action("window.eval", {}, "ADMIN")
    assert bad_act is False
    assert "not in the safe allowlist" in err_act


# ── 3. FRIDAY Agent Natural Language Intents ──────────────────────────────────

def test_agent_read_directions_intent(db_session):
    """Verify 'read navigation directions' intent triggers Tool 33 and front-end action."""
    sid = "test_agent_recite_all"
    navigation_manager.start_navigation(session_id=sid, route_key="safer", db=db_session)
    ctx = context_manager.get_context(sid)

    agent = get_friday_agent()
    result = agent.process_query("Friday, read the navigation directions", context=ctx, db=db_session)

    assert result["success"] is True
    assert any(tc["tool"] == "read_navigation_directions" for tc in result["tools_called"])
    assert any(fa["action"] == "read_navigation_directions" for fa in result["frontend_actions"])
    assert "directions" in result["reply"].lower() or "reading" in result["reply"].lower()


def test_agent_read_next_turn_intent(db_session):
    """Verify 'read next turn' intent triggers get_next_maneuver and front-end action."""
    sid = "test_agent_recite_next"
    sess = navigation_manager.start_navigation(session_id=sid, route_key="safer", db=db_session)
    ctx = context_manager.get_context(sid)

    agent = get_friday_agent()
    result = agent.process_query("Friday, read next turn", context=ctx, db=db_session)

    assert result["success"] is True
    assert any(tc["tool"] == "get_next_maneuver" for tc in result["tools_called"])
    assert any(fa["action"] == "read_next_navigation_direction" for fa in result["frontend_actions"])
    # The spoken reply must be verbatim current instruction
    assert result["reply"] == sess.steps[0].instruction


def test_agent_stop_reading_intent(db_session):
    """Verify 'stop reading' intent dispatches stop_navigation_speech and stops voice output."""
    sid = "test_agent_recite_stop"
    ctx = context_manager.get_context(sid)

    agent = get_friday_agent()
    result = agent.process_query("Friday, stop reading directions", context=ctx, db=db_session)

    assert result["success"] is True
    assert any(fa["action"] == "stop_navigation_speech" for fa in result["frontend_actions"])
    assert "stopped reading directions" in result["reply"].lower()


# ── 4. REST Endpoints Verification ───────────────────────────────────────────

def test_rest_tools_enumeration_count(client):
    """Verify /api/voice/tools reports 33 registered tools including read_navigation_directions."""
    res = client.get("/api/voice/tools")
    assert res.status_code == 200
    data = res.json()
    assert data["success"] is True
    assert data["count"] == 33
    tool_names = [t["name"] for t in data["tools"]]
    assert "read_navigation_directions" in tool_names


def test_rest_navigation_directions_endpoint(client, db_session):
    """Verify GET /api/voice/navigation/directions returns active navigation maneuvers."""
    sid = "test_rest_nav_directions"
    navigation_manager.start_navigation(session_id=sid, route_key="safer", db=db_session)

    res = client.get(f"/api/voice/navigation/directions?session_id={sid}")
    assert res.status_code == 200
    data = res.json()
    assert data["success"] is True
    assert data["status"] == "ACTIVE"
    assert data["total"] > 0
    assert len(data["directions"]) > 0
    assert "text" in data["directions"][0]
