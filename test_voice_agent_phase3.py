# SafeCity Loop V2 — Voice Agent Phase 3 Test Suite
# Tests Safer Route + Voice Navigation integration with RouteEngine,
# conversational opt-in consent, deterministic maneuvers, navigation controls,
# driver safety brevity, and REST endpoints.

import pytest
from fastapi.testclient import TestClient
from backend.main import app
from backend.database import SessionLocal
from backend.voice.friday_agent import FridayAgent
from backend.voice.context import SafeCityContext, context_manager
from backend.voice.navigation import navigation_manager, VoiceNavigationManager
from backend.voice.tools import (
    calculate_routes,
    get_route_details,
    compare_routes,
    select_route,
    start_navigation,
    pause_navigation,
    resume_navigation,
    stop_navigation,
    get_navigation_state,
    get_next_maneuver,
)

client = TestClient(app)


# ── Test 1: RouteEngine Integration & 3-Route Options ─────────────────────────

def test_calculate_routes_three_options():
    """Verify RouteEngine produces Fastest, Balanced, and Safer route alternatives with maneuvers."""
    db = SessionLocal()
    try:
        res = calculate_routes(origin="School Road", destination="Market Junction", db=db)
        assert res["success"] is True
        assert len(res["routes"]) == 3

        keys = [r["route_key"] for r in res["routes"]]
        assert "fastest" in keys
        assert "balanced" in keys
        assert "safer" in keys

        fastest = next(r for r in res["routes"] if r["route_key"] == "fastest")
        safer = next(r for r in res["routes"] if r["route_key"] == "safer")
        balanced = next(r for r in res["routes"] if r["route_key"] == "balanced")

        # Safer route must have lower calculated risk than fastest
        assert safer["overall_calculated_risk"] < fastest["overall_calculated_risk"]
        # Balanced route risk should be between or moderate
        assert balanced["overall_calculated_risk"] <= fastest["overall_calculated_risk"]

        # All 3 routes must have deterministic maneuvers
        for r in [fastest, balanced, safer]:
            assert "maneuvers" in r
            assert len(r["maneuvers"]) > 0
            for m in r["maneuvers"]:
                assert "step_index" in m
                assert "instruction" in m
                assert "distance_meters" in m
                assert "maneuver_type" in m
    finally:
        db.close()


def test_get_route_details():
    """Verify get_route_details retrieves full metrics and maneuvers for a specific route."""
    db = SessionLocal()
    try:
        det = get_route_details(route_id="safer", origin="School Road", destination="Market Junction", db=db)
        assert det["success"] is True
        assert det["route"]["route_key"] == "safer"
        assert len(det["route"]["maneuvers"]) > 0
        assert "voice_summary" in det
    finally:
        db.close()


# ── Test 2: Conversational Safer Route (Phase 3.3 Rule) ───────────────────────

def test_conversational_safer_route_requires_opt_in():
    """
    CRUCIAL RULE: 'find me a safer route' must present route options,
    ask for confirmation, and NEVER automatically start navigation.
    """
    db = SessionLocal()
    agent = FridayAgent(session_id="test_session_opt_in")
    ctx = context_manager.get_context("test_session_opt_in")

    try:
        res = agent.process_query("Friday, find me a safer route", context=ctx, session_id="test_session_opt_in", db=db)
        assert res["success"] is True

        reply = res["reply"]
        # Must present options and ask for user confirmation
        assert "I found three options" in reply or "options" in reply.lower()
        assert "Would you like me to use the safer route?" in reply or "safer route" in reply.lower()

        # Must flag pending opt-in in context
        assert ctx.pending_route_opt_in is True

        # Crucial: Must NOT have called start_navigation or dispatched start_navigation action!
        tool_names = [t["tool"] for t in res["tools_called"]]
        assert "start_navigation" not in tool_names
        action_names = [a["action"] for a in res["frontend_actions"]]
        assert "start_navigation" not in action_names

        # Session in navigation_manager must still be IDLE
        nav_state = navigation_manager.get_navigation_state("test_session_opt_in")
        assert nav_state["status"] == "IDLE"
    finally:
        db.close()


# ── Test 3: Explicit Opt-In Activates Safe Navigation Mode ────────────────────

def test_explicit_opt_in_activates_navigation():
    """
    When user responds 'Yes' following route options, navigation is activated
    and first turn-by-turn maneuver is spoken.
    """
    db = SessionLocal()
    agent = FridayAgent(session_id="test_session_opt_in")
    ctx = context_manager.get_context("test_session_opt_in")

    try:
        # Step 1: User asks for safer route
        agent.process_query("Friday, find me a safer route", context=ctx, session_id="test_session_opt_in", db=db)
        assert ctx.pending_route_opt_in is True

        # Step 2: User says "Yes"
        res = agent.process_query("Yes", context=ctx, session_id="test_session_opt_in", db=db)
        assert res["success"] is True

        # Navigation must now be enabled
        assert "Safer navigation enabled" in res["reply"]

        tool_names = [t["tool"] for t in res["tools_called"]]
        assert "start_navigation" in tool_names

        action_names = [a["action"] for a in res["frontend_actions"]]
        assert "start_navigation" in action_names

        # pending_route_opt_in must be cleared
        assert ctx.pending_route_opt_in is False

        # Live navigation session must be ACTIVE
        nav_state = navigation_manager.get_navigation_state("test_session_opt_in")
        assert nav_state["status"] == "ACTIVE"
        assert nav_state["current_step"] == 0
        assert nav_state["route_key"] == "safer"
    finally:
        # Clean up session
        navigation_manager.stop_navigation("test_session_opt_in")
        db.close()


def test_direct_start_navigation_command():
    """Direct user commands like 'start navigation' or 'use the safer route' activate navigation immediately."""
    db = SessionLocal()
    agent = FridayAgent(session_id="test_session_direct")
    ctx = context_manager.get_context("test_session_direct")

    try:
        res = agent.process_query("Friday, start safer navigation", context=ctx, session_id="test_session_direct", db=db)
        assert res["success"] is True
        assert "Safer navigation enabled" in res["reply"]

        action_names = [a["action"] for a in res["frontend_actions"]]
        assert "start_navigation" in action_names

        nav_state = navigation_manager.get_navigation_state("test_session_direct")
        assert nav_state["status"] == "ACTIVE"
    finally:
        navigation_manager.stop_navigation("test_session_direct")
        db.close()


# ── Test 4: Route Selection (Fastest, Balanced, Safer) ─────────────────────────

def test_route_selection_commands():
    """Verify selecting fastest, balanced, or safer routes updates context and dispatches actions."""
    db = SessionLocal()
    agent = FridayAgent(session_id="test_session_select")
    ctx = context_manager.get_context("test_session_select")

    try:
        # Select fastest
        res_fast = agent.process_query("Show me the fastest route", context=ctx, session_id="test_session_select", db=db)
        assert "fastest" in res_fast["reply"].lower()
        assert ctx.selected_route == "fastest"
        assert any(a["action"] == "select_route" and a["payload"]["route_key"] == "fastest" for a in res_fast["frontend_actions"])

        # Select balanced
        res_bal = agent.process_query("Select balanced route", context=ctx, session_id="test_session_select", db=db)
        assert "balanced" in res_bal["reply"].lower()
        assert ctx.selected_route == "balanced"
        assert any(a["action"] == "select_route" and a["payload"]["route_key"] == "balanced" for a in res_bal["frontend_actions"])

        # Select safer
        res_safe = agent.process_query("Switch to safer route", context=ctx, session_id="test_session_select", db=db)
        assert "safer" in res_safe["reply"].lower()
        assert ctx.selected_route == "safer"
        assert any(a["action"] == "select_route" and a["payload"]["route_key"] == "safer" for a in res_safe["frontend_actions"])
    finally:
        db.close()


# ── Test 5: In-Drive Guidance & Driver Safety Queries ─────────────────────────

def test_turn_by_turn_and_driver_safety_queries():
    """
    Test turn-by-turn maneuver queries:
    - 'What's the next turn?'
    - 'How far until the next turn?'
    - 'How much longer?'
    - 'How much risk does this route avoid?'
    Responses must be concise, clear, and grounded in RouteEngine data.
    """
    db = SessionLocal()
    agent = FridayAgent(session_id="test_session_drive")
    ctx = context_manager.get_context("test_session_drive")

    try:
        # Start navigation
        agent.process_query("Start safer navigation", context=ctx, session_id="test_session_drive", db=db)

        # 1. Next turn
        res_turn = agent.process_query("What's the next turn?", context=ctx, session_id="test_session_drive", db=db)
        assert res_turn["success"] is True
        assert len(res_turn["reply"]) < 180  # Driver safety: concise!
        assert "School Road" in res_turn["reply"] or "Turn" in res_turn["reply"] or "Head" in res_turn["reply"]

        # 2. Distance to next turn
        res_dist = agent.process_query("How far until the next turn?", context=ctx, session_id="test_session_drive", db=db)
        assert res_dist["success"] is True
        assert "metres" in res_dist["reply"].lower() or "km" in res_dist["reply"].lower()

        # 3. Remaining ETA
        res_eta = agent.process_query("How much longer?", context=ctx, session_id="test_session_drive", db=db)
        assert res_eta["success"] is True
        assert "minutes remaining" in res_eta["reply"].lower()

        # 4. Risk avoided
        res_risk = agent.process_query("How much risk does this route avoid?", context=ctx, session_id="test_session_drive", db=db)
        assert res_risk["success"] is True
        assert "reduces calculated risk" in res_risk["reply"].lower()
        assert "%" in res_risk["reply"]
    finally:
        navigation_manager.stop_navigation("test_session_drive")
        db.close()


# ── Test 6: Navigation Session Controls (Pause, Resume, Stop) ─────────────────

def test_navigation_controls():
    """Verify pause, resume, and stop navigation commands and state changes."""
    db = SessionLocal()
    agent = FridayAgent(session_id="test_session_controls")
    ctx = context_manager.get_context("test_session_controls")

    try:
        # Start
        agent.process_query("Start safer navigation", context=ctx, session_id="test_session_controls", db=db)
        sess = navigation_manager.get_session("test_session_controls")
        assert sess.status == "ACTIVE"

        # Pause
        res_pause = agent.process_query("Pause navigation", context=ctx, session_id="test_session_controls", db=db)
        assert "paused" in res_pause["reply"].lower()
        assert sess.status == "PAUSED"
        assert any(a["action"] == "pause_navigation" for a in res_pause["frontend_actions"])

        # Resume
        res_resume = agent.process_query("Resume navigation", context=ctx, session_id="test_session_controls", db=db)
        assert "resuming" in res_resume["reply"].lower()
        assert sess.status == "ACTIVE"
        assert any(a["action"] == "resume_navigation" for a in res_resume["frontend_actions"])

        # Stop / Cancel
        res_stop = agent.process_query("Cancel navigation", context=ctx, session_id="test_session_controls", db=db)
        assert "cancelled" in res_stop["reply"].lower()
        assert sess.status == "IDLE"
        assert any(a["action"] == "stop_navigation" for a in res_stop["frontend_actions"])
    finally:
        navigation_manager.stop_navigation("test_session_controls")
        db.close()


# ── Test 7: Navigation Manager Maneuver Progression ───────────────────────────

def test_navigation_manager_advance_step():
    """Verify turn-by-turn maneuver progression through steps to completion."""
    sid = "test_advance_session"
    sess = navigation_manager.start_navigation(sid, route_key="safer", origin_road_id=1, destination_road_id=2)
    assert sess.status == "ACTIVE"
    assert sess.current_step == 0
    total_steps = len(sess.steps)
    assert total_steps >= 2

    first_step_inst = sess.steps[0].instruction
    m1 = navigation_manager.get_next_maneuver(sid)
    assert m1["instruction"] == first_step_inst

    # Advance to step 1
    m2 = navigation_manager.advance_step(sid)
    assert sess.current_step == 1
    assert m2["instruction"] == sess.steps[1].instruction

    # Advance through remaining steps
    for _ in range(total_steps - 1):
        navigation_manager.advance_step(sid)

    assert sess.status == "COMPLETED"
    navigation_manager.stop_navigation(sid)


# ── Test 8: REST API Navigation Endpoints ─────────────────────────────────────

def test_rest_api_navigation_endpoints():
    """Verify all FastAPI REST navigation endpoints operate correctly."""
    sid = "test_rest_nav"

    # 1. Start navigation
    res_start = client.post("/api/voice/navigation/start", json={
        "session_id": sid,
        "route_key": "safer",
        "origin_road_id": 1,
        "destination_road_id": 2,
    })
    assert res_start.status_code == 200
    data_start = res_start.json()
    assert data_start["success"] is True
    assert data_start["status"] == "ACTIVE"
    assert "first_maneuver" in data_start

    # 2. Get navigation state
    res_state = client.get(f"/api/voice/navigation/state?session_id={sid}")
    assert res_state.status_code == 200
    data_state = res_state.json()
    assert data_state["success"] is True
    assert data_state["navigation_state"]["status"] == "ACTIVE"
    assert data_state["navigation_state"]["route_key"] == "safer"

    # 3. Next maneuver
    res_man = client.get(f"/api/voice/navigation/next-maneuver?session_id={sid}")
    assert res_man.status_code == 200
    assert res_man.json()["success"] is True
    assert "instruction" in res_man.json()

    # 4. Advance step
    res_adv = client.post("/api/voice/navigation/advance", json={"session_id": sid})
    assert res_adv.status_code == 200

    # 5. Pause navigation
    res_pause = client.post("/api/voice/navigation/pause", json={"session_id": sid})
    assert res_pause.status_code == 200
    assert res_pause.json()["status"] == "PAUSED"

    # 6. Resume navigation
    res_resume = client.post("/api/voice/navigation/resume", json={"session_id": sid})
    assert res_resume.status_code == 200
    assert res_resume.json()["status"] == "ACTIVE"

    # 7. Stop navigation
    res_stop = client.post("/api/voice/navigation/stop", json={"session_id": sid})
    assert res_stop.status_code == 200
    assert res_stop.json()["status"] == "IDLE"


# ── Test 9: Tool Enumeration Includes All Phase 3 Tools ───────────────────────

def test_voice_tools_endpoint_lists_phase3_tools():
    """Verify /api/voice/tools registers all 10 Phase 3 navigation tools."""
    res = client.get("/api/voice/tools")
    assert res.status_code == 200
    data = res.json()
    assert data["success"] is True

    tool_names = [t["name"] for t in data["tools"]]
    for expected_tool in [
        "calculate_routes",
        "get_route_details",
        "compare_routes",
        "select_route",
        "start_navigation",
        "pause_navigation",
        "resume_navigation",
        "stop_navigation",
        "get_navigation_state",
        "get_next_maneuver",
    ]:
        assert expected_tool in tool_names, f"Tool '{expected_tool}' should be listed in voice tools"
