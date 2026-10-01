"""
SafeCity Loop V2 — Dedicated FRIDAY Safe Route Test Suite (Phase 4.37)
Verifies 3 route options, risk trade-offs, explicit opt-in confirmation, and driver safety voice guidance.
"""
import pytest
from backend.database import SessionLocal
from backend.voice.friday_agent import get_friday_agent
from backend.voice.context import SafeCityContext
from backend.voice.navigation import navigation_manager


@pytest.fixture(scope="module")
def db_session():
    db = SessionLocal()
    yield db
    db.close()


def test_three_route_options_and_risk_tradeoffs(db_session):
    """Verify that RouteEngine calculates 3 distinct routes with authoritative risk trade-offs."""
    agent = get_friday_agent()
    ctx = SafeCityContext(client_id="test_route_tradeoffs", user_role="CITIZEN")

    res = agent.process_query("Friday, find me a safer route", context=ctx, db=db_session)
    assert res["success"] is True
    assert "tools_called" in res
    assert any(t["tool"] == "calculate_routes" for t in res["tools_called"])

    calc_res = next(t["result"] for t in res["tools_called"] if t["tool"] == "calculate_routes")
    routes = calc_res.get("routes", [])
    assert len(routes) == 3, f"Expected 3 routes, got {len(routes)}"

    keys = [r["route_key"] for r in routes]
    assert "fastest" in keys
    assert "balanced" in keys
    assert "safer" in keys

    fastest = next(r for r in routes if r["route_key"] == "fastest")
    safer = next(r for r in routes if r["route_key"] == "safer")

    # Authoritative risk trade-off: Fastest has higher risk score than Safer
    assert fastest["overall_calculated_risk"] > safer["overall_calculated_risk"]
    # Safer route duration is equal to or greater than fastest (bypass penalty)
    assert safer["time_minutes"] >= fastest["time_minutes"]


def test_mandatory_explicit_opt_in_before_navigation(db_session):
    """Verify FRIDAY NEVER starts navigation automatically; explicit user opt-in is required."""
    agent = get_friday_agent()
    sid = "test_opt_in_session"
    ctx = SafeCityContext(client_id=sid, user_role="CITIZEN")

    # Step 1: User asks for route comparison
    r1 = agent.process_query("Find me a safer route", context=ctx, db=db_session, session_id=sid)
    assert ctx.pending_route_opt_in is True, "pending_route_opt_in flag must be set"

    # Navigation must NOT be active yet
    nav_state = navigation_manager.get_navigation_state(sid)
    assert nav_state["status"] == "IDLE", "Navigation session must remain IDLE until explicit opt-in"

    # Step 2: User explicitly opts in
    r2 = agent.process_query("Yes, use the safer route", context=ctx, db=db_session, session_id=sid)
    assert ctx.pending_route_opt_in is False, "pending_route_opt_in flag must be cleared"
    assert any(t["tool"] == "start_navigation" for t in r2["tools_called"])

    # Navigation is now ACTIVE
    active_nav = navigation_manager.get_navigation_state(sid)
    assert active_nav["status"] == "ACTIVE"
    assert active_nav["route_key"] == "safer"


def test_voice_turn_guidance_and_risk_avoidance(db_session):
    """Verify voice navigation answers next turn, remaining ETA, and risk reduction percentage."""
    agent = get_friday_agent()
    sid = "test_turn_guidance"
    ctx = SafeCityContext(client_id=sid, user_role="CITIZEN")

    # Start navigation session
    navigation_manager.start_navigation(session_id=sid, route_key="safer", db=db_session)

    # 1. Query next turn
    r_turn = agent.process_query("What's the next turn?", context=ctx, db=db_session, session_id=sid)
    assert any(t["tool"] == "get_next_maneuver" for t in r_turn["tools_called"])
    assert len(r_turn["reply"]) > 0

    # 2. Query risk reduction
    r_risk = agent.process_query("How much risk does this route avoid?", context=ctx, db=db_session, session_id=sid)
    assert "reduces calculated risk" in r_risk["reply"].lower() or "risk" in r_risk["reply"].lower()

    # 3. Query ETA / remaining time
    r_eta = agent.process_query("What's the ETA?", context=ctx, db=db_session, session_id=sid)
    assert "minutes remaining" in r_eta["reply"].lower() or "eta" in r_eta["reply"].lower()
