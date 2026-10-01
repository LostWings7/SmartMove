"""
SafeCity Loop V2 — Dedicated FRIDAY Navigation Test Suite (Phase 4.37)
Verifies VoiceNavigationManager state transitions, turn-by-turn guidance, driver failsafes, and zero hallucination.
"""
import pytest
from backend.database import SessionLocal
from backend.voice.navigation import VoiceNavigationManager, navigation_manager


@pytest.fixture(scope="module")
def db_session():
    db = SessionLocal()
    yield db
    db.close()


def test_navigation_lifecycle_state_transitions(db_session):
    """Verify complete state machine: IDLE -> ACTIVE -> PAUSED -> ACTIVE -> IDLE."""
    mgr = VoiceNavigationManager()
    sid = "test_nav_lifecycle"

    # Initial state must be IDLE
    initial = mgr.get_navigation_state(sid)
    assert initial["status"] == "IDLE"

    # Start navigation
    sess = mgr.start_navigation(session_id=sid, route_key="safer", db=db_session)
    assert sess.status == "ACTIVE"
    assert len(sess.steps) > 0
    assert "safer" in sess.route_key

    # Pause navigation
    paused = mgr.pause_navigation(session_id=sid)
    assert paused["success"] is True
    assert paused["status"] == "PAUSED"
    assert "paused" in paused["voice_summary"].lower()

    # Resume navigation
    resumed = mgr.resume_navigation(session_id=sid)
    assert resumed["success"] is True
    assert resumed["status"] == "ACTIVE"
    assert "resuming" in resumed["voice_summary"].lower()

    # Stop navigation
    stopped = mgr.stop_navigation(session_id=sid)
    assert stopped["success"] is True
    assert stopped["status"] == "IDLE"
    assert "cancelled" in stopped["voice_summary"].lower() or "stopped" in stopped["voice_summary"].lower() or "ended" in stopped["voice_summary"].lower()


def test_maneuver_guidance_and_step_advancement(db_session):
    """Verify turn-by-turn instruction retrieval and step advancing."""
    mgr = VoiceNavigationManager()
    sid = "test_maneuver_advancement"

    mgr.start_navigation(session_id=sid, route_key="safer", db=db_session)

    # Step 0
    step0 = mgr.get_next_maneuver(sid)
    assert step0["success"] is True
    assert step0["step_index"] == 0
    assert len(step0["instruction"]) > 0
    assert step0["distance_meters"] > 0
    assert "voice_summary" in step0

    # Advance to next steps until completed
    total_steps = step0["total_steps"]
    for i in range(1, total_steps):
        next_step = mgr.advance_step(sid)
        assert next_step["step_index"] == i
        assert len(next_step["instruction"]) > 0

    # Final advance reaches completion
    final_step = mgr.advance_step(sid)
    assert final_step["status"] == "COMPLETED"
    assert "arrived" in final_step["instruction"].lower() or "reached" in final_step["voice_summary"].lower()


def test_driver_safety_failsafes(db_session):
    """Verify failsafes when navigation is inactive or paused (zero hallucinated turns)."""
    mgr = VoiceNavigationManager()
    sid = "test_failsafes"

    # Query next turn while IDLE
    res_idle = mgr.get_next_maneuver(sid)
    assert res_idle["success"] is False
    assert "no active navigation" in res_idle["voice_summary"].lower()

    # Query next turn while PAUSED
    mgr.start_navigation(session_id=sid, route_key="fastest", db=db_session)
    mgr.pause_navigation(sid)
    res_paused = mgr.get_next_maneuver(sid)
    assert res_paused["status"] == "PAUSED"
    assert "paused" in res_paused["voice_summary"].lower()
