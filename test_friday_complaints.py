"""
SafeCity Loop V2 — Dedicated FRIDAY Voice Complaints Test Suite (Phase 4.37)
Verifies conversational gathering, mandatory explicit confirmation, status tracking, reopening, and audit logging.
"""
import pytest
from backend.database import SessionLocal
from backend.models import Complaint, ComplaintStatusHistory, AuditLog, Notification
from backend.voice.friday_agent import get_friday_agent
from backend.voice.context import SafeCityContext


@pytest.fixture(scope="module")
def db_session():
    db = SessionLocal()
    yield db
    db.close()


def test_complaint_conversational_filing_and_confirmation(db_session):
    """Verify that complaint creation NEVER auto-submits without explicit user confirmation."""
    agent = get_friday_agent()
    ctx = SafeCityContext(
        client_id="test_complaint_filing",
        user_role="CITIZEN",
        is_authenticated=True,
        current_user={"id": 1, "full_name": "Citizen User", "role": "CITIZEN"},
    )

    # 1. Step 1: User provides intent with location
    q1 = "Friday, report a pothole near the college junction"
    r1 = agent.process_query(q1, context=ctx, db=db_session)

    # Must prompt for confirmation and NOT execute tool yet
    assert ctx.pending_action is not None, "Pending action must be stored"
    assert ctx.pending_action["action"] == "file_complaint"
    assert "college junction" in ctx.pending_action["args"]["location_address"].lower()
    assert "ready" in r1["reply"].lower() and "submit it" in r1["reply"].lower()
    assert len(r1["tools_called"]) == 0, "No tools should be executed prior to confirmation"

    # 2. Step 2: User explicitly confirms
    q2 = "Yes, submit it"
    r2 = agent.process_query(q2, context=ctx, db=db_session)

    assert ctx.pending_action is None, "Pending action must be cleared upon execution"
    assert len(r2["tools_called"]) > 0
    assert r2["tools_called"][0]["tool"] == "file_complaint"
    tool_res = r2["tools_called"][0]["result"]
    assert tool_res["success"] is True
    created_ref = tool_res["reference"]
    assert created_ref.startswith("SC-")

    # 3. Step 3: Verify database state
    comp = db_session.query(Complaint).filter(Complaint.public_reference == created_ref).first()
    assert comp is not None
    assert comp.citizen_id == 1
    assert "college junction" in comp.location_address.lower()

    # 4. Verify Status History
    hist = db_session.query(ComplaintStatusHistory).filter(ComplaintStatusHistory.complaint_id == comp.id).first()
    assert hist is not None
    assert "SUBMITTED" in str(hist.new_status)

    # 5. Verify Audit Log
    audit = db_session.query(AuditLog).filter(
        AuditLog.target_type == "Complaint",
        AuditLog.target_id == comp.id,
        AuditLog.action == "VOICE_COMPLAINT_CREATED",
    ).first()
    assert audit is not None
    assert "FRIDAY Voice" in audit.actor

    # 6. Verify Citizen Notification
    notif = db_session.query(Notification).filter(Notification.related_complaint_id == comp.id).first()
    assert notif is not None
    assert created_ref in notif.title


def test_complaint_filing_cancellation(db_session):
    """Verify that saying 'No, cancel' aborts the complaint without writing to the database."""
    agent = get_friday_agent()
    ctx = SafeCityContext(
        client_id="test_complaint_cancellation",
        user_role="CITIZEN",
        is_authenticated=True,
        current_user={"id": 1, "full_name": "Citizen User", "role": "CITIZEN"},
    )

    # Start filing
    agent.process_query("Report waterlogging on School Road", context=ctx, db=db_session)
    assert ctx.pending_action is not None

    # Cancel
    r_cancel = agent.process_query("No, cancel it", context=ctx, db=db_session)
    assert ctx.pending_action is None
    assert "cancelled" in r_cancel["reply"].lower()
    assert any(a["action"] == "cancel_action" for a in r_cancel["frontend_actions"])


def test_complaint_status_tracking(db_session):
    """Verify citizen can check status of their complaints using natural language."""
    agent = get_friday_agent()
    ctx = SafeCityContext(
        client_id="test_complaint_status",
        user_role="CITIZEN",
        is_authenticated=True,
        current_user={"id": 1, "full_name": "Citizen User", "role": "CITIZEN"},
        selected_complaint="SC-2026-000184",
    )

    res = agent.process_query("What is the status of my complaint?", context=ctx, db=db_session)
    assert any(t["tool"] == "get_complaint_status" for t in res["tools_called"])
    assert "SC-2026-000184" in res["reply"]
    assert any(s in res["reply"].lower() for s in ["reopened", "submitted", "in_progress", "status", "assigned"])


def test_complaint_reopening_with_confirmation(db_session):
    """Verify complaint reopening requires explicit confirmation and logs to AuditLog."""
    agent = get_friday_agent()
    # Use existing complaint SC-2026-000184
    comp = db_session.query(Complaint).filter(Complaint.public_reference == "SC-2026-000184").first()
    assert comp is not None
    comp.status = "RESOLVED"
    db_session.commit()

    ctx = SafeCityContext(
        client_id="test_reopen_flow",
        user_role="CITIZEN",
        is_authenticated=True,
        current_user={"id": 1, "full_name": "Citizen User", "role": "CITIZEN"},
        selected_complaint="SC-2026-000184",
    )

    # Step 1: Reopen request prompts confirmation
    r1 = agent.process_query("Reopen this complaint", context=ctx, db=db_session)
    assert ctx.pending_action is not None
    assert ctx.pending_action["action"] == "reopen_complaint"
    assert "reopen complaint sc-2026-000184" in r1["reply"].lower()

    # Step 2: Confirm reopening
    r2 = agent.process_query("Yes, confirm", context=ctx, db=db_session)
    assert ctx.pending_action is None
    assert any(t["tool"] == "reopen_complaint" for t in r2["tools_called"])

    # Step 3: Verify DB status and AuditLog
    db_session.refresh(comp)
    assert comp.status == "REOPENED"

    audit = db_session.query(AuditLog).filter(
        AuditLog.target_type == "Complaint",
        AuditLog.target_id == comp.id,
        AuditLog.action == "VOICE_COMPLAINT_REOPENED",
    ).first()
    assert audit is not None
