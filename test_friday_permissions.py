"""
SafeCity Loop V2 — Dedicated FRIDAY Permissions & RBAC Test Suite (Phase 4.37)
Verifies citizen data isolation, staff privileges, action allowlisting, and security boundaries.
"""
import pytest
from backend.database import SessionLocal
from backend.models import User, Complaint
from backend.voice.context import SafeCityContext
from backend.voice.permissions import VoicePermissions, PAGE_ALLOWLIST, ALLOWLISTED_ACTIONS as ACTION_ALLOWLIST
from backend.voice.tools import get_tool_registry


@pytest.fixture(scope="module")
def db_session():
    db = SessionLocal()
    yield db
    db.close()


def test_citizen_tool_execution_restrictions():
    """Verify citizens are strictly blocked from executing staff/admin tools."""
    # 1. Unresolved complaints queue is staff-only
    can_exec, err = VoicePermissions.can_execute_tool("get_unresolved_complaints", "CITIZEN")
    assert can_exec is False
    assert "restricted" in err.lower() or "denied" in err.lower()

    # 2. Assign complaint is staff-only
    can_assign, a_err = VoicePermissions.can_execute_tool("assign_complaint_to_department", "CITIZEN")
    assert can_assign is False

    # 3. Status transition is staff-only
    can_update, u_err = VoicePermissions.can_execute_tool("update_complaint_status_admin", "CITIZEN")
    assert can_update is False

    # 4. Safe read tools are allowed for citizens
    can_calc, _ = VoicePermissions.can_execute_tool("calculate_routes", "CITIZEN")
    assert can_calc is True
    can_haz, _ = VoicePermissions.can_execute_tool("get_active_hazards", "CITIZEN")
    assert can_haz is True
    can_file, _ = VoicePermissions.can_execute_tool("file_complaint", "CITIZEN")
    assert can_file is True


def test_staff_tool_execution_allowance():
    """Verify staff roles (INSPECTOR, ENGINEER, ADMIN, SUPERUSER) can execute staff tools."""
    for role in ["INSPECTOR", "ENGINEER", "ADMIN", "SUPERUSER"]:
        can_exec, _ = VoicePermissions.can_execute_tool("get_unresolved_complaints", role)
        assert can_exec is True, f"Role {role} should be able to query unresolved complaints"

        can_assign, _ = VoicePermissions.can_execute_tool("assign_complaint_to_department", role)
        assert can_assign is True, f"Role {role} should be able to assign complaints"

        can_update, _ = VoicePermissions.can_execute_tool("update_complaint_status_admin", role)
        assert can_update is True, f"Role {role} should be able to update status"


def test_citizen_complaint_isolation(db_session):
    """Verify a citizen cannot retrieve another citizen's private complaint details."""
    registry = get_tool_registry()

    # Create context for Citizen A (id=999)
    ctx_citizen_a = SafeCityContext(
        client_id="sess_cit_a",
        user_role="CITIZEN",
        is_authenticated=True,
        current_user={"id": 999, "full_name": "Citizen A", "role": "CITIZEN"},
    )

    # In database, SC-2026-000184 belongs to citizen_id=1
    res = registry["get_complaint_details"](
        complaint_reference="SC-2026-000184",
        context=ctx_citizen_a,
        db=db_session,
    )
    # Citizen A does not own SC-2026-000184
    assert res["success"] is False
    assert "access denied" in res["voice_summary"].lower() or "not authorized" in res["voice_summary"].lower()


def test_frontend_action_allowlist_validation():
    """Verify action allowlisting rejects unauthorized or arbitrary frontend actions."""
    valid_actions = [
        "navigate_to_page",
        "open_complaint",
        "open_hazard",
        "open_conflict",
        "open_evidence",
        "focus_map",
        "open_route",
        "show_route_comparison",
        "start_navigation",
        "open_notifications",
    ]
    for act in valid_actions:
        assert act in ACTION_ALLOWLIST

    # Disallowed / malicious actions must fail
    is_valid, err = VoicePermissions.validate_action("delete_database", {}, "ADMIN")
    assert is_valid is False
    assert "allowlist" in err.lower()

    is_valid, err = VoicePermissions.validate_action("eval_javascript", {"code": "alert(1)"}, "ADMIN")
    assert is_valid is False


def test_page_allowlist_and_admin_route_protection():
    """Verify citizen cannot be directed to admin routes via voice actions."""
    # Citizen trying to navigate to /admin/dashboard
    is_valid, err = VoicePermissions.validate_action(
        "navigate_to_page",
        {"path": "/admin/dashboard"},
        user_role="CITIZEN",
    )
    assert is_valid is False
    assert "not authorized" in err.lower() or "access denied" in err.lower()

    # Staff navigating to /admin/dashboard is allowed
    is_valid_staff, _ = VoicePermissions.validate_action(
        "navigate_to_page",
        {"path": "/admin/dashboard"},
        user_role="ADMIN",
    )
    assert is_valid_staff is True

    # Unknown / arbitrary URL is rejected
    is_valid_bad, err_bad = VoicePermissions.validate_action(
        "navigate_to_page",
        {"path": "https://malicious-site.com/exploit"},
        user_role="ADMIN",
    )
    assert is_valid_bad is False
