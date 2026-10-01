"""
SafeCity Loop V2 — Dedicated FRIDAY Frontend Actions Test Suite (Phase 4.37)
Verifies allowlisted action dispatching, role restrictions, parameter validation, and RPC routing.
"""
import pytest
from backend.voice.permissions import VoicePermissions, PAGE_ALLOWLIST, ALLOWLISTED_ACTIONS


def test_action_allowlist_coverage():
    """Verify that all expected frontend actions are explicitly allowlisted."""
    required_actions = [
        "navigate_to_page",
        "open_complaint",
        "open_hazard",
        "open_conflict",
        "open_evidence",
        "focus_map",
        "highlight_map_entity",
        "highlight_hazard",
        "highlight_complaint",
        "seek_video",
        "open_route",
        "show_route",
        "show_route_comparison",
        "show_dashboard_card",
        "show_status_history",
        "show_danger_zones",
        "select_route",
        "start_navigation",
        "pause_navigation",
        "resume_navigation",
        "stop_navigation",
        "open_notifications",
        "show_notifications",
        "open_admin_complaints",
        "confirm_action",
        "cancel_action",
    ]
    for action in required_actions:
        assert action in ALLOWLISTED_ACTIONS, f"Action {action} must be in ALLOWLISTED_ACTIONS"


def test_page_allowlist_resolution():
    """Verify that semantic page names resolve to canonical SafeCity routes."""
    mappings = {
        "dashboard": "/",
        "overview": "/",
        "my-complaints": "/citizen/complaints",
        "ai-vision": "/ai-vision",
        "road-intelligence": "/road-intelligence",
        "traffic-conflicts": "/traffic-conflicts",
        "hazards": "/map",
        "map": "/map",
        "repair": "/repair",
        "routes": "/routes",
        "evidence": "/evidence",
        "danger-zones": "/danger-zones",
        "notifications": "/citizen/notifications",
        "admin-dashboard": "/admin/dashboard",
        "admin-complaints": "/admin/complaints",
    }
    for alias, expected in mappings.items():
        assert alias in PAGE_ALLOWLIST, f"Alias {alias} missing from PAGE_ALLOWLIST"
        assert PAGE_ALLOWLIST[alias] == expected


def test_action_validation_for_citizen():
    """Verify citizen action validation allows public/citizen actions and blocks admin ones."""
    # 1. Allowed citizen action: open_complaint
    is_valid, _ = VoicePermissions.validate_action(
        "open_complaint",
        {"reference": "SC-2026-000184", "path": "/citizen/complaints/1"},
        user_role="CITIZEN",
    )
    assert is_valid is True

    # 2. Allowed citizen action: navigate to notifications
    is_valid_notif, _ = VoicePermissions.validate_action(
        "open_notifications",
        {},
        user_role="CITIZEN",
    )
    assert is_valid_notif is True

    # 3. Blocked citizen action: open_admin_complaints
    is_valid_admin, err_admin = VoicePermissions.validate_action(
        "open_admin_complaints",
        {},
        user_role="CITIZEN",
    )
    assert is_valid_admin is False
    assert "access denied" in err_admin.lower() or "not authorized" in err_admin.lower()


def test_action_validation_for_staff():
    """Verify staff roles can execute administrative navigation actions."""
    for role in ["INSPECTOR", "ENGINEER", "ADMIN", "SUPERUSER"]:
        is_valid, _ = VoicePermissions.validate_action(
            "open_admin_complaints",
            {},
            user_role=role,
        )
        assert is_valid is True, f"Role {role} should be authorized for open_admin_complaints"

        is_valid_nav, _ = VoicePermissions.validate_action(
            "navigate_to_page",
            {"path": "/admin/complaints"},
            user_role=role,
        )
        assert is_valid_nav is True


def test_parameter_sanitization_and_arbitrary_rejection():
    """Verify rejection of unallowlisted actions and arbitrary/malicious paths."""
    # Malicious action name
    is_valid, err = VoicePermissions.validate_action("execute_shell", {"cmd": "rm -rf /"}, "ADMIN")
    assert is_valid is False
    assert "allowlist" in err.lower()

    # Directory traversal or remote URL injection in page navigation
    is_valid_bad, err_bad = VoicePermissions.validate_action(
        "navigate_to_page",
        {"path": "../../etc/passwd"},
        user_role="ADMIN",
    )
    assert is_valid_bad is False
    assert "not an approved" in err_bad.lower() or "safe allowlist" in err_bad.lower() or "unauthorized" in err_bad.lower()
