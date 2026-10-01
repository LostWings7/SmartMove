"""
SafeCity Loop V2 — Dedicated FRIDAY Voice Tools Test Suite (Phase 4.37)
Verifies execution, metadata, return schemas, rate-limiting, and telemetry for all 32 voice tools.
"""
import pytest
from backend.database import SessionLocal
from backend.voice.context import SafeCityContext
from backend.voice.permissions import TOOL_METADATA
from backend.voice.tools import (
    safecity_voice_tools,
    get_tool_registry,
    record_tool_telemetry,
    get_tool_telemetry,
    check_rate_limit,
)


@pytest.fixture(scope="module")
def db_session():
    db = SessionLocal()
    yield db
    db.close()


@pytest.fixture
def citizen_context():
    return SafeCityContext(
        client_id="test_tools_citizen",
        user_role="CITIZEN",
        is_authenticated=True,
        current_user={"id": 1, "full_name": "Test Citizen", "role": "CITIZEN"},
    )


@pytest.fixture
def admin_context():
    return SafeCityContext(
        client_id="test_tools_admin",
        user_role="ADMIN",
        is_authenticated=True,
        current_user={"id": 2, "full_name": "Chief Engineer", "role": "ADMIN"},
    )


def test_tool_registry_completeness():
    """Verify that all 32 tools are registered and exposed to LiveKit agents."""
    registry = get_tool_registry()
    assert len(registry) == 33, f"Expected 33 tools, got {len(registry)}"
    assert len(safecity_voice_tools) == 33, f"Expected 33 decorated tools, got {len(safecity_voice_tools)}"

    for tool_name in TOOL_METADATA:
        assert tool_name in registry, f"Tool {tool_name} missing from registry"


def test_tool_metadata_declarations():
    """Verify that all tools declare name, description, required_role, is_write, and requires_confirmation."""
    for tool_name, meta in TOOL_METADATA.items():
        assert meta["name"] == tool_name
        assert len(meta["description"]) > 10, f"Tool {tool_name} description too short"
        assert meta["required_role"] in ["CITIZEN", "STAFF", "INSPECTOR", "ENGINEER", "ADMIN", "SUPERUSER"]
        assert isinstance(meta["is_write"], bool)
        assert isinstance(meta["requires_confirmation"], bool)

        # Mutating tools must be declared correctly
        if meta["requires_confirmation"]:
            assert meta["is_write"] is True, f"Tool {tool_name} requires confirmation but is not marked as write"


def test_telemetry_and_rate_limiting():
    """Verify tool execution telemetry and rate-limiting enforcement."""
    record_tool_telemetry("calculate_routes", "session_test", 14.5, True)
    record_tool_telemetry("file_complaint", "session_test", 35.2, True)

    telemetry = get_tool_telemetry()
    assert isinstance(telemetry, list)
    assert len(telemetry) >= 2
    assert any(entry["tool"] == "calculate_routes" for entry in telemetry)
    assert any(entry["tool"] == "file_complaint" for entry in telemetry)

    # Rate limiting test
    rate_ok = check_rate_limit("rate_sess", "file_complaint", max_per_minute=2)
    assert rate_ok is True
    rate_ok2 = check_rate_limit("rate_sess", "file_complaint", max_per_minute=2)
    assert rate_ok2 is True
    rate_exceeded = check_rate_limit("rate_sess", "file_complaint", max_per_minute=2)
    assert rate_exceeded is False, "Rate limit should block 3rd call within 1 minute"


def test_read_tools_execution(citizen_context, db_session):
    """Verify that standard SafeCity read tools execute cleanly with authoritative summaries."""
    registry = get_tool_registry()

    # 1. get_current_page
    p_res = registry["get_current_page"](context=citizen_context)
    assert p_res["success"] is True
    assert "voice_summary" in p_res

    # 2. get_current_user
    u_res = registry["get_current_user"](context=citizen_context)
    assert u_res["success"] is True
    assert "Test Citizen" in u_res["voice_summary"]

    # 3. get_dashboard_summary
    d_res = registry["get_dashboard_summary"](db=db_session)
    assert d_res["roads_monitored"] > 0
    assert "citywide overview" in d_res["voice_summary"].lower()

    # 4. get_active_hazards
    h_res = registry["get_active_hazards"](limit=3, db=db_session)
    assert "hazards" in h_res
    assert "voice_summary" in h_res

    # 5. get_traffic_conflicts
    c_res = registry["get_traffic_conflicts"](limit=2, db=db_session)
    assert "conflicts" in c_res
    assert "voice_summary" in c_res

    # 6. get_user_notifications
    n_res = registry["get_user_notifications"](context=citizen_context, db=db_session)
    assert n_res["success"] is True
    assert "voice_summary" in n_res


def test_cross_domain_intelligence_tool(citizen_context, db_session):
    """Verify Tool 32 cross-domain correlation across subsystems."""
    registry = get_tool_registry()
    cd_res = registry["get_cross_domain_intelligence"](
        complaint_reference="SC-2026-000184",
        road_name="School Road",
        db=db_session,
    )
    assert cd_res["success"] is True
    assert "School Road" in cd_res["road_name"]
    assert "voice_summary" in cd_res
    assert "danger_zone_classification" in cd_res
    assert "conflict_count" in cd_res
