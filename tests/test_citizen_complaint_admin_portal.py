"""
SafeCity Loop V2 — End-to-End Test Suite for Citizen Accounts,
Complaint State Machine, Data Isolation, RBAC, AI Linking, and Audit Trails.
"""
import pytest
from fastapi.testclient import TestClient
from backend.main import app
from backend.database import SessionLocal, Base, engine
from backend.models import User, Complaint, ComplaintStatus, AuditLog, Notification
from backend.seed import seed_all

client = TestClient(app)


@pytest.fixture(autouse=True)
def setup_db():
    """Ensures database is initialized and seeded before tests."""
    db = SessionLocal()
    try:
        seed_all(db)
    finally:
        db.close()


# ── 1. Authentication Tests ───────────────────────────────────────────────────

def test_auth_registration_and_duplicate_rejection():
    import uuid
    test_email = f"alpha.citizen.{uuid.uuid4().hex[:6]}@example.com"
    # 1. Register new citizen
    res = client.post("/api/auth/register", json={
        "full_name": "Test Citizen Alpha",
        "email": test_email,
        "password": "SecurePassword123!",
        "phone_number": "+91 99999 11111",
    })
    assert res.status_code == 201
    data = res.json()
    assert "access_token" in data
    assert data["user"]["email"] == test_email
    assert data["user"]["role"] == "CITIZEN"

    # 2. Duplicate registration attempt should return 409 Conflict
    dup_res = client.post("/api/auth/register", json={
        "full_name": "Test Citizen Alpha Duplicate",
        "email": test_email,
        "password": "AnotherPassword123!",
    })
    assert dup_res.status_code == 409
    assert "already exists" in dup_res.json()["detail"].lower()


def test_auth_invalid_inputs():
    # Short password
    res = client.post("/api/auth/register", json={
        "full_name": "Short Pwd",
        "email": "short@example.com",
        "password": "short",
    })
    assert res.status_code == 422


def test_auth_login_success_and_failure():
    # Demo citizen login
    login_res = client.post("/api/auth/login", json={
        "email": "citizen_demo@safecity.gov",
        "password": "CitizenPass123!",
    })
    assert login_res.status_code == 200
    token_data = login_res.json()
    assert "access_token" in token_data
    assert token_data["user"]["role"] == "CITIZEN"

    # Wrong password
    bad_login = client.post("/api/auth/login", json={
        "email": "citizen_demo@safecity.gov",
        "password": "WrongPassword!",
    })
    assert bad_login.status_code == 401


def test_auth_me_endpoint():
    # Login as citizen
    login_res = client.post("/api/auth/login", json={
        "email": "citizen_demo@safecity.gov",
        "password": "CitizenPass123!",
    })
    token = login_res.json()["access_token"]

    # Valid token
    me_res = client.get("/api/auth/me", headers={"Authorization": f"Bearer {token}"})
    assert me_res.status_code == 200
    assert me_res.json()["email"] == "citizen_demo@safecity.gov"

    # No token
    no_token_res = client.get("/api/auth/me")
    assert no_token_res.status_code == 401

    # Invalid token
    bad_token_res = client.get("/api/auth/me", headers={"Authorization": "Bearer invalid.jwt.token"})
    assert bad_token_res.status_code == 401


# ── 2. Citizen Data Isolation & Permissions ───────────────────────────────────

def test_citizen_data_isolation():
    """
    MANDATORY SECURITY TEST:
    Citizen A creates Complaint A.
    Citizen B MUST NOT be able to view, edit, or list Complaint A.
    """
    # Register Citizen A
    import uuid
    uid = uuid.uuid4().hex[:6]
    res_a = client.post("/api/auth/register", json={
        "full_name": "Citizen Alice",
        "email": f"alice_{uid}@isolation.test",
        "password": "AlicePassword123!",
    })
    token_a = res_a.json()["access_token"]

    # Register Citizen B
    res_b = client.post("/api/auth/register", json={
        "full_name": "Citizen Bob",
        "email": f"bob_{uid}@isolation.test",
        "password": "BobPassword123!",
    })
    token_b = res_b.json()["access_token"]

    # Alice files complaint
    complaint_res = client.post(
        "/api/complaints",
        headers={"Authorization": f"Bearer {token_a}"},
        json={
            "category_id": 1,
            "title": "Alice Pothole on School Road",
            "description": "Deep hole near school gate.",
            "location_lat": 51.5074,
            "location_lng": -0.1278,
            "location_address": "School Road 12",
        }
    )
    assert complaint_res.status_code == 201
    complaint_a = complaint_res.json()
    complaint_a_id = complaint_a["id"]
    assert complaint_a["public_reference"].startswith("SC-")

    # Bob tries to view Alice's complaint directly -> MUST BE 403 FORBIDDEN
    bob_view = client.get(
        f"/api/complaints/{complaint_a_id}",
        headers={"Authorization": f"Bearer {token_b}"},
    )
    assert bob_view.status_code == 403

    # Bob views his complaints list -> Alice's complaint MUST NOT appear
    bob_list = client.get(
        "/api/complaints/my",
        headers={"Authorization": f"Bearer {token_b}"},
    )
    assert bob_list.status_code == 200
    bob_complaint_ids = [c["id"] for c in bob_list.json()]
    assert complaint_a_id not in bob_complaint_ids

    # Bob tries to confirm or reopen Alice's complaint -> MUST BE 403 FORBIDDEN
    bob_confirm = client.post(
        f"/api/complaints/{complaint_a_id}/citizen-confirm",
        headers={"Authorization": f"Bearer {token_b}"},
        json={"confirmed": True}
    )
    assert bob_confirm.status_code == 403

    # Alice CAN view her own complaint
    alice_view = client.get(
        f"/api/complaints/{complaint_a_id}",
        headers={"Authorization": f"Bearer {token_a}"},
    )
    assert alice_view.status_code == 200
    assert alice_view.json()["title"] == "Alice Pothole on School Road"


def test_role_based_endpoint_protection():
    # Citizen cannot access admin complaint table
    citizen_login = client.post("/api/auth/login", json={
        "email": "citizen_demo@safecity.gov",
        "password": "CitizenPass123!",
    })
    citizen_token = citizen_login.json()["access_token"]

    res = client.get(
        "/api/admin/complaints",
        headers={"Authorization": f"Bearer {citizen_token}"},
    )
    assert res.status_code == 403

    # Citizen cannot access user management
    user_mgmt_res = client.get(
        "/api/admin/users",
        headers={"Authorization": f"Bearer {citizen_token}"},
    )
    assert user_mgmt_res.status_code == 403

    # Admin CAN access both
    admin_login = client.post("/api/auth/login", json={
        "email": "admin_demo@safecity.gov",
        "password": "AdminPass123!",
    })
    admin_token = admin_login.json()["access_token"]

    admin_res = client.get(
        "/api/admin/complaints",
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert admin_res.status_code == 200

    admin_users_res = client.get(
        "/api/admin/users",
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert admin_users_res.status_code == 200


# ── 3. Complaint Lifecycle & State Machine ───────────────────────────────────

def test_complaint_full_lifecycle():
    # 1. Citizen creates complaint
    citizen_login = client.post("/api/auth/login", json={
        "email": "citizen_demo@safecity.gov",
        "password": "CitizenPass123!",
    })
    c_token = citizen_login.json()["access_token"]

    create_res = client.post(
        "/api/complaints",
        headers={"Authorization": f"Bearer {c_token}"},
        json={
            "category_id": 1,
            "title": "Lifecycle Test Pothole",
            "description": "Hazardous road depression for lifecycle verification.",
            "location_lat": 51.5074,
            "location_lng": -0.1278,
            "priority": "HIGH",
        }
    )
    assert create_res.status_code == 201
    complaint = create_res.json()
    cid = complaint["id"]
    assert complaint["status"] == "SUBMITTED"

    # 2. Inspector logs in and moves SUBMITTED -> UNDER_REVIEW
    insp_login = client.post("/api/auth/login", json={
        "email": "inspector_demo@safecity.gov",
        "password": "InspectorPass123!",
    })
    i_token = insp_login.json()["access_token"]

    review_res = client.post(
        f"/api/admin/complaints/{cid}/status",
        headers={"Authorization": f"Bearer {i_token}"},
        json={"new_status": "UNDER_REVIEW", "comment": "Officer reviewing field photos."}
    )
    assert review_res.status_code == 200
    assert review_res.json()["status"] == "UNDER_REVIEW"

    # 3. Invalid transition test: UNDER_REVIEW -> CLOSED (not allowed directly)
    bad_trans = client.post(
        f"/api/admin/complaints/{cid}/status",
        headers={"Authorization": f"Bearer {i_token}"},
        json={"new_status": "CLOSED", "comment": "Attempting invalid jump."}
    )
    assert bad_trans.status_code == 409

    # 4. Inspector verifies complaint: UNDER_REVIEW -> VERIFIED
    verify_res = client.post(
        f"/api/admin/complaints/{cid}/status",
        headers={"Authorization": f"Bearer {i_token}"},
        json={"new_status": "VERIFIED", "comment": "Confirmed on-site."}
    )
    assert verify_res.status_code == 200
    assert verify_res.json()["status"] == "VERIFIED"

    # 5. Admin assigns complaint: -> ASSIGNED
    admin_login = client.post("/api/auth/login", json={
        "email": "admin_demo@safecity.gov",
        "password": "AdminPass123!",
    })
    a_token = admin_login.json()["access_token"]

    assign_res = client.post(
        f"/api/admin/complaints/{cid}/assign",
        headers={"Authorization": f"Bearer {a_token}"},
        json={"assigned_department": "Public Works Rapid Asphalt", "notes": "High priority school road repair."}
    )
    assert assign_res.status_code == 200
    assert assign_res.json()["status"] == "ASSIGNED"

    # 6. Engineer starts work: ASSIGNED -> IN_PROGRESS
    eng_login = client.post("/api/auth/login", json={
        "email": "engineer_demo@safecity.gov",
        "password": "EngineerPass123!",
    })
    e_token = eng_login.json()["access_token"]

    prog_res = client.post(
        f"/api/admin/complaints/{cid}/status",
        headers={"Authorization": f"Bearer {e_token}"},
        json={"new_status": "IN_PROGRESS", "comment": "Crew on site."}
    )
    assert prog_res.status_code == 200
    assert prog_res.json()["status"] == "IN_PROGRESS"

    # 7. Engineer completes work: IN_PROGRESS -> RESOLVED
    res_res = client.post(
        f"/api/admin/complaints/{cid}/status",
        headers={"Authorization": f"Bearer {e_token}"},
        json={"new_status": "RESOLVED", "comment": "Asphalt patch completed."}
    )
    assert res_res.status_code == 200
    assert res_res.json()["status"] == "RESOLVED"
    assert res_res.json()["resolved_at"] is not None

    # 8. Admin adds public response and internal note
    client.post(
        f"/api/admin/complaints/{cid}/public-response",
        headers={"Authorization": f"Bearer {a_token}"},
        json={"public_response": "Repairs complete. Pothole filled and sealed."}
    )
    client.post(
        f"/api/admin/complaints/{cid}/internal-note",
        headers={"Authorization": f"Bearer {a_token}"},
        json={"internal_note": "Contractor invoice ref #INV-9921 approved."}
    )

    # 9. Citizen checks complaint details
    # STRICT PRIVACY: Internal note MUST NOT appear in citizen response!
    citizen_detail = client.get(
        f"/api/complaints/{cid}",
        headers={"Authorization": f"Bearer {c_token}"},
    )
    assert citizen_detail.status_code == 200
    c_data = citizen_detail.json()
    assert c_data["status"] == "RESOLVED"
    assert c_data["public_response"] == "Repairs complete. Pothole filled and sealed."
    assert "internal_notes" not in c_data
    assert c_data["can_confirm_resolution"] is True

    # 10. Citizen confirms resolution: RESOLVED -> CLOSED
    confirm_res = client.post(
        f"/api/complaints/{cid}/citizen-confirm",
        headers={"Authorization": f"Bearer {c_token}"},
        json={"confirmed": True, "comment": "Verified road is smooth now. Thank you!"}
    )
    assert confirm_res.status_code == 200
    assert confirm_res.json()["status"] == "CLOSED"
    assert confirm_res.json()["closed_at"] is not None

    # 11. Timeline check: ensure all steps are recorded
    timeline_res = client.get(
        f"/api/complaints/{cid}/timeline",
        headers={"Authorization": f"Bearer {c_token}"},
    )
    assert timeline_res.status_code == 200
    timeline = timeline_res.json()
    statuses = [t["new_status"] for t in timeline]
    assert "SUBMITTED" in statuses
    assert "UNDER_REVIEW" in statuses
    assert "VERIFIED" in statuses
    assert "ASSIGNED" in statuses
    assert "IN_PROGRESS" in statuses
    assert "RESOLVED" in statuses
    assert "CLOSED" in statuses


# ── 4. AI Hazard Linking & Creation ──────────────────────────────────────────

def test_complaint_ai_hazard_linking():
    admin_login = client.post("/api/auth/login", json={
        "email": "admin_demo@safecity.gov",
        "password": "AdminPass123!",
    })
    a_token = admin_login.json()["access_token"]

    # Complaint 1 is seeded at School Road coordinates (51.5074, -0.1278)
    complaint_res = client.get(
        "/api/admin/complaints/1",
        headers={"Authorization": f"Bearer {a_token}"},
    )
    assert complaint_res.status_code == 200
    data = complaint_res.json()
    assert "ai_hazard_candidates" in data
    assert len(data["ai_hazard_candidates"]) > 0

    # Promote complaint to new Hazard
    create_hazard_res = client.post(
        "/api/admin/complaints/1/create-hazard",
        headers={"Authorization": f"Bearer {a_token}"},
        json={"road_id": 1, "severity": "HIGH", "notes": "Promoted to AI hazard tracking"}
    )
    assert create_hazard_res.status_code == 200
    comp_out = create_hazard_res.json()
    assert comp_out["ai_linked"] is True
    assert comp_out["hazard_id"] is not None


# ── 5. User Management & Last-Superuser Safety ───────────────────────────────

def test_last_superuser_safety():
    """
    MANDATORY SAFETY TEST:
    System must prevent demoting or deactivating the last active superuser.
    """
    superuser_login = client.post("/api/auth/login", json={
        "email": "superuser_demo@safecity.gov",
        "password": "SuperPass123!",
    })
    s_token = superuser_login.json()["access_token"]

    # Fetch superuser id
    me = client.get("/api/auth/me", headers={"Authorization": f"Bearer {s_token}"}).json()
    su_id = me["id"]

    # Try to demote the only superuser to CITIZEN -> MUST BE REJECTED
    demote_res = client.patch(
        f"/api/admin/users/{su_id}/role",
        headers={"Authorization": f"Bearer {s_token}"},
        json={"role": "CITIZEN"}
    )
    assert demote_res.status_code == 400
    assert "last active superuser" in demote_res.json()["detail"].lower()

    # Try to deactivate the only superuser -> MUST BE REJECTED
    deactivate_res = client.patch(
        f"/api/admin/users/{su_id}/status",
        headers={"Authorization": f"Bearer {s_token}"},
        json={"is_active": False}
    )
    assert deactivate_res.status_code == 400
    assert "last active superuser" in deactivate_res.json()["detail"].lower()


# ── 6. In-App Notifications & Audit Trail ─────────────────────────────────────

def test_notifications_and_audit():
    c_login = client.post("/api/auth/login", json={
        "email": "citizen_demo@safecity.gov",
        "password": "CitizenPass123!",
    })
    c_token = c_login.json()["access_token"]

    # Check citizen notifications
    notifs_res = client.get("/api/notifications", headers={"Authorization": f"Bearer {c_token}"})
    assert notifs_res.status_code == 200
    notifs = notifs_res.json()
    assert len(notifs) >= 1

    # Check unread count
    count_res = client.get("/api/notifications/unread-count", headers={"Authorization": f"Bearer {c_token}"})
    assert count_res.status_code == 200
    assert count_res.json()["unread_count"] >= 0

    # Mark all read
    mark_res = client.post("/api/notifications/read-all", headers={"Authorization": f"Bearer {c_token}"})
    assert mark_res.status_code == 200

    count_after = client.get("/api/notifications/unread-count", headers={"Authorization": f"Bearer {c_token}"})
    assert count_after.json()["unread_count"] == 0

    # Verify tamper-evident AuditLog entries exist in DB
    db = SessionLocal()
    try:
        audit_count = db.query(AuditLog).count()
        assert audit_count >= 5
    finally:
        db.close()
