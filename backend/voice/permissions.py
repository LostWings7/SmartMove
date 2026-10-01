"""
SafeCity Loop V2 — Voice Assistant Permissions & RBAC Guard
Enforces role-aware data access, complaint visibility rules, and strict
allowlisted frontend action dispatch validation.
"""
from typing import Optional, Tuple, Dict, Any, Set
from backend.models import UserRole


STAFF_ROLES: Set[str] = {
    UserRole.INSPECTOR,
    UserRole.ENGINEER,
    UserRole.ADMIN,
    UserRole.SUPERUSER,
}

ALLOWLISTED_ACTIONS: Set[str] = {
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
    "seek_video_to_event",
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
    "read_navigation_directions",
    "read_next_navigation_direction",
    "stop_navigation_speech",
    "open_notifications",
    "show_notifications",
    "open_admin_complaints",
    "confirm_action",
    "cancel_action",
}

PAGE_ALLOWLIST: Dict[str, str] = {
    "dashboard": "/",
    "overview": "/",
    "/": "/",
    "my-complaints": "/citizen/complaints",
    "complaints": "/citizen/complaints",
    "/citizen/complaints": "/citizen/complaints",
    "new-complaint": "/citizen/complaints/new",
    "/citizen/complaints/new": "/citizen/complaints/new",
    "complaint-details": "/citizen/complaints",
    "notifications": "/citizen/notifications",
    "/citizen/notifications": "/citizen/notifications",
    "ai-vision": "/ai-vision",
    "/ai-vision": "/ai-vision",
    "road-intelligence": "/road-intelligence",
    "/road-intelligence": "/road-intelligence",
    "traffic-conflicts": "/traffic-conflicts",
    "/traffic-conflicts": "/traffic-conflicts",
    "hazards": "/map",
    "map": "/map",
    "/map": "/map",
    "repairs": "/repair",
    "repair": "/repair",
    "/repair": "/repair",
    "reports": "/reports",
    "/reports": "/reports",
    "danger-zones": "/danger-zones",
    "/danger-zones": "/danger-zones",
    "routes": "/routes",
    "/routes": "/routes",
    "evidence": "/evidence",
    "/evidence": "/evidence",
    "junction": "/junction",
    "/junction": "/junction",
    "analytics": "/analytics",
    "/analytics": "/analytics",
    "settings": "/settings",
    "/settings": "/settings",
    "admin-dashboard": "/admin/dashboard",
    "/admin/dashboard": "/admin/dashboard",
    "admin-complaints": "/admin/complaints",
    "/admin/complaints": "/admin/complaints",
    "admin-users": "/admin/users",
    "/admin/users": "/admin/users",
}

ADMIN_ONLY_PAGES: Set[str] = {
    "/admin/dashboard",
    "/admin/complaints",
    "/admin/users",
}

TOOL_METADATA: Dict[str, Dict[str, Any]] = {
    # ── Read Tools (Public / Citizen Safe) ──
    "get_current_user": {
        "name": "get_current_user",
        "description": "Returns identity, role, and auth status for current user",
        "required_role": "CITIZEN",
        "is_write": False,
        "requires_confirmation": False,
    },
    "get_current_page": {
        "name": "get_current_page",
        "description": "Identifies active screen, route, and focused element in UI",
        "required_role": "CITIZEN",
        "is_write": False,
        "requires_confirmation": False,
    },
    "get_dashboard_summary": {
        "name": "get_dashboard_summary",
        "description": "Citywide KPIs: monitored roads, active hazards, conflicts, repairs",
        "required_role": "CITIZEN",
        "is_write": False,
        "requires_confirmation": False,
    },
    "get_my_complaints": {
        "name": "get_my_complaints",
        "description": "Retrieves filed citizen complaints for current user",
        "required_role": "CITIZEN",
        "is_write": False,
        "requires_confirmation": False,
    },
    "get_complaint_details": {
        "name": "get_complaint_details",
        "description": "Full complaint record by public reference with strict RBAC isolation",
        "required_role": "CITIZEN",
        "is_write": False,
        "requires_confirmation": False,
    },
    "get_complaint_status": {
        "name": "get_complaint_status",
        "description": "Concise lifecycle status and department for a complaint",
        "required_role": "CITIZEN",
        "is_write": False,
        "requires_confirmation": False,
    },
    "get_active_hazards": {
        "name": "get_active_hazards",
        "description": "Active road surface defects (potholes, ponding, cracks)",
        "required_role": "CITIZEN",
        "is_write": False,
        "requires_confirmation": False,
    },
    "get_hazard_details": {
        "name": "get_hazard_details",
        "description": "Detailed hazard telemetry, confidence, and severity by ID",
        "required_role": "CITIZEN",
        "is_write": False,
        "requires_confirmation": False,
    },
    "get_traffic_summary": {
        "name": "get_traffic_summary",
        "description": "Surrogate traffic safety metrics and monitored junctions",
        "required_role": "CITIZEN",
        "is_write": False,
        "requires_confirmation": False,
    },
    "get_traffic_conflicts": {
        "name": "get_traffic_conflicts",
        "description": "Surrogate near-miss events with TTC and clearance",
        "required_role": "CITIZEN",
        "is_write": False,
        "requires_confirmation": False,
    },
    "get_conflict_details": {
        "name": "get_conflict_details",
        "description": "Detailed kinematics and collision vector for conflict ID",
        "required_role": "CITIZEN",
        "is_write": False,
        "requires_confirmation": False,
    },
    "get_road_conditions": {
        "name": "get_road_conditions",
        "description": "Surface quality and risk scores for named road segment",
        "required_role": "CITIZEN",
        "is_write": False,
        "requires_confirmation": False,
    },
    "get_repair_status": {
        "name": "get_repair_status",
        "description": "Municipal repair work orders and engineering status",
        "required_role": "CITIZEN",
        "is_write": False,
        "requires_confirmation": False,
    },
    "get_route_summary": {
        "name": "get_route_summary",
        "description": "Fastest Route vs. Lower-Risk Safer Route comparison",
        "required_role": "CITIZEN",
        "is_write": False,
        "requires_confirmation": False,
    },
    "get_danger_zones": {
        "name": "get_danger_zones",
        "description": "High-concern corridors and contributing risk factors",
        "required_role": "CITIZEN",
        "is_write": False,
        "requires_confirmation": False,
    },
    "calculate_routes": {
        "name": "calculate_routes",
        "description": "Calculates Fastest, Balanced, and Safer route alternatives",
        "required_role": "CITIZEN",
        "is_write": False,
        "requires_confirmation": False,
    },
    "get_route_details": {
        "name": "get_route_details",
        "description": "Maneuvers, distance, and risk score for a specific route option",
        "required_role": "CITIZEN",
        "is_write": False,
        "requires_confirmation": False,
    },
    "compare_routes": {
        "name": "compare_routes",
        "description": "Compares 3 route alternatives using RouteEngine",
        "required_role": "CITIZEN",
        "is_write": False,
        "requires_confirmation": False,
    },
    "select_route": {
        "name": "select_route",
        "description": "Selects a route alternative for display or pending navigation",
        "required_role": "CITIZEN",
        "is_write": False,
        "requires_confirmation": False,
    },
    "get_navigation_state": {
        "name": "get_navigation_state",
        "description": "Retrieves ETA, remaining distance, maneuvers, and risk state",
        "required_role": "CITIZEN",
        "is_write": False,
        "requires_confirmation": False,
    },
    "get_next_maneuver": {
        "name": "get_next_maneuver",
        "description": "Retrieves immediate next driving instruction and distance",
        "required_role": "CITIZEN",
        "is_write": False,
        "requires_confirmation": False,
    },
    "read_navigation_directions": {
        "name": "read_navigation_directions",
        "description": "Verbatim recitation of displayed turn-by-turn navigation clues",
        "required_role": "CITIZEN",
        "is_write": False,
        "requires_confirmation": False,
    },
    "get_user_notifications": {
        "name": "get_user_notifications",
        "description": "Retrieves unread and recent in-app notifications for caller",
        "required_role": "CITIZEN",
        "is_write": False,
        "requires_confirmation": False,
    },
    "get_cross_domain_intelligence": {
        "name": "get_cross_domain_intelligence",
        "description": "Correlates Complaint -> AI Hazard -> Evidence -> Road Risk -> Traffic Conflicts",
        "required_role": "CITIZEN",
        "is_write": False,
        "requires_confirmation": False,
    },

    # ── Staff / Administrative Read Tools ──
    "get_unresolved_complaints": {
        "name": "get_unresolved_complaints",
        "description": "Queries municipal queue for unresolved, pending, or overdue complaints",
        "required_role": "STAFF",
        "is_write": False,
        "requires_confirmation": False,
    },

    # ── Write / State-Changing Navigation Tools ──
    "start_navigation": {
        "name": "start_navigation",
        "description": "Activates Safe Navigation Mode after explicit user opt-in",
        "required_role": "CITIZEN",
        "is_write": True,
        "requires_confirmation": True,
    },
    "pause_navigation": {
        "name": "pause_navigation",
        "description": "Pauses active turn-by-turn guidance",
        "required_role": "CITIZEN",
        "is_write": True,
        "requires_confirmation": False,
    },
    "resume_navigation": {
        "name": "resume_navigation",
        "description": "Resumes active turn-by-turn guidance",
        "required_role": "CITIZEN",
        "is_write": True,
        "requires_confirmation": False,
    },
    "stop_navigation": {
        "name": "stop_navigation",
        "description": "Stops or cancels active navigation session",
        "required_role": "CITIZEN",
        "is_write": True,
        "requires_confirmation": False,
    },

    # ── Citizen Write Tools (Require Confirmation) ──
    "file_complaint": {
        "name": "file_complaint",
        "description": "Files a new road hazard complaint with category, description, and location",
        "required_role": "CITIZEN",
        "is_write": True,
        "requires_confirmation": True,
    },
    "reopen_complaint": {
        "name": "reopen_complaint",
        "description": "Reopens a resolved or closed complaint for municipal reconsideration",
        "required_role": "CITIZEN",
        "is_write": True,
        "requires_confirmation": True,
    },

    # ── Staff / Admin Write Tools (Consequential Operations) ──
    "assign_complaint_to_department": {
        "name": "assign_complaint_to_department",
        "description": "Assigns municipal complaint to operational department or engineer",
        "required_role": "STAFF",
        "is_write": True,
        "requires_confirmation": True,
    },
    "update_complaint_status_admin": {
        "name": "update_complaint_status_admin",
        "description": "Transitions complaint status (reject, verify, resolve, close)",
        "required_role": "STAFF",
        "is_write": True,
        "requires_confirmation": True,
    },
}


class VoicePermissions:
    """RBAC validation suite for FRIDAY voice agent operations."""

    @staticmethod
    def is_staff(role: str) -> bool:
        return (role or "").upper() in STAFF_ROLES

    @staticmethod
    def can_view_complaint(user_role: str, user_id: Optional[int], complaint_citizen_id: int) -> bool:
        """
        Determines if a user has permission to view a specific complaint:
        - Municipal staff (Inspector, Engineer, Admin, Superuser) can view all complaints.
        - Citizens can ONLY view complaints they personally submitted.
        - Anonymous guests cannot view private citizen complaints.
        """
        role = (user_role or "").upper()
        if role in STAFF_ROLES:
            return True
        if role == UserRole.CITIZEN and user_id is not None:
            return user_id == complaint_citizen_id
        return False

    @staticmethod
    def can_view_admin_metrics(user_role: str) -> bool:
        return (user_role or "").upper() in STAFF_ROLES

    @staticmethod
    def can_access_audit_ledger(user_role: str) -> bool:
        return (user_role or "").upper() in {UserRole.ADMIN, UserRole.SUPERUSER}

    @classmethod
    def get_tool_metadata(cls, tool_name: str) -> Optional[Dict[str, Any]]:
        return TOOL_METADATA.get(tool_name)

    @classmethod
    def is_write_tool(cls, tool_name: str) -> bool:
        meta = TOOL_METADATA.get(tool_name)
        return bool(meta and meta.get("is_write", False))

    @classmethod
    def requires_confirmation(cls, tool_name: str) -> bool:
        meta = TOOL_METADATA.get(tool_name)
        return bool(meta and meta.get("requires_confirmation", False))

    @classmethod
    def can_execute_tool(cls, tool_name: str, user_role: str) -> Tuple[bool, Optional[str]]:
        """
        Validates caller role against tool's declared permission requirements:
        - Tools requiring 'CITIZEN' are accessible by all authenticated or guest users.
        - Tools requiring 'STAFF' require INSPECTOR, ENGINEER, ADMIN, or SUPERUSER.
        - Tools requiring 'ADMIN' require ADMIN or SUPERUSER.
        """
        meta = TOOL_METADATA.get(tool_name)
        if not meta:
            return True, None

        required = meta.get("required_role", "CITIZEN").upper()
        role = (user_role or "CITIZEN").upper()

        if required in ["STAFF", "INSPECTOR", "ENGINEER"] and role not in STAFF_ROLES:
            return False, f"Access Denied: Citizens cannot execute administrative tool '{tool_name}'."

        if required in ["ADMIN", "SUPERUSER"] and role not in {UserRole.ADMIN, UserRole.SUPERUSER}:
            return False, f"Access Denied: Administrative privileges required to execute tool '{tool_name}'."

        return True, None

    @classmethod
    def resolve_page_path(cls, page_identifier: str) -> Optional[str]:
        """Maps semantic or relative page name to normalized canonical route path."""
        clean = (page_identifier or "").strip().lower()
        if clean in PAGE_ALLOWLIST:
            return PAGE_ALLOWLIST[clean]
        # Check if direct valid path
        if clean.startswith("/citizen/complaints/") or clean.startswith("/admin/complaints/"):
            return clean
        return None

    @classmethod
    def validate_page(cls, page_identifier: str) -> Optional[str]:
        """Validates and resolves a page identifier against the allowlist."""
        return cls.resolve_page_path(page_identifier)

    @classmethod
    def can_access_page(cls, user_role: str, page_identifier: str) -> bool:
        """Determines if the given role is permitted to access the target page."""
        resolved = cls.resolve_page_path(page_identifier)
        if not resolved:
            return False
        if resolved in ADMIN_ONLY_PAGES:
            return cls.is_staff(user_role)
        return True

    @classmethod
    def is_allowlisted_action(cls, action_name: str) -> bool:
        """Checks if an action name is in the strictly permitted ALLOWLISTED_ACTIONS set."""
        return action_name in ALLOWLISTED_ACTIONS

    @classmethod
    def validate_action_arguments(cls, action_name: str, payload: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """Strictly validates action-specific payload arguments."""
        if action_name == "focus_map":
            if "lat" not in payload or "lng" not in payload:
                return False, "Action 'focus_map' requires 'lat' and 'lng' arguments."
            try:
                lat = float(payload["lat"])
                lng = float(payload["lng"])
                if not (-90.0 <= lat <= 90.0):
                    return False, f"Latitude {lat} out of valid range [-90, 90]."
                if not (-180.0 <= lng <= 180.0):
                    return False, f"Longitude {lng} out of valid range [-180, 180]."
            except (ValueError, TypeError):
                return False, "Coordinates 'lat' and 'lng' must be valid numbers."

        elif action_name in ["seek_video", "seek_video_to_event"]:
            val = payload.get("timestamp_seconds")
            if val is None:
                val = payload.get("seconds")
            if val is None:
                return False, "Action 'seek_video' requires 'timestamp_seconds' argument."
            try:
                sec = float(val)
                if sec < 0:
                    return False, f"Video seek timestamp cannot be negative: {sec}"
            except (ValueError, TypeError):
                return False, "Timestamp must be a numeric second offset."

        elif action_name == "open_hazard":
            val = payload.get("id") or payload.get("hazard_id")
            if val is None:
                return False, "Action 'open_hazard' requires 'id' argument."
            try:
                h_id = int(val)
                if h_id <= 0:
                    return False, f"Hazard ID must be positive: {h_id}"
            except (ValueError, TypeError):
                return False, "Hazard ID must be an integer."

        elif action_name == "open_conflict":
            val = payload.get("id") or payload.get("conflict_id")
            ttc = payload.get("ttc")
            if val is None and ttc is None:
                return False, "Action 'open_conflict' requires 'id' or 'ttc' argument."
            if val is not None:
                try:
                    c_id = int(val)
                    if c_id <= 0:
                        return False, f"Conflict ID must be positive: {c_id}"
                except (ValueError, TypeError):
                    return False, "Conflict ID must be an integer."

        elif action_name == "open_complaint":
            ref = payload.get("reference")
            c_id = payload.get("id")
            if not ref and not c_id:
                return False, "Action 'open_complaint' requires 'reference' or 'id' argument."

        elif action_name in {"read_navigation_directions", "read_next_navigation_direction"}:
            if "start_index" in payload and payload["start_index"] is not None:
                try:
                    idx = int(payload["start_index"])
                    if idx < 0:
                        return False, "start_index must be greater than or equal to 0."
                except (ValueError, TypeError):
                    return False, "start_index must be an integer."

        return True, None

    @classmethod
    def validate_action(
        cls,
        action_name: str,
        payload: Dict[str, Any],
        user_role: str,
        user_id: Optional[int] = None
    ) -> Tuple[bool, Optional[str]]:
        """
        Validates whether a requested frontend action is authorized:
        1. Action must exist in the strict ALLOWLISTED_ACTIONS set.
        2. Prohibits arbitrary DOM, eval, click(x, y), or type(x, y).
        3. Validates required arguments per action.
        4. Enforces page-level RBAC restrictions.
        """
        if not cls.is_allowlisted_action(action_name):
            return False, f"Action '{action_name}' is not in the safe allowlist. Arbitrary browser automation is strictly prohibited."

        role = (user_role or "CITIZEN").upper()

        # Enforce page navigation role boundaries & allowlist
        if action_name == "navigate_to_page":
            target_page = payload.get("path") or payload.get("page", "")
            resolved = cls.resolve_page_path(target_page)
            if not resolved:
                return False, f"Target page '{target_page}' is not an approved SafeCity route."
            if resolved in ADMIN_ONLY_PAGES and role not in STAFF_ROLES:
                return False, f"Citizens are not authorized to navigate to administrative console '{resolved}'."

        if action_name == "open_admin_complaints" and role not in STAFF_ROLES:
            return False, "Access Denied: Citizens are not authorized to access administrative complaint queues."

        # Validate arguments using validate_action_arguments
        arg_ok, arg_err = cls.validate_action_arguments(action_name, payload)
        if not arg_ok:
            return False, arg_err

        return True, None
